//! `hf-hub` utils

use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use hf_hub::{
    HFClientSync, HFError, HFRepositorySync, HFResult, RepoType, RepoTypeDataset, RepoTypeModel,
};
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// Blocking hub client
#[derive(Clone, Debug)]
pub struct Api(HFClientSync);

impl Api {
    /// Client configured from env (`HF_TOKEN`, `HF_HUB_CACHE`, etc)
    pub fn new() -> HFResult<Self> {
        Ok(Self(HFClientSync::new()?))
    }

    pub fn with_cache_dir<P: Into<PathBuf>>(cache_dir: P) -> HFResult<Self> {
        let client = hf_hub::HFClient::builder().cache_dir(cache_dir).build()?;
        Ok(Self(HFClientSync::from_inner(client)?))
    }

    /// Creates a blocking handle for a model repository
    pub fn model(&self, id: impl AsRef<str>) -> Repo<RepoTypeModel> {
        let (owner, name) = hf_hub::split_id(id.as_ref());
        Repo::new(self.0.model(owner, name))
    }

    /// Creates a blocking handle for a dataset repository
    pub fn dataset(&self, id: impl AsRef<str>) -> Repo<RepoTypeDataset> {
        let (owner, name) = hf_hub::split_id(id.as_ref());
        Repo::new(self.0.dataset(owner, name))
    }
}

impl AsRef<HFClientSync> for Api {
    fn as_ref(&self) -> &HFClientSync {
        &self.0
    }
}

/// Repository handle. Optionally pinned to a revision.
#[derive(Clone, Debug)]
pub struct Repo<T: RepoType> {
    repo: HFRepositorySync<T>,
    revision: Option<String>,
}

impl<T: RepoType> Repo<T> {
    pub fn new(repo: HFRepositorySync<T>) -> Self {
        Self {
            repo,
            revision: None,
        }
    }

    /// Pins every subsequent [`Repo::get`] to `revision`.
    pub fn with_revision<S: AsRef<str>>(mut self, revision: S) -> Self {
        self.revision = Some(revision.as_ref().to_string());
        self
    }

    pub fn revision(&self) -> Option<&str> {
        self.revision.as_deref()
    }

    /// Returns the path to `filename`. Downloads if not in cache.
    ///
    /// Download progress is printed on stderr.
    pub fn get<S: AsRef<str>>(&self, filename: S) -> HFResult<PathBuf> {
        let filename = filename.as_ref();
        let download = self
            .repo
            .download_file()
            .filename(filename)
            .maybe_revision(self.revision.clone());
        match download.clone().local_files_only(true).send() {
            Err(HFError::LocalEntryNotFound { .. }) => {
                download.progress(StderrProgress::new(filename)).send()
            }
            cached => cached,
        }
    }
}

impl<T: RepoType> AsRef<HFRepositorySync<T>> for Repo<T> {
    fn as_ref(&self) -> &HFRepositorySync<T> {
        &self.repo
    }
}

/// Download progress on stderr.
///
/// On terminal redraws in place every 1 percent.
/// On stderr redirect prints full lines every 10 percent.
struct StderrProgress {
    filename: String,
    downloaded: AtomicU64,
    total: AtomicU64,
    /// Last step drawn
    drawn: AtomicU64,
    /// Whether stderr is a terminal and `\r` redraw makes sense.
    tty: bool,
}

const NOT_DRAWN: u64 = u64::MAX;

impl StderrProgress {
    fn new(filename: &str) -> Self {
        Self {
            filename: filename.to_string(),
            downloaded: AtomicU64::new(0),
            total: AtomicU64::new(0),
            drawn: AtomicU64::new(NOT_DRAWN),
            tty: std::io::stderr().is_terminal(),
        }
    }

    fn draw(&self) {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 {
            return;
        }
        let downloaded = self.downloaded.load(Ordering::Relaxed).min(total);
        let percent = downloaded * 100 / total;
        let step = if self.tty { percent } else { percent / 10 };
        if self.drawn.swap(step, Ordering::Relaxed) == step {
            return;
        }
        let line = progress_line(self.tty, &self.filename, percent, downloaded, total);
        let mut stderr = std::io::stderr().lock();
        let _ = stderr.write_all(line.as_bytes());
        let _ = stderr.flush();
    }
}

/// Progress report. In place redraw on a terminal. Whole line elsewhere.
fn progress_line(tty: bool, filename: &str, percent: u64, downloaded: u64, total: u64) -> String {
    let (unit, scale) = human_unit(total);
    let (lead, trail) = if tty { ("\r", "  ") } else { ("", "\n") };
    format!(
        "{lead}{filename}: {percent:>3}% ({:.1}/{:.1} {unit}){trail}",
        downloaded as f64 / scale,
        total as f64 / scale,
    )
}

/// Human readable bytes
fn human_unit(bytes: u64) -> (&'static str, f64) {
    const KIB: f64 = 1024.;
    const MIB: f64 = KIB * 1024.;
    const GIB: f64 = MIB * 1024.;
    match bytes as f64 {
        b if b >= GIB => ("GiB", GIB),
        b if b >= MIB => ("MiB", MIB),
        b if b >= KIB => ("KiB", KIB),
        _ => ("B", 1.),
    }
}

impl ProgressHandler for StderrProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        match event {
            DownloadEvent::Start { total_bytes, .. } => {
                self.total.store(*total_bytes, Ordering::Relaxed);
            }
            DownloadEvent::Progress { files } => {
                if let Some(file) = files.first() {
                    self.downloaded
                        .store(file.bytes_completed, Ordering::Relaxed);
                }
                self.draw();
            }
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            } => {
                self.downloaded.store(*bytes_completed, Ordering::Relaxed);
                self.total.store(*total_bytes, Ordering::Relaxed);
                self.draw();
            }
            DownloadEvent::Complete => {
                if self.tty && self.drawn.load(Ordering::Relaxed) != NOT_DRAWN {
                    let _ = writeln!(std::io::stderr());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{human_unit, progress_line};

    #[test]
    fn terminal_progress_redraws_in_place() {
        let line = progress_line(
            true,
            "model.safetensors",
            42,
            42 * 1024 * 1024,
            100 * 1024 * 1024,
        );
        assert_eq!(line, "\rmodel.safetensors:  42% (42.0/100.0 MiB)  ");
        assert!(!line.contains('\n'), "a redraw must not end the line");
    }

    #[test]
    fn redirected_progress_writes_lines() {
        let line = progress_line(
            false,
            "model.safetensors",
            42,
            42 * 1024 * 1024,
            100 * 1024 * 1024,
        );
        assert_eq!(line, "model.safetensors:  42% (42.0/100.0 MiB)\n");
        assert!(!line.contains('\r'), "a log must not get carriage returns");
    }

    #[test]
    fn verify_human_units() {
        assert_eq!(human_unit(612).0, "B");
        assert_eq!(human_unit(4 * 1024).0, "KiB");
        assert_eq!(human_unit(4 * 1024 * 1024).0, "MiB");
        assert_eq!(human_unit(4 * 1024 * 1024 * 1024).0, "GiB");
    }
}
