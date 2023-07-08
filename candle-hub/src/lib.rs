#![deny(missing_docs)]
//! This crates aims to emulate and be compatible with the
//! [huggingface_hub](https://github.com/huggingface/huggingface_hub/) python package.
//!
//! compatible means the Api should reuse the same files skipping downloads if
//! they are already present and whenever this crate downloads or modifies this cache
//! it should be consistent with [huggingface_hub](https://github.com/huggingface/huggingface_hub/)
//!
//! At this time only a limited subset of the functionality is present, the goal is to add new
//! features over time
use std::io::Write;
use std::path::PathBuf;

/// The actual Api to interact with the hub.
#[cfg(feature = "online")]
pub mod api;

/// The type of repo to interact with
#[derive(Debug, Clone, Copy)]
pub enum RepoType {
    /// This is a model, usually it consists of weight files and some configuration
    /// files
    Model,
    /// This is a dataset, usually contains data within parquet files
    Dataset,
    /// This is a space, usually a demo showcashing a given model or dataset
    Space,
}

/// A local struct used to fetch information from the cache folder.
pub struct Cache {
    path: PathBuf,
}

impl Cache {
    /// Creates a new cache object location
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Creates a new cache object location
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// This will get the location of the file within the cache for the remote
    /// `filename`. Will return `None` if file is not already present in cache.
    pub fn get(&self, repo: &Repo, filename: &str) -> Option<PathBuf> {
        let mut commit_path = self.path.clone();
        commit_path.push(repo.folder_name());
        commit_path.push("refs");
        commit_path.push(repo.revision());
        let commit_hash = std::fs::read_to_string(commit_path).ok()?;
        let mut pointer_path = self.pointer_path(repo, &commit_hash);
        pointer_path.push(filename);
        if pointer_path.exists() {
            Some(pointer_path)
        } else {
            None
        }
    }

    /// Creates a reference in the cache directory that points branches to the correct
    /// commits within the blobs.
    pub fn create_ref(&self, repo: &Repo, commit_hash: &str) -> Result<(), std::io::Error> {
        let mut ref_path = self.path.clone();
        ref_path.push(repo.folder_name());
        ref_path.push("refs");
        ref_path.push(repo.revision());
        // Needs to be done like this because revision might contain `/` creating subfolders here.
        std::fs::create_dir_all(ref_path.parent().unwrap())?;
        let mut file1 = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(&ref_path)?;
        file1.write_all(commit_hash.trim().as_bytes())?;
        Ok(())
    }

    #[cfg(feature = "online")]
    pub(crate) fn blob_path(&self, repo: &Repo, etag: &str) -> PathBuf {
        let mut blob_path = self.path.clone();
        blob_path.push(repo.folder_name());
        blob_path.push("blobs");
        blob_path.push(etag);
        blob_path
    }

    pub(crate) fn pointer_path(&self, repo: &Repo, commit_hash: &str) -> PathBuf {
        let mut pointer_path = self.path.clone();
        pointer_path.push(repo.folder_name());
        pointer_path.push("snapshots");
        pointer_path.push(commit_hash);
        pointer_path
    }
}

impl Default for Cache {
    fn default() -> Self {
        let path = match std::env::var("HF_HOME") {
            Ok(home) => home.into(),
            Err(_) => {
                let mut cache = dirs::home_dir().expect("Cache directory cannot be found");
                cache.push(".cache");
                cache.push("huggingface");
                cache.push("hub");
                cache
            }
        };
        Self::new(path)
    }
}

/// The representation of a repo on the hub.
#[allow(dead_code)] // Repo type unused in offline mode
pub struct Repo {
    repo_id: String,
    repo_type: RepoType,
    revision: String,
}

impl Repo {
    /// Repo with the default branch ("main").
    pub fn new(repo_id: String, repo_type: RepoType) -> Self {
        Self::with_revision(repo_id, repo_type, "main".to_string())
    }

    /// fully qualified Repo
    pub fn with_revision(repo_id: String, repo_type: RepoType, revision: String) -> Self {
        Self {
            repo_id,
            repo_type,
            revision,
        }
    }

    /// Shortcut for [`Repo::new`] with [`RepoType::Model`]
    pub fn model(repo_id: String) -> Self {
        Self::new(repo_id, RepoType::Model)
    }

    /// Shortcut for [`Repo::new`] with [`RepoType::Dataset`]
    pub fn dataset(repo_id: String) -> Self {
        Self::new(repo_id, RepoType::Dataset)
    }

    /// Shortcut for [`Repo::new`] with [`RepoType::Space`]
    pub fn space(repo_id: String) -> Self {
        Self::new(repo_id, RepoType::Space)
    }

    /// The normalized folder nameof the repo within the cache directory
    pub fn folder_name(&self) -> String {
        let prefix = match self.repo_type {
            RepoType::Model => "models",
            RepoType::Dataset => "datasets",
            RepoType::Space => "spaces",
        };
        format!("{prefix}--{}", self.repo_id).replace('/', "--")
    }

    /// The revision
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// The actual URL part of the repo
    #[cfg(feature = "online")]
    pub fn url(&self) -> String {
        match self.repo_type {
            RepoType::Model => self.repo_id.to_string(),
            RepoType::Dataset => {
                format!("datasets/{}", self.repo_id)
            }
            RepoType::Space => {
                format!("spaces/{}", self.repo_id)
            }
        }
    }

    /// Revision needs to be url escaped before being used in a URL
    #[cfg(feature = "online")]
    pub fn url_revision(&self) -> String {
        self.revision.replace('/', "%2F")
    }

    /// Used to compute the repo's url part when accessing the metadata of the repo
    #[cfg(feature = "online")]
    pub fn api_url(&self) -> String {
        let prefix = match self.repo_type {
            RepoType::Model => "models",
            RepoType::Dataset => "datasets",
            RepoType::Space => "spaces",
        };
        format!("{prefix}/{}/revision/{}", self.repo_id, self.url_revision())
    }
}
