use crate::{Cache, Repo};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{distributions::Alphanumeric, thread_rng, Rng};
use reqwest::{
    header::{
        HeaderMap, HeaderName, HeaderValue, InvalidHeaderValue, ToStrError, AUTHORIZATION,
        CONTENT_RANGE, LOCATION, RANGE, USER_AGENT,
    },
    redirect::Policy,
    Client, Error as ReqwestError,
};
use serde::Deserialize;
use std::num::ParseIntError;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tokio::io::{AsyncSeekExt, AsyncWriteExt, SeekFrom};
use tokio::sync::{AcquireError, Semaphore, TryAcquireError};

/// Current version (used in user-agent)
const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Current name (used in user-agent)
const NAME: &str = env!("CARGO_PKG_NAME");

#[derive(Debug, Error)]
/// All errors the API can throw
pub enum ApiError {
    /// Api expects certain header to be present in the results to derive some information
    #[error("Header {0} is missing")]
    MissingHeader(HeaderName),

    /// The header exists, but the value is not conform to what the Api expects.
    #[error("Header {0} is invalid")]
    InvalidHeader(HeaderName),

    /// The value cannot be used as a header during request header construction
    #[error("Invalid header value {0}")]
    InvalidHeaderValue(#[from] InvalidHeaderValue),

    /// The header value is not valid utf-8
    #[error("header value is not a string")]
    ToStr(#[from] ToStrError),

    /// Error in the request
    #[error("request error: {0}")]
    RequestError(#[from] ReqwestError),

    /// Error parsing some range value
    #[error("Cannot parse int")]
    ParseIntError(#[from] ParseIntError),

    /// I/O Error
    #[error("I/O error {0}")]
    IoError(#[from] std::io::Error),

    /// We tried to download chunk too many times
    #[error("Too many retries: {0}")]
    TooManyRetries(Box<ApiError>),

    /// Semaphore cannot be acquired
    #[error("Try acquire: {0}")]
    TryAcquireError(#[from] TryAcquireError),

    /// Semaphore cannot be acquired
    #[error("Acquire: {0}")]
    AcquireError(#[from] AcquireError),
    // /// Semaphore cannot be acquired
    // #[error("Invalid Response: {0:?}")]
    // InvalidResponse(Response),
}

/// Siblings are simplified file descriptions of remote files on the hub
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Siblings {
    /// The path within the repo.
    pub rfilename: String,
}

/// The description of the repo given by the hub
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ModelInfo {
    /// See [`Siblings`]
    pub siblings: Vec<Siblings>,
}

/// Helper to create [`Api`] with all the options.
pub struct ApiBuilder {
    endpoint: String,
    cache: Cache,
    url_template: String,
    token: Option<String>,
    max_files: usize,
    chunk_size: usize,
    parallel_failures: usize,
    max_retries: usize,
    progress: bool,
}

impl Default for ApiBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ApiBuilder {
    /// Default api builder
    /// ```
    /// use candle_hub::api::tokio::ApiBuilder;
    /// let api = ApiBuilder::new().build().unwrap();
    /// ```
    pub fn new() -> Self {
        let cache = Cache::default();
        let mut token_filename = cache.path().clone();
        token_filename.push(".token");
        let token = match std::fs::read_to_string(token_filename) {
            Ok(token_content) => {
                let token_content = token_content.trim();
                if !token_content.is_empty() {
                    Some(token_content.to_string())
                } else {
                    None
                }
            }
            Err(_) => None,
        };

        let progress = true;

        Self {
            endpoint: "https://huggingface.co".to_string(),
            url_template: "{endpoint}/{repo_id}/resolve/{revision}/{filename}".to_string(),
            cache,
            token,
            max_files: num_cpus::get(),
            chunk_size: 10_000_000,
            parallel_failures: 0,
            max_retries: 0,
            progress,
        }
    }

    /// Wether to show a progressbar
    pub fn with_progress(mut self, progress: bool) -> Self {
        self.progress = progress;
        self
    }

    /// Changes the location of the cache directory. Defaults is `~/.cache/huggingface/`.
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache = Cache::new(cache_dir);
        self
    }

    fn build_headers(&self) -> Result<HeaderMap, ApiError> {
        let mut headers = HeaderMap::new();
        let user_agent = format!("unkown/None; {NAME}/{VERSION}; rust/unknown");
        headers.insert(USER_AGENT, HeaderValue::from_str(&user_agent)?);
        if let Some(token) = &self.token {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {token}"))?,
            );
        }
        Ok(headers)
    }

    /// Consumes the builder and buids the final [`Api`]
    pub fn build(self) -> Result<Api, ApiError> {
        let headers = self.build_headers()?;
        let client = Client::builder().default_headers(headers.clone()).build()?;
        let no_redirect_client = Client::builder()
            .redirect(Policy::none())
            .default_headers(headers)
            .build()?;
        Ok(Api {
            endpoint: self.endpoint,
            url_template: self.url_template,
            cache: self.cache,
            client,

            no_redirect_client,
            max_files: self.max_files,
            chunk_size: self.chunk_size,
            parallel_failures: self.parallel_failures,
            max_retries: self.max_retries,
            progress: self.progress,
        })
    }
}

#[derive(Debug)]
struct Metadata {
    commit_hash: String,
    etag: String,
    size: usize,
}

/// The actual Api used to interacto with the hub.
/// You can inspect repos with [`Api::info`]
/// or download files with [`Api::download`]
pub struct Api {
    endpoint: String,
    url_template: String,
    cache: Cache,
    client: Client,
    no_redirect_client: Client,
    max_files: usize,
    chunk_size: usize,
    parallel_failures: usize,
    max_retries: usize,
    progress: bool,
}

fn temp_filename() -> PathBuf {
    let s: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(7)
        .map(char::from)
        .collect();
    let mut path = std::env::temp_dir();
    path.push(s);
    path
}

fn make_relative(src: &Path, dst: &Path) -> PathBuf {
    let path = src;
    let base = dst;

    if path.is_absolute() != base.is_absolute() {
        panic!("This function is made to look at absolute paths only");
    }
    let mut ita = path.components();
    let mut itb = base.components();

    loop {
        match (ita.next(), itb.next()) {
            (Some(a), Some(b)) if a == b => (),
            (some_a, _) => {
                // Ignoring b, because 1 component is the filename
                // for which we don't need to go back up for relative
                // filename to work.
                let mut new_path = PathBuf::new();
                for _ in itb {
                    new_path.push(Component::ParentDir);
                }
                if let Some(a) = some_a {
                    new_path.push(a);
                    for comp in ita {
                        new_path.push(comp);
                    }
                }
                return new_path;
            }
        }
    }
}

fn symlink_or_rename(src: &Path, dst: &Path) -> Result<(), std::io::Error> {
    if dst.exists() {
        return Ok(());
    }

    let src = make_relative(src, dst);
    #[cfg(target_os = "windows")]
    std::os::windows::fs::symlink_file(src, dst)?;

    #[cfg(target_family = "unix")]
    std::os::unix::fs::symlink(src, dst)?;

    #[cfg(not(any(target_family = "unix", target_os = "windows")))]
    std::fs::rename(src, dst)?;

    Ok(())
}

fn jitter() -> usize {
    thread_rng().gen_range(0..=500)
}

fn exponential_backoff(base_wait_time: usize, n: usize, max: usize) -> usize {
    (base_wait_time + n.pow(2) + jitter()).min(max)
}

impl Api {
    /// Creates a default Api, for Api options See [`ApiBuilder`]
    pub fn new() -> Result<Self, ApiError> {
        ApiBuilder::new().build()
    }

    /// Get the fully qualified URL of the remote filename
    /// ```
    /// # use candle_hub::{api::tokio::Api, Repo};
    /// let api = Api::new().unwrap();
    /// let repo = Repo::model("gpt2".to_string());
    /// let url = api.url(&repo, "model.safetensors");
    /// assert_eq!(url, "https://huggingface.co/gpt2/resolve/main/model.safetensors");
    /// ```
    pub fn url(&self, repo: &Repo, filename: &str) -> String {
        let endpoint = &self.endpoint;
        let revision = &repo.url_revision();
        self.url_template
            .replace("{endpoint}", endpoint)
            .replace("{repo_id}", &repo.url())
            .replace("{revision}", revision)
            .replace("{filename}", filename)
    }

    /// Get the underlying api client
    /// Allows for lower level access
    pub fn client(&self) -> &Client {
        &self.client
    }

    async fn metadata(&self, url: &str) -> Result<Metadata, ApiError> {
        let response = self
            .no_redirect_client
            .get(url)
            .header(RANGE, "bytes=0-0")
            .send()
            .await?;
        let response = response.error_for_status()?;
        let headers = response.headers();
        let header_commit = HeaderName::from_static("x-repo-commit");
        let header_linked_etag = HeaderName::from_static("x-linked-etag");
        let header_etag = HeaderName::from_static("etag");

        let etag = match headers.get(&header_linked_etag) {
            Some(etag) => etag,
            None => headers
                .get(&header_etag)
                .ok_or(ApiError::MissingHeader(header_etag))?,
        };
        // Cleaning extra quotes
        let etag = etag.to_str()?.to_string().replace('"', "");
        let commit_hash = headers
            .get(&header_commit)
            .ok_or(ApiError::MissingHeader(header_commit))?
            .to_str()?
            .to_string();

        // The response was redirected o S3 most likely which will
        // know about the size of the file
        let response = if response.status().is_redirection() {
            self.client
                .get(headers.get(LOCATION).unwrap().to_str()?.to_string())
                .header(RANGE, "bytes=0-0")
                .send()
                .await?
        } else {
            response
        };
        let headers = response.headers();
        let content_range = headers
            .get(CONTENT_RANGE)
            .ok_or(ApiError::MissingHeader(CONTENT_RANGE))?
            .to_str()?;

        let size = content_range
            .split('/')
            .last()
            .ok_or(ApiError::InvalidHeader(CONTENT_RANGE))?
            .parse()?;
        Ok(Metadata {
            commit_hash,
            etag,
            size,
        })
    }

    async fn download_tempfile(
        &self,
        url: &str,
        length: usize,
        progressbar: Option<ProgressBar>,
    ) -> Result<PathBuf, ApiError> {
        let mut handles = vec![];
        let semaphore = Arc::new(Semaphore::new(self.max_files));
        let parallel_failures_semaphore = Arc::new(Semaphore::new(self.parallel_failures));
        let filename = temp_filename();

        // Create the file and set everything properly
        tokio::fs::File::create(&filename)
            .await?
            .set_len(length as u64)
            .await?;

        let chunk_size = self.chunk_size;
        for start in (0..length).step_by(chunk_size) {
            let url = url.to_string();
            let filename = filename.clone();
            let client = self.client.clone();

            let stop = std::cmp::min(start + chunk_size - 1, length);
            let permit = semaphore.clone().acquire_owned().await?;
            let parallel_failures = self.parallel_failures;
            let max_retries = self.max_retries;
            let parallel_failures_semaphore = parallel_failures_semaphore.clone();
            let progress = progressbar.clone();
            handles.push(tokio::spawn(async move {
                let mut chunk = Self::download_chunk(&client, &url, &filename, start, stop).await;
                let mut i = 0;
                if parallel_failures > 0 {
                    while let Err(dlerr) = chunk {
                        let parallel_failure_permit =
                            parallel_failures_semaphore.clone().try_acquire_owned()?;

                        let wait_time = exponential_backoff(300, i, 10_000);
                        tokio::time::sleep(tokio::time::Duration::from_millis(wait_time as u64))
                            .await;

                        chunk = Self::download_chunk(&client, &url, &filename, start, stop).await;
                        i += 1;
                        if i > max_retries {
                            return Err(ApiError::TooManyRetries(dlerr.into()));
                        }
                        drop(parallel_failure_permit);
                    }
                }
                drop(permit);
                if let Some(p) = progress {
                    p.inc((stop - start) as u64);
                }
                chunk
            }));
        }

        // Output the chained result
        let results: Vec<Result<Result<(), ApiError>, tokio::task::JoinError>> =
            futures::future::join_all(handles).await;
        let results: Result<(), ApiError> = results.into_iter().flatten().collect();
        results?;
        if let Some(p) = progressbar {
            p.finish()
        }
        Ok(filename)
    }

    async fn download_chunk(
        client: &reqwest::Client,
        url: &str,
        filename: &PathBuf,
        start: usize,
        stop: usize,
    ) -> Result<(), ApiError> {
        // Process each socket concurrently.
        let range = format!("bytes={start}-{stop}");
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .open(filename)
            .await?;
        file.seek(SeekFrom::Start(start as u64)).await?;
        let response = client
            .get(url)
            .header(RANGE, range)
            .send()
            .await?
            .error_for_status()?;
        let content = response.bytes().await?;
        file.write_all(&content).await?;
        Ok(())
    }

    /// This will attempt the fetch the file locally first, then [`Api.download`]
    /// if the file is not present.
    /// ```no_run
    /// # use candle_hub::{api::tokio::ApiBuilder, Repo};
    /// # tokio_test::block_on(async {
    /// let api = ApiBuilder::new().build().unwrap();
    /// let repo = Repo::model("gpt2".to_string());
    /// let local_filename = api.get(&repo, "model.safetensors").await.unwrap();
    /// # })
    pub async fn get(&self, repo: &Repo, filename: &str) -> Result<PathBuf, ApiError> {
        if let Some(path) = self.cache.get(repo, filename) {
            Ok(path)
        } else {
            self.download(repo, filename).await
        }
    }

    /// Downloads a remote file (if not already present) into the cache directory
    /// to be used locally.
    /// This functions require internet access to verify if new versions of the file
    /// exist, even if a file is already on disk at location.
    /// ```no_run
    /// # use candle_hub::{api::tokio::ApiBuilder, Repo};
    /// # tokio_test::block_on(async {
    /// let api = ApiBuilder::new().build().unwrap();
    /// let repo = Repo::model("gpt2".to_string());
    /// let local_filename = api.download(&repo, "model.safetensors").await.unwrap();
    /// # })
    /// ```
    pub async fn download(&self, repo: &Repo, filename: &str) -> Result<PathBuf, ApiError> {
        let url = self.url(repo, filename);
        let metadata = self.metadata(&url).await?;

        let blob_path = self.cache.blob_path(repo, &metadata.etag);
        std::fs::create_dir_all(blob_path.parent().unwrap())?;

        let progressbar = if self.progress {
            let progress = ProgressBar::new(metadata.size as u64);
            progress.set_style(
                ProgressStyle::with_template(
                    "{msg} [{elapsed_precise}] [{wide_bar}] {bytes}/{total_bytes} {bytes_per_sec} ({eta})",
                )
                .unwrap(), // .progress_chars("━ "),
            );
            let maxlength = 30;
            let message = if filename.len() > maxlength {
                format!("..{}", &filename[filename.len() - maxlength..])
            } else {
                filename.to_string()
            };
            progress.set_message(message);
            Some(progress)
        } else {
            None
        };

        let tmp_filename = self
            .download_tempfile(&url, metadata.size, progressbar)
            .await?;

        if tokio::fs::rename(&tmp_filename, &blob_path).await.is_err() {
            // Renaming may fail if locations are different mount points
            std::fs::File::create(&blob_path)?;
            tokio::fs::copy(tmp_filename, &blob_path).await?;
        }

        let mut pointer_path = self.cache.pointer_path(repo, &metadata.commit_hash);
        pointer_path.push(filename);
        std::fs::create_dir_all(pointer_path.parent().unwrap()).ok();

        symlink_or_rename(&blob_path, &pointer_path)?;
        self.cache.create_ref(repo, &metadata.commit_hash)?;

        Ok(pointer_path)
    }

    /// Get information about the Repo
    /// ```
    /// # use candle_hub::{api::tokio::Api, Repo};
    /// # tokio_test::block_on(async {
    /// let api = Api::new().unwrap();
    /// let repo = Repo::model("gpt2".to_string());
    /// api.info(&repo);
    /// # })
    /// ```
    pub async fn info(&self, repo: &Repo) -> Result<ModelInfo, ApiError> {
        let url = format!("{}/api/{}", self.endpoint, repo.api_url());
        let response = self.client.get(url).send().await?;
        let response = response.error_for_status()?;

        let model_info = response.json().await?;

        Ok(model_info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RepoType;
    use rand::{distributions::Alphanumeric, Rng};
    use sha256::try_digest;

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        pub fn new() -> Self {
            let s: String = rand::thread_rng()
                .sample_iter(&Alphanumeric)
                .take(7)
                .map(char::from)
                .collect();
            let mut path = std::env::temp_dir();
            path.push(s);
            std::fs::create_dir(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.path).unwrap()
        }
    }

    #[tokio::test]
    async fn simple() {
        let tmp = TempDir::new();
        let api = ApiBuilder::new()
            .with_progress(false)
            .with_cache_dir(tmp.path.clone())
            .build()
            .unwrap();
        let repo = Repo::new("julien-c/dummy-unknown".to_string(), RepoType::Model);
        let downloaded_path = api.download(&repo, "config.json").await.unwrap();
        assert!(downloaded_path.exists());
        let val = try_digest(&*downloaded_path).unwrap();
        assert_eq!(
            val,
            "b908f2b7227d4d31a2105dfa31095e28d304f9bc938bfaaa57ee2cacf1f62d32"
        );

        // Make sure the file is now seeable without connection
        let cache_path = api.cache.get(&repo, "config.json").unwrap();
        assert_eq!(cache_path, downloaded_path);
    }

    #[tokio::test]
    async fn dataset() {
        let tmp = TempDir::new();
        let api = ApiBuilder::new()
            .with_progress(false)
            .with_cache_dir(tmp.path.clone())
            .build()
            .unwrap();
        let repo = Repo::with_revision(
            "wikitext".to_string(),
            RepoType::Dataset,
            "refs/convert/parquet".to_string(),
        );
        let downloaded_path = api
            .download(&repo, "wikitext-103-v1/wikitext-test.parquet")
            .await
            .unwrap();
        assert!(downloaded_path.exists());
        let val = try_digest(&*downloaded_path).unwrap();
        assert_eq!(
            val,
            "59ce09415ad8aa45a9e34f88cec2548aeb9de9a73fcda9f6b33a86a065f32b90"
        )
    }

    #[tokio::test]
    async fn info() {
        let tmp = TempDir::new();
        let api = ApiBuilder::new()
            .with_progress(false)
            .with_cache_dir(tmp.path.clone())
            .build()
            .unwrap();
        let repo = Repo::with_revision(
            "wikitext".to_string(),
            RepoType::Dataset,
            "refs/convert/parquet".to_string(),
        );
        let model_info = api.info(&repo).await.unwrap();
        assert_eq!(
            model_info,
            ModelInfo {
                siblings: vec![
                    Siblings {
                        rfilename: ".gitattributes".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-raw-v1/wikitext-test.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-raw-v1/wikitext-train-00000-of-00002.parquet"
                            .to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-raw-v1/wikitext-train-00001-of-00002.parquet"
                            .to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-raw-v1/wikitext-validation.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-v1/test/index.duckdb".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-v1/validation/index.duckdb".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-v1/wikitext-test.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-v1/wikitext-train-00000-of-00002.parquet"
                            .to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-v1/wikitext-train-00001-of-00002.parquet"
                            .to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-v1/wikitext-validation.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-raw-v1/test/index.duckdb".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-raw-v1/train/index.duckdb".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-raw-v1/validation/index.duckdb".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-raw-v1/wikitext-test.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-raw-v1/wikitext-train.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-raw-v1/wikitext-validation.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-v1/wikitext-test.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-v1/wikitext-train.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-v1/wikitext-validation.parquet".to_string()
                    }
                ],
            }
        )
    }
}
