use std::{path::PathBuf, str::FromStr};

use crate::opfs::exist_file;

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
#[derive(Clone)]
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

    /// Creates a new handle [`CacheRepo`] which contains operations
    /// on a particular [`Repo`]
    pub fn repo(&self, repo: Repo) -> CacheRepo {
        CacheRepo::new(self.clone(), repo)
    }

    /// Simple wrapper over
    /// ```
    /// # use wasm_helper::hfhub::{Cache, Repo, RepoType};
    /// # let model_id = "gpt2".to_string();
    /// let cache = Cache::new("/tmp/".into());
    /// let cache = cache.repo(Repo::new(model_id, RepoType::Model));
    /// ```
    pub fn model(&self, model_id: String) -> CacheRepo {
        self.repo(Repo::new(model_id, RepoType::Model))
    }

    /// Simple wrapper over
    /// ```
    /// # use wasm_helper::hfhub::{Cache, Repo, RepoType};
    /// # let model_id = "gpt2".to_string();
    /// let cache = Cache::new("/tmp/".into());
    /// let cache = cache.repo(Repo::new(model_id, RepoType::Dataset));
    /// ```
    pub fn dataset(&self, model_id: String) -> CacheRepo {
        self.repo(Repo::new(model_id, RepoType::Dataset))
    }

    /// Simple wrapper over
    /// ```
    /// # use wasm_helper::hfhub::{Cache, Repo, RepoType};
    /// # let model_id = "gpt2".to_string();
    /// let cache = Cache::new("/tmp/".into());
    /// let cache = cache.repo(Repo::new(model_id, RepoType::Space));
    /// ```
    pub fn space(&self, model_id: String) -> CacheRepo {
        self.repo(Repo::new(model_id, RepoType::Space))
    }
}

/// Shorthand for accessing things within a particular repo
pub struct CacheRepo {
    cache: Cache,
    repo: Repo,
}

impl CacheRepo {
    fn new(cache: Cache, repo: Repo) -> Self {
        Self { cache, repo }
    }
    /// This will get the location of the file within the cache for the remote
    /// `filename`. Will return `None` if file is not already present in cache.
    pub async fn get(&self, filename: &str) -> Option<PathBuf> {
        let mut pointer_path = self.path();
        pointer_path.push(filename);
        if exist_file(&pointer_path).await{
            Some(pointer_path)
        } else {
            None
        }
    }

    pub fn path(&self) -> PathBuf {
        let mut ref_path = self.cache.path.clone();
        ref_path.push(self.repo.folder_name());
        ref_path
    }
}

impl Default for Cache {
    fn default() -> Self {
        let mut cache = PathBuf::from_str("/").unwrap();
        cache.push(".cache");
        cache.push("huggingface");
        Self::new(cache)
    }
}

/// The representation of a repo on the hub.
#[derive(Clone)]
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
    pub fn url_revision(&self) -> String {
        self.revision.replace('/', "%2F")
    }

    /// Used to compute the repo's url part when accessing the metadata of the repo
    pub fn api_url(&self) -> String {
        let prefix = match self.repo_type {
            RepoType::Model => "models",
            RepoType::Dataset => "datasets",
            RepoType::Space => "spaces",
        };
        format!("{prefix}/{}/revision/{}", self.repo_id, self.url_revision())
    }
}
