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

/// The actual Api to interact with the hub.
#[cfg(feature = "online")]
pub mod api;

/// Current version (used in user-agent)
const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Current name (used in user-agent)
const NAME: &str = env!("CARGO_PKG_NAME");

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

/// The representation of a repo on the hub.
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
        match self.repo_type {
            RepoType::Model => self.repo_id.replace('/', "--"),
            RepoType::Dataset => {
                format!("datasets/{}", self.repo_id.replace('/', "--"))
            }
            RepoType::Space => {
                format!("spaces/{}", self.repo_id.replace('/', "--"))
            }
        }
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
