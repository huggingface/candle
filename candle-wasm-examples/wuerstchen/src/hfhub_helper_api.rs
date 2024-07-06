
use crate::generic_error::GenericError;
use crate::{create_file, download_file, write_file};
use crate::hfhub_helper::{Cache, Repo, RepoType};

use std::path::PathBuf;

/// Helper to create [`Api`] with all the options.
pub struct ApiBuilder {
    endpoint: String,
    cache: Cache,
    url_template: String,
}

impl Default for ApiBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ApiBuilder {
    /// Default api builder
    /// ```
    /// use hf_hub::api::sync::ApiBuilder;
    /// let api = ApiBuilder::new().build().unwrap();
    /// ```
    pub fn new() -> Self {
        let cache = Cache::default();
        Self::from_cache(cache)
    }

    /// From a given cache
    /// ```
    /// use hf_hub::{api::sync::ApiBuilder, Cache};
    /// let path = std::path::PathBuf::from("/tmp");
    /// let cache = Cache::new(path);
    /// let api = ApiBuilder::from_cache(cache).build().unwrap();
    /// ```
    pub fn from_cache(cache: Cache) -> Self {
        Self {
            endpoint: "https://huggingface.co".to_string(),
            url_template: "{endpoint}/{repo_id}/resolve/{revision}/{filename}".to_string(),
            cache,
        }
    }

    /// Changes the location of the cache directory. Defaults is `~/.cache/huggingface/`.
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache = Cache::new(cache_dir);
        self
    }


    /// Consumes the builder and buids the final [`Api`]
    pub fn build(self) -> Result<Api, GenericError> {
        Ok(Api {
            endpoint: self.endpoint,
            url_template: self.url_template,
            cache: self.cache,
        })
    }
}

/// The actual Api used to interacto with the hub.
/// You can inspect repos with [`Api::info`]
/// or download files with [`Api::download`]
#[derive(Clone)]
pub struct Api {
    endpoint: String,
    url_template: String,
    cache: Cache
}


impl Api {
    /// Creates a default Api, for Api options See [`ApiBuilder`]
    pub fn new() -> Result<Self, GenericError> {
        ApiBuilder::new().build()
    }

    /// Creates a new handle [`ApiRepo`] which contains operations
    /// on a particular [`Repo`]
    pub fn repo(&self, repo: Repo) -> ApiRepo {
        ApiRepo::new(self.clone(), repo)
    }

    /// Simple wrapper over
    /// ```
    /// # use hf_hub::{api::sync::Api, Repo, RepoType};
    /// # let model_id = "gpt2".to_string();
    /// let api = Api::new().unwrap();
    /// let api = api.repo(Repo::new(model_id, RepoType::Model));
    /// ```
    pub fn model(&self, model_id: String) -> ApiRepo {
        self.repo(Repo::new(model_id, RepoType::Model))
    }

    /// Simple wrapper over
    /// ```
    /// # use hf_hub::{api::sync::Api, Repo, RepoType};
    /// # let model_id = "gpt2".to_string();
    /// let api = Api::new().unwrap();
    /// let api = api.repo(Repo::new(model_id, RepoType::Dataset));
    /// ```
    pub fn dataset(&self, model_id: String) -> ApiRepo {
        self.repo(Repo::new(model_id, RepoType::Dataset))
    }

    /// Simple wrapper over
    /// ```
    /// # use hf_hub::{api::sync::Api, Repo, RepoType};
    /// # let model_id = "gpt2".to_string();
    /// let api = Api::new().unwrap();
    /// let api = api.repo(Repo::new(model_id, RepoType::Space));
    /// ```
    pub fn space(&self, model_id: String) -> ApiRepo {
        self.repo(Repo::new(model_id, RepoType::Space))
    }
}

/// Shorthand for accessing things within a particular repo
pub struct ApiRepo {
    api: Api,
    repo: Repo,
}

impl ApiRepo {
    fn new(api: Api, repo: Repo) -> Self {
        Self { api, repo }
    }
}

impl ApiRepo {
    /// Get the fully qualified URL of the remote filename
    /// ```
    /// # use hf_hub::api::sync::Api;
    /// let api = Api::new().unwrap();
    /// let url = api.model("gpt2".to_string()).url("model.safetensors");
    /// assert_eq!(url, "https://huggingface.co/gpt2/resolve/main/model.safetensors");
    /// ```
    pub fn url(&self, filename: &str) -> String {
        let endpoint = &self.api.endpoint;
        let revision = &self.repo.url_revision();
        self.api
            .url_template
            .replace("{endpoint}", endpoint)
            .replace("{repo_id}", &self.repo.url())
            .replace("{revision}", revision)
            .replace("{filename}", filename)
    }

    /// This will attempt the fetch the file locally first, then [`Api.download`]
    /// if the file is not present.
    /// ```no_run
    /// use hf_hub::{api::sync::Api};
    /// let api = Api::new().unwrap();
    /// let local_filename = api.model("gpt2".to_string()).get("model.safetensors").unwrap();
    pub async fn get(&self, filename: &str) -> Result<PathBuf, GenericError> {
        if let Some(path) = self.api.cache.repo(self.repo.clone()).get(filename).await {
            log::info!("loading file: {filename} from cache: {:?}", path);
            Ok(path)
        } else {
            self.download(filename).await
        }
    }

    /// Downloads a remote file (if not already present) into the cache directory
    /// to be used locally.
    /// This functions require internet access to verify if new versions of the file
    /// exist, even if a file is already on disk at location.
    /// ```no_run
    /// # use hf_hub::api::sync::Api;
    /// let api = Api::new().unwrap();
    /// let local_filename = api.model("gpt2".to_string()).download("model.safetensors").unwrap();
    /// ```
    pub async fn download(&self, filename: &str) -> Result<PathBuf, GenericError> {
        let url = self.url(filename);
       
        let mut pointer_path = self
            .api
            .cache
            .repo(self.repo.clone())
            .path();
        pointer_path.push(filename);

        log::info!("download file: {filename} to {:?}", pointer_path);

        let file = create_file(&pointer_path).await?;
        let data = download_file(&url).await?;
        write_file(file, &data).await?;
        Ok(pointer_path)
    }
}