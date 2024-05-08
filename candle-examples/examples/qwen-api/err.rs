#[derive(Debug)]
pub enum AppErrorType {
    Database,
    Template,
}

type Cause = Box<dyn std::error::Error>;

#[derive(Debug)]
pub enum AppErrorItem {
    Message(Option<String>),
    Cause(Cause),
}

#[derive(Debug)]
pub struct AppError {
    pub types: AppErrorType,
    pub error: AppErrorItem,
}

impl AppError {
    pub fn new(types: AppErrorType, error: AppErrorItem) -> Self {
        Self { types, error }
    }
    pub fn from_err(cause: Cause, types: AppErrorType) -> Self {
        Self::new(types, AppErrorItem::Cause(cause))
    }
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for AppError {}

impl From<askama::Error> for AppError {
    fn from(err: askama::Error) -> Self {
        Self::from_err(Box::new(err), AppErrorType::Template)
    }
}

impl axum::response::IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let msg = match self.error {
            AppErrorItem::Cause(err) => err.to_string(),
            AppErrorItem::Message(msg) => msg.unwrap_or("发生错误".to_string()),
        };
        msg.into_response()
    }
}
