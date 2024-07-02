


use axum::{http::StatusCode, response::IntoResponse};
use lazy_static::lazy_static;
use std::io;
pub async fn web_handle_error(_err: io::Error) -> impl IntoResponse {
    (StatusCode::INTERNAL_SERVER_ERROR, "Something went wrong...")
}
// lazy_static! {
//     pub static ref CONFIG_PATH: String = "./cpp_lib/res/voiceprint_online.conf".to_string();
// }

