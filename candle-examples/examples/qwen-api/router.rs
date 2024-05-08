use axum::{
    extract::Request, handler::HandlerWithoutStateExt, http::StatusCode, routing::get, Router,
};
use std::net::SocketAddr;
// use tower::ServiceExt;
use tower_http::{
    // services::{ServeDir, ServeFile},
    trace::TraceLayer,
};
pub fn init()->Router{
    // let serve_dir = ServeDir::new("static");
    // let css_dir = ServeDir::new("static/css");
    // let js_dir = ServeDir::new("static/js");
    // let images_dir = ServeDir::new("static/images");

    axum::Router::new()
        // .nest_service("/", serve_dir.clone())
        // .nest_service("/css", css_dir.clone())
        // .nest_service("/js", js_dir.clone())
        // .nest_service("/images", images_dir.clone())
        .route("/greet", get(|| async { "Hello, qwen api copilot!🌱🌎" }))
        .nest("/", crate::controller::pingserver::index_router())
        // .nest("/prompt", controller::prompt::index_router())
}

// pub fn xinit()->Router{
//     let serve_dir = ServeDir::new("static");
//     let css_dir = ServeDir::new("static/css");
//     let js_dir = ServeDir::new("static/js");
//     let images_dir = ServeDir::new("static/images");

//     axum::Router::new()
//         .nest_service("/", serve_dir.clone())
//         .nest_service("/css", css_dir.clone())
//         .nest_service("/js", js_dir.clone())
//         .nest_service("/images", images_dir.clone())
//         .route("/greet", get(|| async { "Hello, axum World!🌱🌎" }))
//         // .nest("/vprservice", controller::vpr::x_router())

// }
// pub fn using_serve_dir() -> Router {
//     // serve the file in the "assets" directory under `/assets`
//     Router::new().nest_service("/assets", ServeDir::new("static/assets"))
//         .route("/greet", get(|| async { "Hello, axum World!🌱🌎" }))
// }
// #[allow(dead_code)]
// pub fn using_serve_dir_with_assets_fallback() -> Router {
//     // `ServeDir` allows setting a fallback if an asset is not found
//     // so with this `GET /assets/doesnt-exist.jpg` will return `index.html`
//     // rather than a 404
//     let serve_dir = ServeDir::new("static/assets").not_found_service(ServeFile::new("assets/index.html"));

//     Router::new()
//         .route("/foo", get(|| async { "Hi from /foo" }))
//         .nest_service("/assets", serve_dir.clone())
//         .fallback_service(serve_dir)
// }
// #[allow(dead_code)]
// pub fn using_serve_dir_only_from_root_via_fallback() -> Router {
//     // you can also serve the assets directly from the root (not nested under `/assets`)
//     // by only setting a `ServeDir` as the fallback
//     let serve_dir = ServeDir::new("static/assets").not_found_service(ServeFile::new("assets/index.html"));

//     Router::new()
//         .route("/foo", get(|| async { "Hi from /foo" }))
//         .fallback_service(serve_dir)
// }
// #[allow(dead_code)]
// pub fn using_serve_dir_with_handler_as_service() -> Router {
//     async fn handle_404() -> (StatusCode, &'static str) {
//         (StatusCode::NOT_FOUND, "Not found")
//     }

//     // you can convert handler function to service
//     let service = handle_404.into_service();

//     let serve_dir = ServeDir::new("static/assets").not_found_service(service);

//     Router::new()
//         .route("/foo", get(|| async { "Hi from /foo" }))
//         .fallback_service(serve_dir)
// }
// #[allow(dead_code)]
// pub fn two_serve_dirs() -> Router {
//     // you can also have two `ServeDir`s nested at different paths
//     let serve_dir_from_assets = ServeDir::new("static/assets");
//     let serve_dir_from_dist = ServeDir::new("static/dist");

//     Router::new()
//         .nest_service("/assets", serve_dir_from_assets)
//         .nest_service("/dist", serve_dir_from_dist)
// }

// #[allow(clippy::let_and_return)]
// pub fn calling_serve_dir_from_a_handler() -> Router {
//     // via `tower::Service::call`, or more conveniently `tower::ServiceExt::oneshot` you can
//     // call `ServeDir` yourself from a handler
//     Router::new().nest_service(
//         "/foo",
//         get(|request: Request| async {
//             let service = ServeDir::new("static/assets");
//             let result = service.oneshot(request).await;
//             result
//         }),
//     )
// }
// #[allow(dead_code)]
// pub fn using_serve_file_from_a_route() -> Router {
//     Router::new().route_service("/foo", ServeFile::new("static/assets/index.html"))
// }

pub async  fn serve(app: Router, port: u16) {
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    tracing::debug!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app.layer(TraceLayer::new_for_http()))
        .await
        .unwrap();
}