//! Error and Result helpers.

pub type Result<T> = anyhow::Result<T>;

pub fn not_implemented(what: &'static str) -> anyhow::Error {
    anyhow::anyhow!("not implemented: {what}")
}
