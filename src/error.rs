/// Main library error type.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("invalid shapes in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    BinaryInvalidShape {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
        op: &'static str,
    },

    #[error("unexpected rank, expected: {expected}, got: {got} ({shape:?})")]
    UnexpectedNumberOfDims {
        expected: usize,
        got: usize,
        shape: Vec<usize>,
    },
}

pub type Result<T> = std::result::Result<T, Error>;
