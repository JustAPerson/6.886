use std;
use std::convert::From;

use cuda;
use failure;


pub type Result<T> = std::result::Result<T, CugraError>;

#[derive(Debug, Fail)]
pub enum CugraError {
    #[fail(display = "CUDA error: {:?}", _0)]
    CudaError(cuda::driver::Error),

    #[fail(display = "IO error: {}", _0)]
    IoError(std::io::Error),

    #[fail(display = "Malformed input graph: {}", _0)]
    MalformedGraph(ParseError),

    #[fail(display = "{} error: {}", _0, _1)]
    ProgramError(&'static str, failure::Error),
}

impl From<cuda::driver::Error> for CugraError {
    fn from(e: cuda::driver::Error) -> Self {
        CugraError::CudaError(e)
    }
}

impl From<std::io::Error> for CugraError {
    fn from(e: std::io::Error) -> Self {
        CugraError::IoError(e)
    }
}

impl From<ParseError> for CugraError {
    fn from(e: ParseError) -> Self {
        CugraError::MalformedGraph(e)
    }
}

#[derive(Debug, Fail)]
pub enum ParseError {
    #[fail(display = "unsupported graph format")]
    UnsupportedFormat,

    #[fail(display = "could not parse int `{}` ({}) on line {}", _0, _1, _2)]
    InvalidNumber(String, std::num::ParseIntError, usize),

    #[fail(display = "expected {} lines", _0)]
    IncorrectLength(usize),
}
