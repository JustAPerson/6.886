#![feature(try_from)]
use std::iter::FromIterator;

pub extern crate clap;
// #[macro_use]
pub extern crate failure;
#[macro_use]
extern crate failure_derive;
#[macro_use]
extern crate log;
pub extern crate cuda;

pub mod error;
pub mod graph;
mod cli;

pub use error::CugraError;
pub use graph::*;
pub use cli::*;


pub trait Program {
    type Input: From<EdgeList>;

    fn process(&mut self, input: Self::Input) -> Result<(), failure::Error>;
    fn name(&self) -> &'static str {
        "Unnamed program"
    }
    fn default_graph(&self) -> &'static str {
        "./examples/rMat_1k"
    }
}

pub fn compile_ptx(input: &str) -> Result<String, failure::Error> {
    use std::path::PathBuf;
    use std::process::Command;
    use failure::ResultExt;

    let mut output = PathBuf::from(input);
    output.set_extension("ptx");
    let output = output.to_str().unwrap();

    let mut cmd = Command::new("nvcc");
    cmd.args(&[
        "-ccbin",
        "clang-3.8",
        "-arch",
        "sm_30",
        "-lcudadevrt",
        "-dc",
        "-g",
        "-G",
        "-ptx",
        input,
        "-o",
        &output,
    ]);

    assert!(cmd.status().context("running nvcc")?.success());

    Ok(std::fs::read_to_string(output).context("reading nvcc output")?)
}
