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

mod cli;
pub mod error;
pub mod graph;

pub use cli::*;
pub use error::CugraError;
pub use graph::*;

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
    use failure::{err_msg, ResultExt};
    use std::{env, path, process};

    let mut output = path::PathBuf::from(input);
    output.set_extension("ptx");
    let output = output.to_str().unwrap();

    let mut cmd = process::Command::new("nvcc");
    if let Some(osstring) = env::var_os("NVCCFLAGS") {
        let nvccflags = osstring
            .into_string()
            .map_err(|e| err_msg("could not parse NVCCFLAGS as utf8"))?;
        cmd.args(nvccflags.split_whitespace());
    }
    cmd.args(&["-lcudadevrt", "-dc", "-ptx", input, "-o", &output]);

    assert!(cmd.status().context("running nvcc")?.success());

    Ok(std::fs::read_to_string(output).context("reading nvcc output")?)
}
