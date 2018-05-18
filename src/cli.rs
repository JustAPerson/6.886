use clap;
use graph;

use std::path::Path;

use Program;
use error::{CugraError, Result};

pub fn run_file<T: Program>(program: &mut T, path: impl AsRef<Path>) -> Result<()> {
    let edgelist = graph::from_file(path)?;

    let graph = T::Input::from(edgelist);
    program.process(graph).map_err(|e| CugraError::ProgramError(program.name(), e))
}

pub fn run<T: Program>(program: &mut T) -> Result<()> {
    let args = parse();

    if args.is_present("devicelist") {
        list_devices()
    } else {
        let path = args.value_of("input").unwrap_or(program.default_graph());
        run_file(program, path)
    }
}

fn parse<'a>() -> clap::ArgMatches<'a> {
    use clap::{App, Arg};

    App::new("cugra")
        .author("Jason Priest <jpriest@mit.edu>")
        .arg(
            Arg::with_name("input")
                .help("graph file to process"),
        )
        .arg(
            Arg::with_name("devicelist")
                .help("list cuda capable devices"),
        )
        .get_matches()
}

fn list_devices() -> Result<()> {
    use cuda::driver::Device;

    let n = Device::count()? as u16;
    for i in 0..n {
        let d = Device::from_index(i)?;
        println!("device[{}]: {:?}", i, d.name()?);
    }

    Ok(())
}
