#![feature(try_from)]
extern crate cugra;
extern crate failure;

use failure::ResultExt;

use cugra::cuda::driver as cuda;
use cugra::{CompressedOutList, EdgeList};

// shrink IDs from usize -> u32
struct Graph {
    n: u32,
    m: u32,
    offsets: Vec<u32>,
    outlist: Vec<u32>,
}
impl From<EdgeList> for Graph {
    fn from(list: EdgeList) -> Self {
        let list = CompressedOutList::from(list);

        fn cast(n: usize) -> u32 {
            assert!(n < u32::max_value() as usize);
            n as u32
        }

        // cast usize down to u32, checking for overflows
        Graph {
            n: cast(list.n),
            m: cast(list.m),
            offsets: list.offsets.iter().map(|&x| cast(x)).collect(),
            outlist: list.outlist.iter().map(|&x| cast(x)).collect(),
        }
    }
}

struct Program<'a> {
    module: cuda::Module<'a>,
}

impl<'a> cugra::Program for Program<'a> {
    type Input = Graph;
    fn process(&mut self, graph: Self::Input) -> Result<(), failure::Error> {
        use std::iter::repeat;
        use std::mem::ManuallyDrop;

        let start = std::time::Instant::now();
        let bfs = self.module.function("bfs").context("loading cuda kernel")?;
        let step = self.module.function("step").context("loading cuda kernel")?;
        let reduce = self.module
            .function("reduce")
            .context("loading cuda kernel")?;

        let read_err_code = |_| {
            let err: i32 = unsafe { *self.module.get_symbol("err").unwrap() };
            failure::err_msg(match err {
                -1 => "out of memory",
                0 => "no error reported",
                _ => "unknown error",
            })
        };

        let gpu_offsets = ManuallyDrop::new(cuda::Buffer::from_slice(graph.offsets)?);
        let gpu_outlist = ManuallyDrop::new(cuda::Buffer::from_slice(graph.outlist)?);
        let gpu_parents = ManuallyDrop::new(cuda::Buffer::from_iter(
            repeat(u32::max_value()).take(graph.n as usize),
        )?);
        let gpu_frontier =
            ManuallyDrop::new(cuda::Buffer::from_iter(repeat(0).take(graph.n as usize))?);

        gpu_frontier.copy_from(&[0])?; // start from 0
        gpu_parents.copy_from(&[0])?; // set parent of start to itself
        self.module.set_symbol("offsets", gpu_offsets.addr())?;
        self.module.set_symbol("outlist", gpu_outlist.addr())?;
        self.module.set_symbol("parents", gpu_parents.addr())?;
        self.module.set_symbol("frontier", gpu_offsets.addr())?;
        self.module.set_symbol("n", &graph.n)?;
        self.module.set_symbol("err", gpu_err);

        let mut len_frontier = 1;
        while len_frontier > 0 {
            assert!(len_frontier < graph.n as usize);

            let gpu_nf =
                ManuallyDrop::new(cuda::Buffer::from_iter(repeat(0u32).take(len_frontier))?);
            let gpu_nl =
                ManuallyDrop::new(cuda::Buffer::from_iter(repeat(0u32).take(len_frontier))?);

            println!("{} {}", gpu_nf.size(), gpu_nl.size());
            println!("{} {}", gpu_nf.len(), gpu_nl.len());
            println!("{} {}", gpu_nf.addr(), gpu_nl.addr());

            self.module.set_symbol("next_frontier", gpu_nf.addr())?;
            self.module.set_symbol("next_len", gpu_nl.addr())?;

            bfs.launch(&[], cuda::Grid::x(len_frontier as u32), cuda::Block::x(1))
                .map_err(read_err_code)
                .context("bfs kernel")?;

            // flatten frontier arrays
            if len_frontier != 1 {
                let mut width = (len_frontier + 1) / 2;
                println!("do we reduce width = {}", width);
                loop {
                    reduce
                        .launch(
                            &[cuda::Any(&len_frontier), cuda::Any(&width)],
                            cuda::Grid::x(width as u32),
                            cuda::Block::x(1),
                        )
                        .map_err(read_err_code)
                        .context("reduce kernel")?;

                    if width == 1 {
                        break;
                    }
                    width = (width + 1) / 2;
                }
            }

            step.launch(&[], cuda::Grid::x(1), cuda::Block::x(1))
                .map_err(read_err_code)
                .context("step kernel")?;

            let lens = gpu_nl.read_to_vec()?;
            len_frontier = lens[0] as usize;

            ManuallyDrop::into_inner(gpu_nl);
            ManuallyDrop::into_inner(gpu_nf);
        }

        // let parents = gpu_parents.read_to_vec()?;
        // for parent in parents {
        //     println!("{}", parent);
        // }

        let end = std::time::Instant::now();
        let len = end - start;
        println!(
            "{}",
            len.as_secs() as f64 + len.subsec_nanos() as f64 * 1e-9
        );

        ManuallyDrop::into_inner(gpu_frontier);
        ManuallyDrop::into_inner(gpu_parents);
        ManuallyDrop::into_inner(gpu_outlist);
        ManuallyDrop::into_inner(gpu_offsets);

        Ok(())
    }
}

fn main() -> Result<(), failure::Error> {
    cuda::initialize()?;
    let device = cuda::Device(0).context("loading device")?;
    assert!(device.unified_memory().unwrap_or(false));
    let context = device.create_context()?;
    let ptx = cugra::compile_ptx("src/bin/simple_bfs.cu")?;
    let module = context.load_module(ptx)?;

    let mut p = Program { module };
    // cugra::run(&mut p);
    // cugra::run_file(&mut p, "examples/rMatGraph_J_5_100")?;
    cugra::run_file(&mut p, "examples/rMat_1k")?;
    Ok(())
}
