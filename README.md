# cuGra - CUDA Graph Framework

## Building
`cuGra` is written with [Rust]. Their homepage includes installation instructions:

```bash
curl https://sh.rustup.rs -sSf | sh
```

Afterwards, `cuGra` is simply built and ran with
```bash
cargo build --release
cargo run --bin simple_bfs -- ./examples/rMatGraph_J_5_100
```

[Rust]: https://www.rust-lang.org/en-US/ 
