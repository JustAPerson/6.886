use std;

use error::{ParseError, Result};

pub type Edge = (usize, usize);

pub struct EdgeList {
    pub n: usize,
    pub m: usize,
    pub edges: Vec<Edge>,
}

pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<EdgeList> {
    use std::fs;

    let input = fs::read_to_string(path)?;
    let lines = input.lines().collect::<Vec<_>>();

    if lines[0] != "AdjacencyGraph" {
        return Err(ParseError::UnsupportedFormat.into());
    }

    let mut nums = parse_nums(&lines)?;
    let n = nums[0] as usize;
    let m = nums[1] as usize;
    if lines.len() != n + m + 3 {
        return Err(ParseError::IncorrectLength(n + m + 3).into());
    }

    let mut offsets = nums.split_off(2); // ignore n, m
    let outlist = offsets.split_off(n); // separate out list from offsets

    offsets.push(m); // see below
    debug_assert!(offsets.len() == n + 1);

    // list of edges (a, b) in graph
    let mut pairs = Vec::with_capacity(m);
    for i in 0..n {
        let a = i; // i will become the origin of these edges
        for j in offsets[i]..offsets[i+1] { // some nodes have no out edges
            let b = outlist[j];
            pairs.push((a, b));
        }
    }

    Ok(EdgeList {
        n, m, edges: pairs,
    })
}

impl EdgeList {
    pub fn indegrees(&self) -> Vec<usize> {
        let mut v = Vec::new();
        v.resize(0, self.n);

        for &(_, b) in &self.edges {
            v[b] += 1;
        }
        v
    }

    pub fn outdegrees(&self) -> Vec<usize> {
        let mut v = Vec::new();
        v.resize(0, self.n);

        for &(a, _) in &self.edges {
            v[a] += 1;
        }
        v
    }
}

pub struct CompressedOutList {
    /// Number of vertices
    pub n: usize,
    /// Number of edges
    pub m: usize,
    pub offsets: Vec<usize>,
    pub outlist: Vec<usize>,
}

impl From<EdgeList> for CompressedOutList {
    fn from(list: EdgeList) -> Self {
        let mut edges = list.edges;

        edges.sort(); // sort by first element, then by second

        let mut offsets = Vec::with_capacity(list.n + 1);

        let mut iter = edges.iter();
        let mut offset = 0;
        for i in 0..list.n {
            offsets.push(offset);
            while offset < list.m && edges[offset].0 == i {
                offset += 1
            }
        }
        offsets.push(list.n);

        let outlist = edges.iter().map(|&(_, b)| b).collect();

        CompressedOutList {
            n: list.n,
            m: list.m,
            offsets,
            outlist,
        }
    }
}

fn parse_nums(lines: &Vec<&str>) -> Result<Vec<usize>> {
    (1..lines.len())
        .map(|i: usize| {
            let line = lines[i];
            let lineno = i + 1;
            line.trim()
                .parse::<usize>()
                .map_err(|e| ParseError::InvalidNumber(line.into(), e, lineno))
                .map_err(|e| e.into()) // cast up from ParseError
        })
        .collect()
}


