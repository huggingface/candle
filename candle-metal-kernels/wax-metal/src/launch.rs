//! Launch geometry as compiler output.

/// How to dispatch a kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Launch {
    pub grid: (usize, usize, usize),
    pub threadgroup: (usize, usize, usize),
}

impl Launch {
    pub fn new(grid: (usize, usize, usize), threadgroup: (usize, usize, usize)) -> Self {
        Self { grid, threadgroup }
    }

    /// Build from `wax` launch geometry.
    ///
    // TODO: Tile can be any rank. We use the total elements in the tile as the single value when
    // dispatching threads. May not be optimal.
    pub fn from_tile_blocks(
        block: wax_ir::passes::launch::TileBlock,
        tile: &wax_ir::passes::launch::Tile,
    ) -> Self {
        Self::new((block.x, block.y, block.z), (tile.elems(), 1, 1))
    }
}

// TODO: The first/secondary dispatch model is wrong. Should support any number of dispatches.
#[derive(Debug, Clone)]
pub struct SecondaryDispatch {
    /// Entry point name inside the same `metallib`.
    pub entry: String,
    pub launch: Launch,
    /// Runtime scalars to bind after the buffers, in order.
    pub scalar_args: Vec<i32>,
}

/// A compiled kernel bundled with everything the caller needs to run it.
// TODO: Refactor to capture multiple entries in a single kernel.
//       either something like `entries: Vec<Dispatch>` (remove `SecondaryDispatch`) or `metallib`
//       is shared between instances.
#[derive(Debug, Clone)]
pub struct LoweredKernel {
    /// The container, ready for `newLibraryWithData`.
    pub metallib: Vec<u8>,
    /// Entry point name inside `metallib`.
    pub entry: String,
    pub launch: Launch,
    /// Bytes for a scratch buffer the kernel needs. `0` implies no scratch buffer.
    // TODO: Should this be `Option<usize>` or is `0` enough?
    pub scratch_bytes: usize,
    /// Runtime scalars to bind after the buffers, in order.
    pub scalar_args: Vec<i32>,
    /// The secondary dispatch of this kernel.
    pub second: Option<SecondaryDispatch>,
}
