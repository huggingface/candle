//! Record launch geometry directly on the IR.

use crate::build::OpId;
use pliron::context::Context;
use pliron::derive::{format, pliron_attr};

pliron::dict_key!(
    /// Launch geometry attribute key.
    ATTR_LAUNCH,
    "wax_launch"
);

#[format]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileBlock {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

/// The shape of a tile.
#[format("`[` vec($0, CharSpace(`,`)) `]`")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tile(pub Vec<usize>);

impl Tile {
    /// Total elements in the tile, across every axis.
    pub fn elems(&self) -> usize {
        self.0.iter().product()
    }
}

/// Launch defined by tile blocks and tile shape.
///
/// Tile thread is currently excluded. Policy is one tile thread per tile element.
#[pliron_attr(name = "wax.launch", verifier = "succ", format)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Launch {
    pub block: TileBlock,
    pub tile: Tile,
}

impl Launch {
    /// Record launch geometry on `func`.
    pub fn stamp(&self, ctx: &Context, func: OpId) {
        func.deref_mut(ctx)
            .attributes
            .set(ATTR_LAUNCH.clone(), self.clone());
    }

    /// Read launch geometry from IR, if present.
    pub fn of(ctx: &Context, func: OpId) -> Option<Launch> {
        func.deref(ctx)
            .attributes
            .get::<Launch>(&ATTR_LAUNCH)
            .cloned()
    }
}
