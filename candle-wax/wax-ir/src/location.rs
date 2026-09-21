/*
 * Derived from cutile-ir, unmodified:
 *   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 *   SPDX-License-Identifier: Apache-2.0
 */
//! Source location tracking for debug info emission.

/// Source location attached to operations for debug info.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Location {
    /// No known source location.
    Unknown,
    /// File, line, column.
    FileLineCol {
        filename: String,
        line: u32,
        column: u32,
    },
    /// A location with full debug info scope.
    DebugInfo(DebugInfoLoc),
    /// A call-site location linking callee to caller.
    CallSite {
        callee: Box<Location>,
        caller: Box<Location>,
    },
}

/// Debug-info-annotated location with scope chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebugInfoLoc {
    pub filename: String,
    pub line: u32,
    pub column: u32,
    pub scope: DebugScope,
}

/// Debug info scope hierarchy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DebugScope {
    Subprogram(DISubprogram),
    LexicalBlock(DILexicalBlock),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DIFile {
    pub name: String,
    pub directory: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DICompileUnit {
    pub file: DIFile,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DISubprogram {
    pub file: DIFile,
    pub line: u32,
    pub name: String,
    pub linkage_name: String,
    pub compile_unit: DICompileUnit,
    pub scope_line: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DILexicalBlock {
    pub scope: Box<DebugScope>,
    pub file: DIFile,
    pub line: u32,
    pub column: u32,
}

// TODO: Only use pliron `Location`.

/// To pliron location
pub fn to_pliron(ctx: &pliron::context::Context, l: &Location) -> pliron::location::Location {
    use pliron::location::{Location as P, Source};
    let src_pos = |filename: &str, line: u32, column: u32| P::SrcPos {
        src: Source::new_from_file(ctx, filename),
        pos: pliron::combine::stream::position::SourcePosition {
            line: line as i32,
            column: column as i32,
        },
    };
    match l {
        Location::Unknown => P::Unknown,
        Location::FileLineCol {
            filename,
            line,
            column,
        } => src_pos(filename, *line, *column),
        Location::DebugInfo(d) => src_pos(&d.filename, d.line, d.column),
        Location::CallSite { callee, caller } => P::CallSite {
            callee: Box::new(to_pliron(ctx, callee)),
            caller: Box::new(to_pliron(ctx, caller)),
        },
    }
}

/// From pliron location
pub fn from_pliron(ctx: &pliron::context::Context, l: &pliron::location::Location) -> Location {
    use pliron::location::{Location as P, Source};
    match l {
        P::SrcPos { src, pos } => match src {
            Source::File(key) => Location::FileLineCol {
                filename: pliron::uniqued_any::get(ctx, *key).display().to_string(),
                line: pos.line.max(0) as u32,
                column: pos.column.max(0) as u32,
            },
            Source::InMemory => Location::Unknown,
        },
        P::CallSite { callee, caller } => Location::CallSite {
            callee: Box::new(from_pliron(ctx, callee)),
            caller: Box::new(from_pliron(ctx, caller)),
        },
        // `Named` and `Fused` have no counterpart here; neither is ever built on this side.
        P::Named { child_loc, .. } => from_pliron(ctx, child_loc),
        P::Fused { .. } | P::Unknown => Location::Unknown,
    }
}
