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
