/*
 * Derived from cutile-ir, unmodified:
 *   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 *   SPDX-License-Identifier: Apache-2.0
 */

//! Module-scope globals a frontend can declare.

use crate::attr::DenseElements;

/// A module-level global variable with static initialization.
///
/// Allocated in GPU global memory at module load time.
#[derive(Debug, Clone)]
pub struct Global {
    pub sym_name: String,
    pub value: DenseElements,
    pub alignment: u64,
    pub constant: bool,
    pub symbol_visibility: SymbolVisibility,
}

/// Visibility of a module global symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SymbolVisibility {
    Public = 0,
    Private = 1,
}
