// TEAM-506: CUDA parity for ROCm - conditional compilation
#[cfg(feature = "cuda")]
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

#[cfg(feature = "rocm")]
mod hsaco {
    include!(concat!(env!("OUT_DIR"), "/hsaco.rs"));
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    Indexing,
    Quantized,
    Reduce,
    Sort,
    Ternary,
    Unary,
}

pub const ALL_IDS: [Id; 11] = [
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::Fill,
    Id::Indexing,
    Id::Quantized,
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
    Id::Unary,
];

pub struct Module {
    index: usize,
    #[cfg(feature = "cuda")]
    ptx: &'static str,
    #[cfg(feature = "rocm")]
    hsaco: &'static [u8],
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    #[cfg(feature = "cuda")]
    pub fn ptx(&self) -> &'static str {
        self.ptx
    }
    
    #[cfg(feature = "rocm")]
    pub fn hsaco(&self) -> &'static [u8] {
        self.hsaco
    }
}

const fn module_index(id: Id) -> usize {
    let mut i = 0;
    while i < ALL_IDS.len() {
        if ALL_IDS[i] as u32 == id as u32 {
            return i;
        }
        i += 1;
    }
    panic!("id not found")
}

// CUDA macro (existing)
#[cfg(feature = "cuda")]
macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            ptx: ptx::$cst,
        };
    };
}

// TEAM-506: ROCm macro (EXACT PARITY)
#[cfg(feature = "rocm")]
macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            hsaco: hsaco::$cst,
        };
    };
}

mdl!(AFFINE, Affine);
mdl!(BINARY, Binary);
mdl!(CAST, Cast);
mdl!(CONV, Conv);
mdl!(FILL, Fill);
mdl!(INDEXING, Indexing);
mdl!(QUANTIZED, Quantized);
mdl!(REDUCE, Reduce);
mdl!(SORT, Sort);
mdl!(TERNARY, Ternary);
mdl!(UNARY, Unary);
