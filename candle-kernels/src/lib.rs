mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

mod moe_ptx {
    include!(concat!(env!("OUT_DIR"), "/moe_ptx.rs"));
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
    MoeUtils,
    MoeGguf,
    MoeWmma,
    MoeWmmaGguf,
}

pub const ALL_IDS: [Id; 15] = [
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
    Id::MoeUtils,
    Id::MoeGguf,
    Id::MoeWmma,
    Id::MoeWmmaGguf,
];

pub struct Module {
    index: usize,
    ptx: &'static str,
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn ptx(&self) -> &'static str {
        self.ptx
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

macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            ptx: ptx::$cst,
        };
    };
}

macro_rules! moe_mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            ptx: moe_ptx::$cst,
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
moe_mdl!(MOE_UTILS, MoeUtils);
moe_mdl!(MOE_GGUF, MoeGguf);
moe_mdl!(MOE_WMMA, MoeWmma);
moe_mdl!(MOE_WMMA_GGUF, MoeWmmaGguf);
