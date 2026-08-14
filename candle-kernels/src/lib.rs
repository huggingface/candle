mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
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
    ptx: &'static str,
    name: &'static str,
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn ptx(&self) -> &'static str {
        self.ptx
    }

    pub fn name(&self) -> &'static str {
        self.name
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
    ($cst:ident, $id:ident, $name:literal) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            ptx: ptx::$cst,
            name: $name,
        };
    };
}

mdl!(AFFINE, Affine, "affine");
mdl!(BINARY, Binary, "binary");
mdl!(CAST, Cast, "cast");
mdl!(CONV, Conv, "conv");
mdl!(FILL, Fill, "fill");
mdl!(INDEXING, Indexing, "indexing");
mdl!(QUANTIZED, Quantized, "quantized");
mdl!(REDUCE, Reduce, "reduce");
mdl!(SORT, Sort, "sort");
mdl!(TERNARY, Ternary, "ternary");
mdl!(UNARY, Unary, "unary");

pub mod ffi;
