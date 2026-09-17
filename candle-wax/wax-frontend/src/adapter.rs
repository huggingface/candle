//! `cutile_ir::Module` as a [`WaxSource`].

use cutile_ir::bytecode::Opcode as CirOpcode;
use cutile_ir::ir as cir;
use std::collections::HashMap;
use wax_ir as wax;
use wax_ir::source::WaxSource;

pub struct CutileSource<'a> {
    m: &'a cir::Module,
    /// (op, result_index) -> value, for every result something actually uses.
    results: HashMap<(cir::OpId, u32), cir::Value>,
}

impl<'a> CutileSource<'a> {
    pub fn new(m: &'a cir::Module) -> Self {
        let mut seen: Vec<cir::Value> = Vec::new();
        let mut stack: Vec<cir::OpId> = m.functions.clone();
        while let Some(op) = stack.pop() {
            seen.extend(m.op(op).operands.iter().copied());
            for &r in &m.op(op).regions {
                for &b in &m.region(r).blocks {
                    seen.extend(m.block(b).args.iter().map(|(v, _)| *v));
                    stack.extend(m.block(b).ops.iter().copied());
                }
            }
        }
        let mut results = HashMap::new();
        for v in seen {
            if let cir::ValueProducer::OpResult { op, result_index } = m.value_data(v).producer {
                results.insert((op, result_index), v);
            }
        }
        Self { m, results }
    }
}

fn scalar(s: &cir::ScalarType) -> wax::ScalarType {
    use cir::ScalarType as C;
    match s {
        C::I1 => wax::ScalarType::I1,
        C::I4 => wax::ScalarType::I4,
        C::I8 => wax::ScalarType::I8,
        C::I16 => wax::ScalarType::I16,
        C::I32 => wax::ScalarType::I32,
        C::I64 => wax::ScalarType::I64,
        C::F16 => wax::ScalarType::F16,
        C::BF16 => wax::ScalarType::BF16,
        C::F32 => wax::ScalarType::F32,
        C::TF32 => wax::ScalarType::TF32,
        C::F64 => wax::ScalarType::F64,
        C::F8E4M3FN => wax::ScalarType::F8E4M3FN,
        C::F8E5M2 => wax::ScalarType::F8E5M2,
        C::F8E8M0FNU => wax::ScalarType::F8E8M0FNU,
        C::F4E2M1FN => wax::ScalarType::F4E2M1FN,
    }
}

fn elem(e: &cir::TileElementType) -> wax::TileElementType {
    match e {
        cir::TileElementType::Scalar(s) => wax::TileElementType::Scalar(scalar(s)),
        cir::TileElementType::Pointer(p) => wax::TileElementType::Pointer(Box::new(pointer(p))),
    }
}

fn pointer(p: &cir::PointerType) -> wax::PointerType {
    wax::PointerType {
        pointee: scalar(&p.pointee),
    }
}

fn tensor_view(t: &cir::TensorViewType) -> wax::TensorViewType {
    wax::TensorViewType {
        element_type: scalar(&t.element_type),
        shape: t.shape.clone(),
        strides: t.strides.clone(),
    }
}

fn ty(t: &cir::Type) -> wax::Type {
    use cir::Type as C;
    match t {
        C::Scalar(s) => wax::Type::Scalar(scalar(s)),
        C::Pointer(p) => wax::Type::Pointer(pointer(p)),
        C::Tile(t) => wax::Type::Tile(wax::TileType {
            shape: t.shape.clone(),
            element_type: elem(&t.element_type),
        }),
        C::TensorView(t) => wax::Type::TensorView(tensor_view(t)),
        C::PartitionView(p) => wax::Type::PartitionView(wax::PartitionViewType {
            tile_shape: p.tile_shape.clone(),
            tensor_view: tensor_view(&p.tensor_view),
            dim_map: p.dim_map.clone(),
            padding_value: p.padding_value.as_ref().map(padding),
        }),
        C::Token => wax::Type::Token,
        C::Func(f) => wax::Type::Func(wax::FuncType {
            inputs: f.inputs.iter().map(ty).collect(),
            results: f.results.iter().map(ty).collect(),
        }),
        C::StridedView(v) => wax::Type::StridedView(wax::StridedViewType {
            tile_shape: v.tile_shape.clone(),
            traversal_strides: v.traversal_strides.clone(),
            tensor_view: tensor_view(&v.tensor_view),
            dim_map: v.dim_map.clone(),
            padding_value: v.padding_value.as_ref().map(padding),
        }),
        C::GatherScatterView(v) => wax::Type::GatherScatterView(wax::GatherScatterViewType {
            tile_shape: v.tile_shape.clone(),
            tensor_view: tensor_view(&v.tensor_view),
            sparse_dim: v.sparse_dim,
            padding_value: v.padding_value.as_ref().map(padding),
        }),
    }
}

fn padding(p: &cir::PaddingValue) -> wax::PaddingValue {
    use cir::PaddingValue as C;
    match p {
        C::Zero => wax::PaddingValue::Zero,
        C::NegZero => wax::PaddingValue::NegZero,
        C::Nan => wax::PaddingValue::Nan,
        C::PosInf => wax::PaddingValue::PosInf,
        C::NegInf => wax::PaddingValue::NegInf,
    }
}

fn attr(a: &cir::Attribute) -> wax::Attribute {
    use cir::Attribute as C;
    match a {
        C::Integer(i, t) => wax::Attribute::Integer(*i, ty(t)),
        // The only conversion. cutile uses a f64, we use the bit pattern.
        C::Float(f, t) => wax::Attribute::Float(wax::FloatBits::new(*f), ty(t)),
        C::Bool(b) => wax::Attribute::Bool(*b),
        C::Type(t) => wax::Attribute::Type(ty(t)),
        C::String(s) => wax::Attribute::String(s.clone()),
        C::Array(v) => wax::Attribute::Array(v.iter().map(attr).collect()),
        C::DenseI32Array(v) => wax::Attribute::DenseI32Array(v.clone()),
        C::Dictionary(d) => {
            wax::Attribute::Dictionary(d.iter().map(|(k, v)| (k.clone(), attr(v))).collect())
        }
        C::DenseElements(d) => wax::Attribute::DenseElements(wax::DenseElements {
            element_type: ty(&d.element_type),
            shape: d.shape.clone(),
            data: d.data.clone(),
        }),
        C::DivBy(d) => wax::Attribute::DivBy(wax::DivBy {
            divisor: d.divisor,
            every: d.every,
            along: d.along,
        }),
        C::SameElements(e) => wax::Attribute::SameElements(wax::SameElements {
            values: e.values.clone(),
        }),
        C::Bounded(b) => wax::Attribute::Bounded(wax::Bounded { lb: b.lb, ub: b.ub }),
        C::OptimizationHints(h) => wax::Attribute::OptimizationHints(wax::OptimizationHints {
            entries: h
                .entries
                .iter()
                .map(|(k, vs)| {
                    (
                        k.clone(),
                        vs.iter().map(|(k, v)| (k.clone(), attr(v))).collect(),
                    )
                })
                .collect(),
        }),
    }
}

fn opcode(o: CirOpcode) -> wax::Opcode {
    // TODO: We unwrap here because the opcodes are expected to be equal.
    // Fairly safe to expect, but not guaranteed.
    wax::Opcode::try_from(o.as_u16()).unwrap()
}

impl WaxSource for CutileSource<'_> {
    type Op = cir::OpId;
    type Val = cir::Value;
    type Block = cir::BlockId;
    type Region = cir::RegionId;

    fn opcode(&self, op: Self::Op) -> wax::Opcode {
        opcode(self.m.op(op).opcode)
    }
    fn operands(&self, op: Self::Op) -> Vec<Self::Val> {
        self.m.op(op).operands.clone()
    }
    fn result_types(&self, op: Self::Op) -> Vec<wax::Type> {
        self.m.op(op).result_types.iter().map(ty).collect()
    }
    fn attributes(&self, op: Self::Op) -> Vec<(String, wax::Attribute)> {
        self.m
            .op(op)
            .attributes
            .iter()
            .map(|(k, v)| (k.clone(), attr(v)))
            .collect()
    }
    fn regions(&self, op: Self::Op) -> Vec<Self::Region> {
        self.m.op(op).regions.clone()
    }
    fn op_result(&self, op: Self::Op, result_index: u32) -> Option<Self::Val> {
        self.results.get(&(op, result_index)).copied()
    }
    fn value_type(&self, v: Self::Val) -> wax::Type {
        ty(self.m.value_type(v))
    }
    fn region_blocks(&self, r: Self::Region) -> Vec<Self::Block> {
        self.m.region(r).blocks.clone()
    }
    fn block_ops(&self, b: Self::Block) -> Vec<Self::Op> {
        self.m.block(b).ops.clone()
    }
    fn block_args(&self, b: Self::Block) -> Vec<(Self::Val, wax::Type)> {
        self.m
            .block(b)
            .args
            .iter()
            .map(|(v, t)| (*v, ty(t)))
            .collect()
    }
}
