// wax-metal/src/ir_builder.rs
//
// Typed LLVM IR builder: type-checked instruction emission that catches
// mismatches (wrong vector width, half vs float, i8 vs i16) at IR-construction
// time rather than producing malformed textual IR.

use std::fmt::Write as _;

// Type

/// LLVM IR type representation used for type-checked instruction emission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlType {
    I1,
    I8,
    I16,
    I32,
    I64,
    Half,   // f16
    BFloat, // bf16
    Float,  // f32
    Double, // f64
    /// `<N x T>` vector
    Vec(u32, Box<LlType>),
    /// `T addrspace(A)*`  (addrspace 1 = device, 2 = constant)
    Ptr {
        pointee: Box<LlType>,
        addrspace: u32,
    },
    Void,
}

impl LlType {
    /// LLVM text representation, e.g. `"i8"`, `"<64 x float>"`, `"i8 addrspace(1)*"`.
    pub fn ll_str(&self) -> String {
        match self {
            LlType::I1 => "i1".into(),
            LlType::I8 => "i8".into(),
            LlType::I16 => "i16".into(),
            LlType::I32 => "i32".into(),
            LlType::I64 => "i64".into(),
            LlType::Half => "half".into(),
            LlType::BFloat => "bfloat".into(),
            LlType::Float => "float".into(),
            LlType::Double => "double".into(),
            LlType::Vec(n, t) => format!("<{} x {}>", n, t.ll_str()),
            LlType::Ptr { pointee, addrspace } => {
                if *addrspace == 0 {
                    format!("{}*", pointee.ll_str())
                } else {
                    format!("{} addrspace({})*", pointee.ll_str(), addrspace)
                }
            }
            LlType::Void => "void".into(),
        }
    }

    pub fn device_ptr(pointee: LlType) -> LlType {
        LlType::Ptr {
            pointee: Box::new(pointee),
            addrspace: 1,
        }
    }
    pub fn const_ptr(pointee: LlType) -> LlType {
        LlType::Ptr {
            pointee: Box::new(pointee),
            addrspace: 2,
        }
    }
    pub fn vec(n: u32, elem: LlType) -> LlType {
        LlType::Vec(n, Box::new(elem))
    }

    pub fn is_int(&self) -> bool {
        matches!(
            self,
            LlType::I1 | LlType::I8 | LlType::I16 | LlType::I32 | LlType::I64
        )
    }
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            LlType::Half | LlType::BFloat | LlType::Float | LlType::Double
        )
    }

    /// Element type of a Vec, or self if scalar.
    pub fn elem(&self) -> &LlType {
        if let LlType::Vec(_, t) = self {
            t
        } else {
            self
        }
    }

    /// Pointee of a Ptr.
    pub fn pointee(&self) -> &LlType {
        if let LlType::Ptr { pointee, .. } = self {
            pointee
        } else {
            panic!("not a ptr: {self:?}")
        }
    }
}

// Value

/// A typed LLVM SSA value.  `name` is the full textual representation
/// (e.g. `%_t0`, `42`, `undef`).
#[derive(Clone, Debug)]
pub struct LlValue {
    pub name: String,
    pub ty: LlType,
}

impl LlValue {
    pub fn new(name: impl Into<String>, ty: LlType) -> Self {
        LlValue {
            name: name.into(),
            ty,
        }
    }

    /// Constant integer.
    pub fn imm(v: i64, ty: LlType) -> Self {
        assert!(ty.is_int(), "imm: must be integer type, got {ty:?}");
        LlValue {
            name: v.to_string(),
            ty,
        }
    }

    /// Typed `undef` (useful for insertelement base).
    pub fn undef(ty: LlType) -> Self {
        LlValue {
            name: "undef".into(),
            ty,
        }
    }

    /// `"TY NAME"` - the LLVM operand form with the type prefix.
    pub fn typed(&self) -> String {
        format!("{} {}", self.ty.ll_str(), self.name)
    }
}

// Instruction emitters
//
// Each function writes one `    {dst} = {opname} {args}\n` line into `out`
// and returns `LlValue { name: dst.into(), ty: result_type }`.
//
// Panics on type invariant violations - these are compiler bugs, not user errors.

// Integer arithmetic

fn emit_int_binop(op: &str, dst: &str, lhs: &LlValue, rhs: &LlValue, out: &mut String) -> LlValue {
    assert!(
        lhs.ty.is_int() && lhs.ty == rhs.ty,
        "{op}: operands must be same integer type, got {:?} and {:?}",
        lhs.ty,
        rhs.ty
    );
    let _ = writeln!(
        out,
        "    {dst} = {op} {} {}, {}",
        lhs.ty.ll_str(),
        lhs.name,
        rhs.name
    );
    LlValue::new(dst, lhs.ty.clone())
}

pub fn add(dst: &str, lhs: &LlValue, rhs: &LlValue, out: &mut String) -> LlValue {
    emit_int_binop("add", dst, lhs, rhs, out)
}
pub fn sub(dst: &str, lhs: &LlValue, rhs: &LlValue, out: &mut String) -> LlValue {
    emit_int_binop("sub", dst, lhs, rhs, out)
}
pub fn mul(dst: &str, lhs: &LlValue, rhs: &LlValue, out: &mut String) -> LlValue {
    emit_int_binop("mul", dst, lhs, rhs, out)
}
pub fn and(dst: &str, lhs: &LlValue, rhs: &LlValue, out: &mut String) -> LlValue {
    emit_int_binop("and", dst, lhs, rhs, out)
}
pub fn or(dst: &str, lhs: &LlValue, rhs: &LlValue, out: &mut String) -> LlValue {
    emit_int_binop("or", dst, lhs, rhs, out)
}
pub fn xor(dst: &str, lhs: &LlValue, rhs: &LlValue, out: &mut String) -> LlValue {
    emit_int_binop("xor", dst, lhs, rhs, out)
}

pub fn lshr(dst: &str, val: &LlValue, shift: &LlValue, out: &mut String) -> LlValue {
    assert!(
        val.ty.is_int() && val.ty == shift.ty,
        "lshr: val and shift must be same integer type, got {:?} and {:?}",
        val.ty,
        shift.ty
    );
    let _ = writeln!(
        out,
        "    {dst} = lshr {} {}, {}",
        val.ty.ll_str(),
        val.name,
        shift.name
    );
    LlValue::new(dst, val.ty.clone())
}

pub fn shl(dst: &str, val: &LlValue, shift: &LlValue, out: &mut String) -> LlValue {
    assert!(
        val.ty.is_int() && val.ty == shift.ty,
        "shl: val and shift must be same integer type, got {:?} and {:?}",
        val.ty,
        shift.ty
    );
    let _ = writeln!(
        out,
        "    {dst} = shl {} {}, {}",
        val.ty.ll_str(),
        val.name,
        shift.name
    );
    LlValue::new(dst, val.ty.clone())
}

// Float arithmetic

fn emit_float_binop(
    op: &str,
    dst: &str,
    lhs: &LlValue,
    rhs: &LlValue,
    out: &mut String,
) -> LlValue {
    assert!(
        lhs.ty.is_float() && lhs.ty == rhs.ty,
        "{op}: operands must be same float type, got {:?} and {:?}",
        lhs.ty,
        rhs.ty
    );
    let _ = writeln!(
        out,
        "    {dst} = {op} {} {}, {}",
        lhs.ty.ll_str(),
        lhs.name,
        rhs.name
    );
    LlValue::new(dst, lhs.ty.clone())
}

pub fn fadd(dst: &str, lhs: &LlValue, rhs: &LlValue, out: &mut String) -> LlValue {
    emit_float_binop("fadd", dst, lhs, rhs, out)
}
pub fn fsub(dst: &str, lhs: &LlValue, rhs: &LlValue, out: &mut String) -> LlValue {
    emit_float_binop("fsub", dst, lhs, rhs, out)
}
pub fn fmul(dst: &str, lhs: &LlValue, rhs: &LlValue, out: &mut String) -> LlValue {
    emit_float_binop("fmul", dst, lhs, rhs, out)
}
pub fn fdiv(dst: &str, lhs: &LlValue, rhs: &LlValue, out: &mut String) -> LlValue {
    emit_float_binop("fdiv", dst, lhs, rhs, out)
}

// Casts

pub fn zext(dst: &str, src: &LlValue, to: LlType, out: &mut String) -> LlValue {
    assert!(
        src.ty.is_int() && to.is_int(),
        "zext: src and dst must be integer types, got {:?} -> {:?}",
        src.ty,
        to
    );
    let _ = writeln!(
        out,
        "    {dst} = zext {} {} to {}",
        src.ty.ll_str(),
        src.name,
        to.ll_str()
    );
    LlValue::new(dst, to)
}

pub fn sext(dst: &str, src: &LlValue, to: LlType, out: &mut String) -> LlValue {
    assert!(
        src.ty.is_int() && to.is_int(),
        "sext: src and dst must be integer types, got {:?} -> {:?}",
        src.ty,
        to
    );
    let _ = writeln!(
        out,
        "    {dst} = sext {} {} to {}",
        src.ty.ll_str(),
        src.name,
        to.ll_str()
    );
    LlValue::new(dst, to)
}

pub fn trunc(dst: &str, src: &LlValue, to: LlType, out: &mut String) -> LlValue {
    assert!(
        src.ty.is_int() && to.is_int(),
        "trunc: src and dst must be integer types, got {:?} -> {:?}",
        src.ty,
        to
    );
    let _ = writeln!(
        out,
        "    {dst} = trunc {} {} to {}",
        src.ty.ll_str(),
        src.name,
        to.ll_str()
    );
    LlValue::new(dst, to)
}

pub fn fpext(dst: &str, src: &LlValue, to: LlType, out: &mut String) -> LlValue {
    assert!(
        src.ty.is_float() && to.is_float(),
        "fpext: src and dst must be float types, got {:?} -> {:?}",
        src.ty,
        to
    );
    let _ = writeln!(
        out,
        "    {dst} = fpext {} {} to {}",
        src.ty.ll_str(),
        src.name,
        to.ll_str()
    );
    LlValue::new(dst, to)
}

pub fn fptrunc(dst: &str, src: &LlValue, to: LlType, out: &mut String) -> LlValue {
    assert!(
        src.ty.is_float() && to.is_float(),
        "fptrunc: src and dst must be float types, got {:?} -> {:?}",
        src.ty,
        to
    );
    let _ = writeln!(
        out,
        "    {dst} = fptrunc {} {} to {}",
        src.ty.ll_str(),
        src.name,
        to.ll_str()
    );
    LlValue::new(dst, to)
}

pub fn bitcast(dst: &str, src: &LlValue, to: LlType, out: &mut String) -> LlValue {
    let _ = writeln!(
        out,
        "    {dst} = bitcast {} {} to {}",
        src.ty.ll_str(),
        src.name,
        to.ll_str()
    );
    LlValue::new(dst, to)
}

pub fn sitofp(dst: &str, src: &LlValue, to: LlType, out: &mut String) -> LlValue {
    assert!(
        src.ty.is_int() && to.is_float(),
        "sitofp: src must be int, dst must be float, got {:?} -> {:?}",
        src.ty,
        to
    );
    let _ = writeln!(
        out,
        "    {dst} = sitofp {} {} to {}",
        src.ty.ll_str(),
        src.name,
        to.ll_str()
    );
    LlValue::new(dst, to)
}

pub fn fptosi(dst: &str, src: &LlValue, to: LlType, out: &mut String) -> LlValue {
    assert!(
        src.ty.is_float() && to.is_int(),
        "fptosi: src must be float, dst must be int, got {:?} -> {:?}",
        src.ty,
        to
    );
    let _ = writeln!(
        out,
        "    {dst} = fptosi {} {} to {}",
        src.ty.ll_str(),
        src.name,
        to.ll_str()
    );
    LlValue::new(dst, to)
}

// Memory

/// Emit `{dst} = load {pointee}, {ptr.typed()}, align {align}`.
/// `ptr` must be a `Ptr { addrspace }`. Returns value of pointee type.
pub fn load(dst: &str, ptr: &LlValue, align: u32, out: &mut String) -> LlValue {
    let pointee = match &ptr.ty {
        LlType::Ptr { pointee, .. } => (**pointee).clone(),
        other => panic!("load: ptr must be Ptr type, got {other:?}"),
    };
    let _ = writeln!(
        out,
        "    {dst} = load {}, {}, align {align}",
        pointee.ll_str(),
        ptr.typed()
    );
    LlValue::new(dst, pointee)
}

/// Emit `store {val.typed()}, {ptr.typed()}, align {align}`.
/// `ptr.pointee()` must equal `val.ty`.
pub fn store(val: &LlValue, ptr: &LlValue, align: u32, out: &mut String) {
    let pointee = ptr.ty.pointee();
    assert!(
        *pointee == val.ty,
        "store: ptr pointee type {:?} does not match value type {:?}",
        pointee,
        val.ty
    );
    let _ = writeln!(
        out,
        "    store {}, {}, align {align}",
        val.typed(),
        ptr.typed()
    );
}

/// Emit `{dst} = getelementptr {elem_ty}, {ptr.typed()}, {idx.typed()}`.
/// `ptr` is Ptr, `idx` is I32. Returns same Ptr type as input.
pub fn gep(dst: &str, elem_ty: &LlType, ptr: &LlValue, idx: &LlValue, out: &mut String) -> LlValue {
    assert!(
        matches!(ptr.ty, LlType::Ptr { .. }),
        "gep: ptr must be Ptr type, got {:?}",
        ptr.ty
    );
    assert!(
        idx.ty == LlType::I32,
        "gep: idx must be I32, got {:?}",
        idx.ty
    );
    let _ = writeln!(
        out,
        "    {dst} = getelementptr {}, {}, {}",
        elem_ty.ll_str(),
        ptr.typed(),
        idx.typed()
    );
    LlValue::new(dst, ptr.ty.clone())
}

// Comparison

/// Emit `{dst} = icmp {pred} {lhs.typed()}, {rhs.name}`. Returns I1.
pub fn icmp(dst: &str, pred: &str, lhs: &LlValue, rhs: &LlValue, out: &mut String) -> LlValue {
    assert!(
        lhs.ty.is_int() && lhs.ty == rhs.ty,
        "icmp: operands must be same integer type, got {:?} and {:?}",
        lhs.ty,
        rhs.ty
    );
    let _ = writeln!(out, "    {dst} = icmp {pred} {}, {}", lhs.typed(), rhs.name);
    LlValue::new(dst, LlType::I1)
}

/// Emit `{dst} = fcmp {pred} {lhs.typed()}, {rhs.name}`. Returns I1.
pub fn fcmp(dst: &str, pred: &str, lhs: &LlValue, rhs: &LlValue, out: &mut String) -> LlValue {
    assert!(
        lhs.ty.is_float() && lhs.ty == rhs.ty,
        "fcmp: operands must be same float type, got {:?} and {:?}",
        lhs.ty,
        rhs.ty
    );
    let _ = writeln!(out, "    {dst} = fcmp {pred} {}, {}", lhs.typed(), rhs.name);
    LlValue::new(dst, LlType::I1)
}

// Select

/// Emit `{dst} = select i1 {cond.name}, {then_v.typed()}, {else_v.typed()}`.
/// Returns then_v type.
pub fn select(
    dst: &str,
    cond: &LlValue,
    then_v: &LlValue,
    else_v: &LlValue,
    out: &mut String,
) -> LlValue {
    assert!(
        cond.ty == LlType::I1,
        "select: cond must be I1, got {:?}",
        cond.ty
    );
    assert!(
        then_v.ty == else_v.ty,
        "select: then and else must have same type, got {:?} and {:?}",
        then_v.ty,
        else_v.ty
    );
    let _ = writeln!(
        out,
        "    {dst} = select i1 {}, {}, {}",
        cond.name,
        then_v.typed(),
        else_v.typed()
    );
    LlValue::new(dst, then_v.ty.clone())
}

// Vector

/// Emit `{dst} = extractelement {vec.typed()}, {idx.typed()}`.
/// `vec.ty` must be Vec; `idx` is I32. Returns elem type.
pub fn extractelement(dst: &str, vec: &LlValue, idx: &LlValue, out: &mut String) -> LlValue {
    let elem_ty = match &vec.ty {
        LlType::Vec(_, t) => (**t).clone(),
        other => panic!("extractelement: vec must be Vec type, got {other:?}"),
    };
    assert!(
        idx.ty == LlType::I32,
        "extractelement: idx must be I32, got {:?}",
        idx.ty
    );
    let _ = writeln!(
        out,
        "    {dst} = extractelement {}, {}",
        vec.typed(),
        idx.typed()
    );
    LlValue::new(dst, elem_ty)
}

/// Emit `{dst} = insertelement {vec.typed()}, {elem.typed()}, {idx.typed()}`.
/// `vec.ty` is Vec; `elem.ty == *vec.ty.elem()`; `idx` is I32. Returns same Vec type.
pub fn insertelement(
    dst: &str,
    vec: &LlValue,
    elem: &LlValue,
    idx: &LlValue,
    out: &mut String,
) -> LlValue {
    let expected_elem = match &vec.ty {
        LlType::Vec(_, t) => (**t).clone(),
        other => panic!("insertelement: vec must be Vec type, got {other:?}"),
    };
    assert!(
        elem.ty == expected_elem,
        "insertelement: elem type {:?} does not match vector element type {:?}",
        elem.ty,
        expected_elem
    );
    assert!(
        idx.ty == LlType::I32,
        "insertelement: idx must be I32, got {:?}",
        idx.ty
    );
    let _ = writeln!(
        out,
        "    {dst} = insertelement {}, {}, {}",
        vec.typed(),
        elem.typed(),
        idx.typed()
    );
    LlValue::new(dst, vec.ty.clone())
}
