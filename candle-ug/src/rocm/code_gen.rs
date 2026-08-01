//! HIP code generation for `ug` SSA kernels.
//!
//! `ug` ships a code generator per backend — `ug::cpu_code_gen` emits C,
//! `ug-cuda` emits CUDA C, `ug-metal` emits MSL — and there is no HIP one
//! upstream. `ug-cuda`'s output would in fact compile under `hipcc`, but that
//! crate depends on `cudarc` with `cuda-version-from-build-system`, whose build
//! script shells out to `nvcc`, so it cannot even be built on a ROCm-only
//! machine. Hence this module: the same SSA walk, emitting the HIP dialect
//! directly.
//!
//! The output deliberately carries no `#include`. `candle-rocm-kernels` force
//! includes its HIP shim (which pulls in `hip_runtime.h`, `hip_fp16.h` and
//! `hip_bf16.h`) ahead of every source it compiles, exactly as it does for the
//! `.cu` files shared with the CUDA backend.

use ug::lang::ssa;
use ug::Result;

struct V(ssa::VarId);

impl std::fmt::Display for V {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "__var{}", self.0.as_usize())
    }
}

struct C(ssa::Const);

fn fmt_f32(v: f32, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use std::num::FpCategory;
    match v.classify() {
        // `INFINITY`/`NAN` come from <cmath>, which the generated source does not
        // include, so the values are spelled as arithmetic instead.
        FpCategory::Nan => write!(f, "0. / 0."),
        FpCategory::Infinite if v > 0. => write!(f, "1. / 0."),
        FpCategory::Infinite => write!(f, "-1. / 0."),
        // Debug rather than Display for floats: Display does not round trip
        // f32::MIN.
        FpCategory::Zero | FpCategory::Normal | FpCategory::Subnormal => write!(f, "{v:?}"),
    }
}

impl std::fmt::Display for C {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            ssa::Const::BF16(v) => fmt_f32((*v).to_f32(), f),
            ssa::Const::F16(v) => fmt_f32((*v).to_f32(), f),
            ssa::Const::F32(v) => fmt_f32(*v, f),
            ssa::Const::I32(v) => write!(f, "{v}"),
            ssa::Const::I64(v) => write!(f, "{v}"),
        }
    }
}

struct A(ssa::A);

impl std::fmt::Display for A {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            ssa::A::Var(v) => V(v).fmt(f),
            ssa::A::Const(c) => C(c).fmt(f),
        }
    }
}

struct D(ssa::DType);

impl std::fmt::Display for D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dtype = match self.0 {
            // HIP's own spellings. `__half` matches CUDA's, bfloat16 does not.
            ssa::DType::BF16 => "__hip_bfloat16",
            ssa::DType::F16 => "__half",
            ssa::DType::F32 => "float",
            ssa::DType::I32 => "int",
            ssa::DType::I64 => "long long",
        };
        f.write_str(dtype)
    }
}

/// Emit a HIP `__global__` function named `func_name` implementing `kernel`.
pub fn gen<W: std::io::Write>(w: &mut W, func_name: &str, kernel: &ssa::Kernel) -> Result<()> {
    let instrs = kernel.instrs();
    let contains_reduce_local = instrs
        .iter()
        .any(|v| matches!(v, ssa::Instr::ReduceLocal { .. }));
    if contains_reduce_local {
        w.write_all(include_bytes!("reduce.hip"))?;
    }
    writeln!(w, "extern \"C\" __global__ void {func_name}(")?;
    for (arg_idx, &(arg, var_id)) in kernel.args().iter().enumerate() {
        let is_last = arg_idx == kernel.args().len() - 1;
        let delim = if is_last { "" } else { "," };
        let ty_ = match arg.type_() {
            ssa::Type::Value(dtype) => format!("{}", D(dtype)),
            ssa::Type::Ptr(dtype) => format!("{}*", D(dtype)),
        };
        writeln!(w, "  {ty_} {}{delim}", V(ssa::VarId::new(var_id)))?
    }
    writeln!(w, ") {{")?;

    let mut depth = 0;
    for (var_id, instr) in instrs.iter().enumerate() {
        use ssa::Instr as I;
        let var_id = V(ssa::VarId::new(var_id));
        let indent = " ".repeat(2 * depth + 2);
        match instr {
            I::DefineGlobal { index: _, dtype: _ } => {}
            I::DefineLocal { dtype, size } => {
                writeln!(w, "{indent}__shared__ {} {var_id}[{size}];", D(*dtype))?
            }
            I::DefineAcc(cst) | I::Const(cst) => {
                writeln!(w, "{indent}{} {var_id} = {};", D(cst.dtype()), C(*cst))?
            }
            I::If { cond, end_idx: _ } => {
                writeln!(w, "{indent}if ({}) {{", A(*cond))?;
                depth += 1;
            }
            I::Range {
                lo,
                up,
                step,
                end_idx: _,
            } => {
                writeln!(
                    w,
                    "{indent}for (int {var_id} = {}; {var_id} < {}; {var_id}+={step}) {{",
                    A(*lo),
                    A(*up)
                )?;
                depth += 1;
            }
            I::EndIf | I::EndRange { start_idx: _ } => {
                if depth == 0 {
                    ug::bail!("unmatched EndRange")
                }
                depth -= 1;
                let indent = " ".repeat(2 * depth + 2);
                writeln!(w, "{indent}}}")?;
            }
            I::Load { src, offset, dtype } => writeln!(
                w,
                "{indent}{} {var_id} = {}[{}];",
                D(*dtype),
                V(*src),
                A(*offset)
            )?,
            I::Assign { dst, src } => writeln!(w, "{indent}{} = {};", V(*dst), A(*src))?,
            I::Store {
                dst,
                offset,
                value,
                dtype: _,
            } => writeln!(w, "{indent}{}[{}] = {};", V(*dst), A(*offset), A(*value))?,
            I::Binary {
                op,
                lhs,
                rhs,
                dtype,
            } => {
                let op = match op {
                    ssa::BinaryOp::Add => format!("{} + {}", A(*lhs), A(*rhs)),
                    ssa::BinaryOp::Mul => format!("{} * {}", A(*lhs), A(*rhs)),
                    ssa::BinaryOp::Sub => format!("{} - {}", A(*lhs), A(*rhs)),
                    ssa::BinaryOp::Div => format!("{} / {}", A(*lhs), A(*rhs)),
                    ssa::BinaryOp::Min => format!("min({}, {})", A(*lhs), A(*rhs)),
                    ssa::BinaryOp::Max => format!("max({}, {})", A(*lhs), A(*rhs)),
                    ssa::BinaryOp::Mod => format!("{} % {}", A(*lhs), A(*rhs)),
                };
                writeln!(w, "{indent}{} {var_id} = {op};", D(*dtype))?;
            }
            I::Unary { op, arg, dtype } => {
                let op = match op {
                    ssa::UnaryOp::Exp => "__expf",
                    ssa::UnaryOp::Sqrt => "sqrtf",
                    ssa::UnaryOp::Sin => "sinf",
                    ssa::UnaryOp::Cos => "cosf",
                    ssa::UnaryOp::Neg => "-",
                    ssa::UnaryOp::Id => "",
                    ssa::UnaryOp::Cast(_) => match dtype {
                        ssa::DType::BF16 => "static_cast<__hip_bfloat16>",
                        ssa::DType::F16 => "static_cast<__half>",
                        ssa::DType::F32 => "static_cast<float>",
                        ssa::DType::I32 => "static_cast<int>",
                        ssa::DType::I64 => "static_cast<long long>",
                    },
                };
                writeln!(w, "{indent}{} {var_id} = {op}({});", D(*dtype), A(*arg))?;
            }
            I::Special(ssa::Special::ThreadIdx) => {
                writeln!(w, "{indent}int {var_id} = threadIdx.x;")?
            }
            I::Special(ssa::Special::BlockIdx) => {
                writeln!(w, "{indent}int {var_id} = blockIdx.x;")?
            }
            I::Barrier => writeln!(w, "{indent}__syncthreads();")?,
            I::ReduceLocal { op, arg, dtype } => {
                let op = match op {
                    ssa::ReduceOp::Sum => "block_reduce_sum",
                    ssa::ReduceOp::Min => "block_reduce_min",
                    ssa::ReduceOp::Max => "block_reduce_max",
                };
                writeln!(w, "{indent}{} {var_id} = {op}({});", D(*dtype), A(*arg))?;
            }
        }
    }
    writeln!(w, "}}")?;
    if depth > 0 {
        ug::bail!("unmatched Range")
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_str(name: &str, kernel: &ssa::Kernel) -> Result<String> {
        let mut buf = vec![];
        gen(&mut buf, name, kernel)?;
        Ok(String::from_utf8(buf)?)
    }

    #[test]
    fn exp_kernel_emits_hip() -> Result<()> {
        let kernel = ug::samples::ssa::exp(32)?;
        let src = gen_str("ug_exp", &kernel)?;
        assert!(src.contains("extern \"C\" __global__ void ug_exp("));
        assert!(src.contains("float* __var"));
        assert!(src.contains("__expf("));
        assert!(src.contains("int __var2 = blockIdx.x;"));
        assert!(src.contains("int __var3 = threadIdx.x;"));
        // Nothing NVIDIA-only may survive into the HIP source.
        assert!(!src.contains("__nv_bfloat16"));
        // The shim is force-included by the compiler, so the source adds none.
        assert!(!src.contains("#include"));
        Ok(())
    }

    #[test]
    fn reduce_kernel_pulls_in_the_block_reduction_helpers() -> Result<()> {
        let kernel = ug::samples::ssa::softmax_reduce(1, 32)?;
        let src = gen_str("ug_softmax", &kernel)?;
        assert!(src.contains("block_reduce_max"));
        assert!(src.contains("block_reduce_sum"));
        assert!(src.contains("__device__ __forceinline__"));
        // HIP's masked shuffle static_asserts on CUDA's 32-bit full mask, so it
        // must never be *called* (the header comment does name it).
        assert!(!src.contains("__shfl_xor_sync("));
        assert!(src.contains("__shfl_xor(x, mask, 32)"));
        Ok(())
    }

    #[test]
    fn a_kernel_with_an_unmatched_range_is_rejected() {
        let instrs = vec![
            ssa::Instr::DefineGlobal {
                index: 0,
                dtype: ssa::DType::F32,
            },
            ssa::Instr::Range {
                lo: ssa::A::Const(ssa::Const::I32(0)),
                up: ssa::A::Const(ssa::Const::I32(4)),
                step: 1,
                end_idx: ssa::VarId::new(2),
            },
        ];
        // `from_instrs` accepts it; the code generator is what has to notice.
        let Ok(kernel) = ssa::Kernel::from_instrs(instrs) else {
            return;
        };
        assert!(gen_str("ug_bad", &kernel).is_err());
    }
}
