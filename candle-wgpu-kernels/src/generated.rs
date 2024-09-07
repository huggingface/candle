use crate::*; 

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Pipelines{
	Binary(DType, kernels::binary::Functions),
	Cmp(DType, kernels::cmp::Functions),
	Conv1d(DType, kernels::conv1d::Functions),
	Conv2d(DType, kernels::conv2d::Functions),
	Convert(DType, kernels::convert::Functions),
	Copy(DType, kernels::copy::Functions),
	Gather(DType, kernels::gather::Functions),
	IndexSelect(DType, kernels::index_select::Functions),
	Matmul(DType, kernels::matmul::Functions),
	Pool2d(DType, kernels::pool2d::Functions),
	Reduce(DType, kernels::reduce::Functions),
	RmsNorm(DType, kernels::rms_norm::Functions),
	Matmul16x64(DType, kernels::sgemm::matmul16x64::Functions),
	Matmul1x128(DType, kernels::sgemm::matmul1x128::Functions),
	Matmul1x64(DType, kernels::sgemm::matmul1x64::Functions),
	Matmul1x64b(DType, kernels::sgemm::matmul1x64b::Functions),
	Matmul24x24(DType, kernels::sgemm::matmul24x24::Functions),
	Matmul24x48(DType, kernels::sgemm::matmul24x48::Functions),
	Matmul32x32(DType, kernels::sgemm::matmul32x32::Functions),
	Matmul64x64(DType, kernels::sgemm::matmul64x64::Functions),
	Matmul64x644x8(DType, kernels::sgemm::matmul64x64_4x8::Functions),
	Matmul64x648x8(DType, kernels::sgemm::matmul64x64_8x8::Functions),
	Softmax(DType, kernels::softmax::Functions),
	Unary(DType, kernels::unary::Functions),
	Upsample(DType, kernels::upsample::Functions),
	WhereCond(DType, kernels::where_cond::Functions),
}
impl crate::EntryPoint for Pipelines{
    fn get_entry_point(&self) -> &'static str{
        match self{
			Pipelines::Binary(_, f) => f.get_entry_point(),
			Pipelines::Cmp(_, f) => f.get_entry_point(),
			Pipelines::Conv1d(_, f) => f.get_entry_point(),
			Pipelines::Conv2d(_, f) => f.get_entry_point(),
			Pipelines::Convert(_, f) => f.get_entry_point(),
			Pipelines::Copy(_, f) => f.get_entry_point(),
			Pipelines::Gather(_, f) => f.get_entry_point(),
			Pipelines::IndexSelect(_, f) => f.get_entry_point(),
			Pipelines::Matmul(_, f) => f.get_entry_point(),
			Pipelines::Pool2d(_, f) => f.get_entry_point(),
			Pipelines::Reduce(_, f) => f.get_entry_point(),
			Pipelines::RmsNorm(_, f) => f.get_entry_point(),
			Pipelines::Matmul16x64(_, f) => f.get_entry_point(),
			Pipelines::Matmul1x128(_, f) => f.get_entry_point(),
			Pipelines::Matmul1x64(_, f) => f.get_entry_point(),
			Pipelines::Matmul1x64b(_, f) => f.get_entry_point(),
			Pipelines::Matmul24x24(_, f) => f.get_entry_point(),
			Pipelines::Matmul24x48(_, f) => f.get_entry_point(),
			Pipelines::Matmul32x32(_, f) => f.get_entry_point(),
			Pipelines::Matmul64x64(_, f) => f.get_entry_point(),
			Pipelines::Matmul64x644x8(_, f) => f.get_entry_point(),
			Pipelines::Matmul64x648x8(_, f) => f.get_entry_point(),
			Pipelines::Softmax(_, f) => f.get_entry_point(),
			Pipelines::Unary(_, f) => f.get_entry_point(),
			Pipelines::Upsample(_, f) => f.get_entry_point(),
			Pipelines::WhereCond(_, f) => f.get_entry_point()
        }
    } 
}
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Shaders{
	Binary(DType),
	Cmp(DType),
	Conv1d(DType),
	Conv2d(DType),
	Convert(DType),
	Copy(DType),
	Gather(DType),
	IndexSelect(DType),
	Matmul(DType),
	Pool2d(DType),
	Reduce(DType),
	RmsNorm(DType),
	Matmul16x64(DType),
	Matmul1x128(DType),
	Matmul1x64(DType),
	Matmul1x64b(DType),
	Matmul24x24(DType),
	Matmul24x48(DType),
	Matmul32x32(DType),
	Matmul64x64(DType),
	Matmul64x644x8(DType),
	Matmul64x648x8(DType),
	Softmax(DType),
	Unary(DType),
	Upsample(DType),
	WhereCond(DType),
}
impl Pipelines {
    pub fn get_shader(&self) -> Shaders{
        match self{
			Pipelines::Binary(typ, _) => Shaders::Binary(typ.clone()),
			Pipelines::Cmp(typ, _) => Shaders::Cmp(typ.clone()),
			Pipelines::Conv1d(typ, _) => Shaders::Conv1d(typ.clone()),
			Pipelines::Conv2d(typ, _) => Shaders::Conv2d(typ.clone()),
			Pipelines::Convert(typ, _) => Shaders::Convert(typ.clone()),
			Pipelines::Copy(typ, _) => Shaders::Copy(typ.clone()),
			Pipelines::Gather(typ, _) => Shaders::Gather(typ.clone()),
			Pipelines::IndexSelect(typ, _) => Shaders::IndexSelect(typ.clone()),
			Pipelines::Matmul(typ, _) => Shaders::Matmul(typ.clone()),
			Pipelines::Pool2d(typ, _) => Shaders::Pool2d(typ.clone()),
			Pipelines::Reduce(typ, _) => Shaders::Reduce(typ.clone()),
			Pipelines::RmsNorm(typ, _) => Shaders::RmsNorm(typ.clone()),
			Pipelines::Matmul16x64(typ, _) => Shaders::Matmul16x64(typ.clone()),
			Pipelines::Matmul1x128(typ, _) => Shaders::Matmul1x128(typ.clone()),
			Pipelines::Matmul1x64(typ, _) => Shaders::Matmul1x64(typ.clone()),
			Pipelines::Matmul1x64b(typ, _) => Shaders::Matmul1x64b(typ.clone()),
			Pipelines::Matmul24x24(typ, _) => Shaders::Matmul24x24(typ.clone()),
			Pipelines::Matmul24x48(typ, _) => Shaders::Matmul24x48(typ.clone()),
			Pipelines::Matmul32x32(typ, _) => Shaders::Matmul32x32(typ.clone()),
			Pipelines::Matmul64x64(typ, _) => Shaders::Matmul64x64(typ.clone()),
			Pipelines::Matmul64x644x8(typ, _) => Shaders::Matmul64x644x8(typ.clone()),
			Pipelines::Matmul64x648x8(typ, _) => Shaders::Matmul64x648x8(typ.clone()),
			Pipelines::Softmax(typ, _) => Shaders::Softmax(typ.clone()),
			Pipelines::Unary(typ, _) => Shaders::Unary(typ.clone()),
			Pipelines::Upsample(typ, _) => Shaders::Upsample(typ.clone()),
			Pipelines::WhereCond(typ, _) => Shaders::WhereCond(typ.clone())
        }
    }

    pub fn load_shader(&self) -> &'static str{
        match self{
		Pipelines::Binary(typ, _) => kernels::binary::load_shader(typ.clone()),
		Pipelines::Cmp(typ, _) => kernels::cmp::load_shader(typ.clone()),
		Pipelines::Conv1d(typ, _) => kernels::conv1d::load_shader(typ.clone()),
		Pipelines::Conv2d(typ, _) => kernels::conv2d::load_shader(typ.clone()),
		Pipelines::Convert(typ, _) => kernels::convert::load_shader(typ.clone()),
		Pipelines::Copy(typ, _) => kernels::copy::load_shader(typ.clone()),
		Pipelines::Gather(typ, _) => kernels::gather::load_shader(typ.clone()),
		Pipelines::IndexSelect(typ, _) => kernels::index_select::load_shader(typ.clone()),
		Pipelines::Matmul(typ, _) => kernels::matmul::load_shader(typ.clone()),
		Pipelines::Pool2d(typ, _) => kernels::pool2d::load_shader(typ.clone()),
		Pipelines::Reduce(typ, _) => kernels::reduce::load_shader(typ.clone()),
		Pipelines::RmsNorm(typ, _) => kernels::rms_norm::load_shader(typ.clone()),
		Pipelines::Matmul16x64(typ, _) => kernels::sgemm::matmul16x64::load_shader(typ.clone()),
		Pipelines::Matmul1x128(typ, _) => kernels::sgemm::matmul1x128::load_shader(typ.clone()),
		Pipelines::Matmul1x64(typ, _) => kernels::sgemm::matmul1x64::load_shader(typ.clone()),
		Pipelines::Matmul1x64b(typ, _) => kernels::sgemm::matmul1x64b::load_shader(typ.clone()),
		Pipelines::Matmul24x24(typ, _) => kernels::sgemm::matmul24x24::load_shader(typ.clone()),
		Pipelines::Matmul24x48(typ, _) => kernels::sgemm::matmul24x48::load_shader(typ.clone()),
		Pipelines::Matmul32x32(typ, _) => kernels::sgemm::matmul32x32::load_shader(typ.clone()),
		Pipelines::Matmul64x64(typ, _) => kernels::sgemm::matmul64x64::load_shader(typ.clone()),
		Pipelines::Matmul64x644x8(typ, _) => kernels::sgemm::matmul64x64_4x8::load_shader(typ.clone()),
		Pipelines::Matmul64x648x8(typ, _) => kernels::sgemm::matmul64x64_8x8::load_shader(typ.clone()),
		Pipelines::Softmax(typ, _) => kernels::softmax::load_shader(typ.clone()),
		Pipelines::Unary(typ, _) => kernels::unary::load_shader(typ.clone()),
		Pipelines::Upsample(typ, _) => kernels::upsample::load_shader(typ.clone()),
		Pipelines::WhereCond(typ, _) => kernels::where_cond::load_shader(typ.clone())        
        }
    }
} 

impl Shaders {
    pub fn get_shader(&self) -> Shaders{
        match self{
			Shaders::Binary(typ) => Shaders::Binary(typ.clone()),
			Shaders::Cmp(typ) => Shaders::Cmp(typ.clone()),
			Shaders::Conv1d(typ) => Shaders::Conv1d(typ.clone()),
			Shaders::Conv2d(typ) => Shaders::Conv2d(typ.clone()),
			Shaders::Convert(typ) => Shaders::Convert(typ.clone()),
			Shaders::Copy(typ) => Shaders::Copy(typ.clone()),
			Shaders::Gather(typ) => Shaders::Gather(typ.clone()),
			Shaders::IndexSelect(typ) => Shaders::IndexSelect(typ.clone()),
			Shaders::Matmul(typ) => Shaders::Matmul(typ.clone()),
			Shaders::Pool2d(typ) => Shaders::Pool2d(typ.clone()),
			Shaders::Reduce(typ) => Shaders::Reduce(typ.clone()),
			Shaders::RmsNorm(typ) => Shaders::RmsNorm(typ.clone()),
			Shaders::Matmul16x64(typ) => Shaders::Matmul16x64(typ.clone()),
			Shaders::Matmul1x128(typ) => Shaders::Matmul1x128(typ.clone()),
			Shaders::Matmul1x64(typ) => Shaders::Matmul1x64(typ.clone()),
			Shaders::Matmul1x64b(typ) => Shaders::Matmul1x64b(typ.clone()),
			Shaders::Matmul24x24(typ) => Shaders::Matmul24x24(typ.clone()),
			Shaders::Matmul24x48(typ) => Shaders::Matmul24x48(typ.clone()),
			Shaders::Matmul32x32(typ) => Shaders::Matmul32x32(typ.clone()),
			Shaders::Matmul64x64(typ) => Shaders::Matmul64x64(typ.clone()),
			Shaders::Matmul64x644x8(typ) => Shaders::Matmul64x644x8(typ.clone()),
			Shaders::Matmul64x648x8(typ) => Shaders::Matmul64x648x8(typ.clone()),
			Shaders::Softmax(typ) => Shaders::Softmax(typ.clone()),
			Shaders::Unary(typ) => Shaders::Unary(typ.clone()),
			Shaders::Upsample(typ) => Shaders::Upsample(typ.clone()),
			Shaders::WhereCond(typ) => Shaders::WhereCond(typ.clone())
        }
    }

    pub fn load_shader(&self) -> &'static str{
        match self{
		Shaders::Binary(typ) => kernels::binary::load_shader(typ.clone()),
		Shaders::Cmp(typ) => kernels::cmp::load_shader(typ.clone()),
		Shaders::Conv1d(typ) => kernels::conv1d::load_shader(typ.clone()),
		Shaders::Conv2d(typ) => kernels::conv2d::load_shader(typ.clone()),
		Shaders::Convert(typ) => kernels::convert::load_shader(typ.clone()),
		Shaders::Copy(typ) => kernels::copy::load_shader(typ.clone()),
		Shaders::Gather(typ) => kernels::gather::load_shader(typ.clone()),
		Shaders::IndexSelect(typ) => kernels::index_select::load_shader(typ.clone()),
		Shaders::Matmul(typ) => kernels::matmul::load_shader(typ.clone()),
		Shaders::Pool2d(typ) => kernels::pool2d::load_shader(typ.clone()),
		Shaders::Reduce(typ) => kernels::reduce::load_shader(typ.clone()),
		Shaders::RmsNorm(typ) => kernels::rms_norm::load_shader(typ.clone()),
		Shaders::Matmul16x64(typ) => kernels::sgemm::matmul16x64::load_shader(typ.clone()),
		Shaders::Matmul1x128(typ) => kernels::sgemm::matmul1x128::load_shader(typ.clone()),
		Shaders::Matmul1x64(typ) => kernels::sgemm::matmul1x64::load_shader(typ.clone()),
		Shaders::Matmul1x64b(typ) => kernels::sgemm::matmul1x64b::load_shader(typ.clone()),
		Shaders::Matmul24x24(typ) => kernels::sgemm::matmul24x24::load_shader(typ.clone()),
		Shaders::Matmul24x48(typ) => kernels::sgemm::matmul24x48::load_shader(typ.clone()),
		Shaders::Matmul32x32(typ) => kernels::sgemm::matmul32x32::load_shader(typ.clone()),
		Shaders::Matmul64x64(typ) => kernels::sgemm::matmul64x64::load_shader(typ.clone()),
		Shaders::Matmul64x644x8(typ) => kernels::sgemm::matmul64x64_4x8::load_shader(typ.clone()),
		Shaders::Matmul64x648x8(typ) => kernels::sgemm::matmul64x64_8x8::load_shader(typ.clone()),
		Shaders::Softmax(typ) => kernels::softmax::load_shader(typ.clone()),
		Shaders::Unary(typ) => kernels::unary::load_shader(typ.clone()),
		Shaders::Upsample(typ) => kernels::upsample::load_shader(typ.clone()),
		Shaders::WhereCond(typ) => kernels::where_cond::load_shader(typ.clone())        
        }
    }
} 

#[derive(Debug, Clone, PartialEq, Eq, Hash, std::marker::Copy)]
pub enum Constants {
    None,
	ConstDims3,
	Constv5,
	UseZ,
	ConstIsContiguous1,
	Constv3,
	Constv1,
	ConstIsContiguous2,
	ConstDims2,
	ConstIsStartoffsetZero2,
	ConstIsContiguous3,
	Isoutputpadded,
	Constv2,
	ConstIsStartoffsetZero1,
	Constv7,
	Constv6,
	Constv9,
	ConstDims1,
	Constv8,
	Constv4,
	Constv0,
	ConstIsStartoffsetZero3
}

impl crate::EntryPoint for Constants{
    fn get_entry_point(&self) -> &'static str{
        match self{
			Constants::ConstDims3 => "yg",
			Constants::Constv5 => "yo",
			Constants::UseZ => "yt",
			Constants::ConstIsContiguous1 => "yb",
			Constants::Constv3 => "ym",
			Constants::Constv1 => "yk",
			Constants::ConstIsContiguous2 => "ye",
			Constants::ConstDims2 => "yd",
			Constants::ConstIsStartoffsetZero2 => "yf",
			Constants::ConstIsContiguous3 => "yh",
			Constants::Isoutputpadded => "yu",
			Constants::Constv2 => "yl",
			Constants::ConstIsStartoffsetZero1 => "yc",
			Constants::Constv7 => "yq",
			Constants::Constv6 => "yp",
			Constants::Constv9 => "ys",
			Constants::ConstDims1 => "ya",
			Constants::Constv8 => "yr",
			Constants::Constv4 => "yn",
			Constants::Constv0 => "yj",
			Constants::ConstIsStartoffsetZero3 => "yi",
            Constants::None => panic!("not expected")
        }
    } 
}

impl Default for Constants {
    fn default() -> Self {
        Constants::None
    }
}
pub mod kernels {
    pub mod binary {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{BinaryBufferFromBuffer,BinaryBufferFromBufferContiguousBoth,BinaryBufferInplace2ContiguousBoth,BinaryBufferInplace1ContiguousBoth}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::BinaryBufferFromBuffer => "za",Functions::BinaryBufferFromBufferContiguousBoth => "zb",Functions::BinaryBufferInplace2ContiguousBoth => "zd",Functions::BinaryBufferInplace1ContiguousBoth => "zc"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/binary.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/binary.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/binary.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels//generated/binary.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels//generated/binary.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod cmp {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{CmpBufferFromBuffer}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::CmpBufferFromBuffer => "za"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/cmp.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/cmp.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/cmp.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels//generated/cmp.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels//generated/cmp.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod conv1d {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{Conv1dTranspose,Conv1d}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Conv1dTranspose => "zb",Functions::Conv1d => "za"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/conv1d.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/conv1d.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/conv1d.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels//generated/conv1d.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels//generated/conv1d.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod conv2d {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{Im2col,Conv2dNopadding,Conv2dLongchannels2Nopadding,Conv2d5,Conv2dKernelSize1Nopadding,Conv2dTranspose,Conv2d2,Conv2d,Conv2dLongchannels2,Conv2dLongchannel,Conv2dLongchannelNopadding,Conv2d7,Conv2dKernelSize1}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Im2col => "zl",Functions::Conv2dNopadding => "zh",Functions::Conv2dLongchannels2Nopadding => "ze",Functions::Conv2d5 => "zf",Functions::Conv2dKernelSize1Nopadding => "zj",Functions::Conv2dTranspose => "zm",Functions::Conv2d2 => "zk",Functions::Conv2d => "za",Functions::Conv2dLongchannels2 => "zd",Functions::Conv2dLongchannel => "zb",Functions::Conv2dLongchannelNopadding => "zc",Functions::Conv2d7 => "zg",Functions::Conv2dKernelSize1 => "zi"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/conv2d.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/conv2d.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/conv2d.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels//generated/conv2d.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels//generated/conv2d.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod convert {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{ConvertF32ToU8,ConvertU8ToF32,ConvertToF64,ConvertToF32,ConvertU32ToU8,ConvertToI64,ConvertToU32}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::ConvertF32ToU8 => "zd",Functions::ConvertU8ToF32 => "zg",Functions::ConvertToF64 => "zb",Functions::ConvertToF32 => "ze",Functions::ConvertU32ToU8 => "zf",Functions::ConvertToI64 => "zc",Functions::ConvertToU32 => "za"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/convert.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/convert.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/convert.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels//generated/convert.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels//generated/convert.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod copy {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{Copy2dTranspose2,CopyStrided,Copy4,Copy2d,Copy3dPaddedNobatch,Copy4dPadded,Copy3dPadded,Copy,Copy3d,Copy2d2,Copy2dTranspose}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Copy2dTranspose2 => "zg",Functions::CopyStrided => "za",Functions::Copy4 => "zc",Functions::Copy2d => "zd",Functions::Copy3dPaddedNobatch => "zj",Functions::Copy4dPadded => "zk",Functions::Copy3dPadded => "zi",Functions::Copy => "zb",Functions::Copy3d => "zh",Functions::Copy2d2 => "ze",Functions::Copy2dTranspose => "zf"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/copy.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/copy.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/copy.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels//generated/copy.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels//generated/copy.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod gather {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{IndexAddInplace,Gather,ScatterAddInplace}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::IndexAddInplace => "zc",Functions::Gather => "za",Functions::ScatterAddInplace => "zb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/gather.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/gather.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/gather.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels//generated/gather.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels//generated/gather.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod index_select {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{IndexSelectU32,IndexSelectI64}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::IndexSelectU32 => "za",Functions::IndexSelectI64 => "zb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/index_select.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/index_select.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/index_select.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels//generated/index_select.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels//generated/index_select.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod matmul {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{Matmul7,Matmul5,Matmul116,Matmul1,Matmul1End}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul7 => "ze",Functions::Matmul5 => "zd",Functions::Matmul116 => "zb",Functions::Matmul1 => "za",Functions::Matmul1End => "zc"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/matmul.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/matmul.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/matmul.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels//generated/matmul.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels//generated/matmul.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod pool2d {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{MaxPool2d,AvgPool2d}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::MaxPool2d => "za",Functions::AvgPool2d => "zb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/pool2d.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/pool2d.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/pool2d.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels//generated/pool2d.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels//generated/pool2d.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod reduce {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{ReduceIndex,Reduce}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::ReduceIndex => "zb",Functions::Reduce => "za"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/reduce.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/reduce.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/reduce.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels//generated/reduce.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels//generated/reduce.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod rms_norm {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{RmsNorm}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::RmsNorm => "za"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/rms_norm.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/rms_norm.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/rms_norm.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels//generated/rms_norm.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels//generated/rms_norm.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod softmax {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{Softmax}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Softmax => "za"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/softmax.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/softmax.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/softmax.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels//generated/softmax.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels//generated/softmax.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod unary {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{UnaryFromBuffer,UnaryFromBufferContiguous,UnaryInplaceContiguous}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::UnaryFromBuffer => "za",Functions::UnaryFromBufferContiguous => "zb",Functions::UnaryInplaceContiguous => "zc"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/unary.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/unary.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/unary.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels//generated/unary.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels//generated/unary.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod upsample {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{Upsample1d,Upsample2d}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Upsample1d => "za",Functions::Upsample2d => "zb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/upsample.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/upsample.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/upsample.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels//generated/upsample.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels//generated/upsample.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod where_cond {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{WhereCondIndexI64,WhereCondIndexU32}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::WhereCondIndexI64 => "zb",Functions::WhereCondIndexU32 => "za"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/where_cond.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/where_cond.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/where_cond.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels//generated/where_cond.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels//generated/where_cond.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod sgemm {
    pub mod matmul16x64 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "za",Functions::MatmulNoPadded => "zb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul16x64.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul16x64.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul16x64.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels/sgemm//generated/matmul16x64.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels/sgemm//generated/matmul16x64.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod matmul1x128 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "za",Functions::MatmulNoPadded => "zb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul1x128.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul1x128.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul1x128.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels/sgemm//generated/matmul1x128.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels/sgemm//generated/matmul1x128.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod matmul1x64 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{Matmul}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "za"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul1x64.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul1x64.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul1x64.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels/sgemm//generated/matmul1x64.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels/sgemm//generated/matmul1x64.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod matmul1x64b {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{Matmul}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "za"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul1x64b.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul1x64b.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul1x64b.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels/sgemm//generated/matmul1x64b.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels/sgemm//generated/matmul1x64b.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod matmul24x24 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "za",Functions::MatmulNoPadded => "zb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul24x24.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul24x24.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul24x24.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels/sgemm//generated/matmul24x24.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels/sgemm//generated/matmul24x24.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod matmul24x48 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{MatmulNoPadded,Matmul}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::MatmulNoPadded => "zb",Functions::Matmul => "za"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul24x48.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul24x48.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul24x48.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels/sgemm//generated/matmul24x48.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels/sgemm//generated/matmul24x48.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod matmul32x32 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{MatmulNoPadded,Matmul}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::MatmulNoPadded => "zb",Functions::Matmul => "za"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul32x32.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul32x32.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul32x32.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels/sgemm//generated/matmul32x32.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels/sgemm//generated/matmul32x32.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod matmul64x64 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "za",Functions::MatmulNoPadded => "zb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul64x64.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul64x64.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul64x64.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels/sgemm//generated/matmul64x64.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels/sgemm//generated/matmul64x64.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod matmul64x64_4x8 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "za",Functions::MatmulNoPadded => "zb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul64x64_4x8.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul64x64_4x8.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul64x64_4x8.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels/sgemm//generated/matmul64x64_4x8.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels/sgemm//generated/matmul64x64_4x8.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    pub mod matmul64x64_8x8 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature="wgpu_debug_serialize", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{MatmulNoPadded,Matmul}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::MatmulNoPadded => "zb",Functions::Matmul => "za"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul64x64_8x8.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul64x64_8x8.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul64x64_8x8.pwgsl_generated_u8.wgsl"),
                crate::DType::I64 => include_str!("kernels/sgemm//generated/matmul64x64_8x8.pwgsl_generated_i64.wgsl"),
                crate::DType::F64 => include_str!("kernels/sgemm//generated/matmul64x64_8x8.pwgsl_generated_f64.wgsl"),
            }
        }
    }
        
    }
}
