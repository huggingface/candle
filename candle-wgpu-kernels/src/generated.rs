use crate::*; 

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Pipelines{
	RmsNorm(DType, rms_norm::Functions),
	Unary(DType, unary::Functions),
	IndexSelect(DType, index_select::Functions),
	Convert(DType, convert::Functions),
	Upsample(DType, upsample::Functions),
	Matmul(DType, matmul::Functions),
	Gather(DType, gather::Functions),
	Copy(DType, copy::Functions),
	Cmp(DType, cmp::Functions),
	Reduce(DType, reduce::Functions),
	Pool2d(DType, pool2d::Functions),
	WhereCond(DType, where_cond::Functions),
	Binary(DType, binary::Functions),
	Conv2d(DType, conv2d::Functions),
	Softmax(DType, softmax::Functions),
	Conv1d(DType, conv1d::Functions),
}
impl crate::EntryPoint for Pipelines{
    fn get_entry_point(&self) -> &'static str{
        match self{
			Pipelines::RmsNorm(_, f) => f.get_entry_point(),
			Pipelines::Unary(_, f) => f.get_entry_point(),
			Pipelines::IndexSelect(_, f) => f.get_entry_point(),
			Pipelines::Convert(_, f) => f.get_entry_point(),
			Pipelines::Upsample(_, f) => f.get_entry_point(),
			Pipelines::Matmul(_, f) => f.get_entry_point(),
			Pipelines::Gather(_, f) => f.get_entry_point(),
			Pipelines::Copy(_, f) => f.get_entry_point(),
			Pipelines::Cmp(_, f) => f.get_entry_point(),
			Pipelines::Reduce(_, f) => f.get_entry_point(),
			Pipelines::Pool2d(_, f) => f.get_entry_point(),
			Pipelines::WhereCond(_, f) => f.get_entry_point(),
			Pipelines::Binary(_, f) => f.get_entry_point(),
			Pipelines::Conv2d(_, f) => f.get_entry_point(),
			Pipelines::Softmax(_, f) => f.get_entry_point(),
			Pipelines::Conv1d(_, f) => f.get_entry_point()
        }
    } 
}
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Shaders{
	RmsNorm(DType),
	Unary(DType),
	IndexSelect(DType),
	Convert(DType),
	Upsample(DType),
	Matmul(DType),
	Gather(DType),
	Copy(DType),
	Cmp(DType),
	Reduce(DType),
	Pool2d(DType),
	WhereCond(DType),
	Binary(DType),
	Conv2d(DType),
	Softmax(DType),
	Conv1d(DType),
}
impl Pipelines {
    pub fn get_shader(&self) -> Shaders{
        match self{
			Pipelines::RmsNorm(typ, _) => Shaders::RmsNorm(typ.clone()),
			Pipelines::Unary(typ, _) => Shaders::Unary(typ.clone()),
			Pipelines::IndexSelect(typ, _) => Shaders::IndexSelect(typ.clone()),
			Pipelines::Convert(typ, _) => Shaders::Convert(typ.clone()),
			Pipelines::Upsample(typ, _) => Shaders::Upsample(typ.clone()),
			Pipelines::Matmul(typ, _) => Shaders::Matmul(typ.clone()),
			Pipelines::Gather(typ, _) => Shaders::Gather(typ.clone()),
			Pipelines::Copy(typ, _) => Shaders::Copy(typ.clone()),
			Pipelines::Cmp(typ, _) => Shaders::Cmp(typ.clone()),
			Pipelines::Reduce(typ, _) => Shaders::Reduce(typ.clone()),
			Pipelines::Pool2d(typ, _) => Shaders::Pool2d(typ.clone()),
			Pipelines::WhereCond(typ, _) => Shaders::WhereCond(typ.clone()),
			Pipelines::Binary(typ, _) => Shaders::Binary(typ.clone()),
			Pipelines::Conv2d(typ, _) => Shaders::Conv2d(typ.clone()),
			Pipelines::Softmax(typ, _) => Shaders::Softmax(typ.clone()),
			Pipelines::Conv1d(typ, _) => Shaders::Conv1d(typ.clone())
        }
    }

    pub fn load_shader(&self) -> &'static str{
        match self{
		Pipelines::RmsNorm(typ, _) => rms_norm::load_shader(typ.clone()),
		Pipelines::Unary(typ, _) => unary::load_shader(typ.clone()),
		Pipelines::IndexSelect(typ, _) => index_select::load_shader(typ.clone()),
		Pipelines::Convert(typ, _) => convert::load_shader(typ.clone()),
		Pipelines::Upsample(typ, _) => upsample::load_shader(typ.clone()),
		Pipelines::Matmul(typ, _) => matmul::load_shader(typ.clone()),
		Pipelines::Gather(typ, _) => gather::load_shader(typ.clone()),
		Pipelines::Copy(typ, _) => copy::load_shader(typ.clone()),
		Pipelines::Cmp(typ, _) => cmp::load_shader(typ.clone()),
		Pipelines::Reduce(typ, _) => reduce::load_shader(typ.clone()),
		Pipelines::Pool2d(typ, _) => pool2d::load_shader(typ.clone()),
		Pipelines::WhereCond(typ, _) => where_cond::load_shader(typ.clone()),
		Pipelines::Binary(typ, _) => binary::load_shader(typ.clone()),
		Pipelines::Conv2d(typ, _) => conv2d::load_shader(typ.clone()),
		Pipelines::Softmax(typ, _) => softmax::load_shader(typ.clone()),
		Pipelines::Conv1d(typ, _) => conv1d::load_shader(typ.clone())        
        }
    }
} 

impl Shaders {
    pub fn get_shader(&self) -> Shaders{
        match self{
			Shaders::RmsNorm(typ) => Shaders::RmsNorm(typ.clone()),
			Shaders::Unary(typ) => Shaders::Unary(typ.clone()),
			Shaders::IndexSelect(typ) => Shaders::IndexSelect(typ.clone()),
			Shaders::Convert(typ) => Shaders::Convert(typ.clone()),
			Shaders::Upsample(typ) => Shaders::Upsample(typ.clone()),
			Shaders::Matmul(typ) => Shaders::Matmul(typ.clone()),
			Shaders::Gather(typ) => Shaders::Gather(typ.clone()),
			Shaders::Copy(typ) => Shaders::Copy(typ.clone()),
			Shaders::Cmp(typ) => Shaders::Cmp(typ.clone()),
			Shaders::Reduce(typ) => Shaders::Reduce(typ.clone()),
			Shaders::Pool2d(typ) => Shaders::Pool2d(typ.clone()),
			Shaders::WhereCond(typ) => Shaders::WhereCond(typ.clone()),
			Shaders::Binary(typ) => Shaders::Binary(typ.clone()),
			Shaders::Conv2d(typ) => Shaders::Conv2d(typ.clone()),
			Shaders::Softmax(typ) => Shaders::Softmax(typ.clone()),
			Shaders::Conv1d(typ) => Shaders::Conv1d(typ.clone())
        }
    }

    pub fn load_shader(&self) -> &'static str{
        match self{
		Shaders::RmsNorm(typ) => rms_norm::load_shader(typ.clone()),
		Shaders::Unary(typ) => unary::load_shader(typ.clone()),
		Shaders::IndexSelect(typ) => index_select::load_shader(typ.clone()),
		Shaders::Convert(typ) => convert::load_shader(typ.clone()),
		Shaders::Upsample(typ) => upsample::load_shader(typ.clone()),
		Shaders::Matmul(typ) => matmul::load_shader(typ.clone()),
		Shaders::Gather(typ) => gather::load_shader(typ.clone()),
		Shaders::Copy(typ) => copy::load_shader(typ.clone()),
		Shaders::Cmp(typ) => cmp::load_shader(typ.clone()),
		Shaders::Reduce(typ) => reduce::load_shader(typ.clone()),
		Shaders::Pool2d(typ) => pool2d::load_shader(typ.clone()),
		Shaders::WhereCond(typ) => where_cond::load_shader(typ.clone()),
		Shaders::Binary(typ) => binary::load_shader(typ.clone()),
		Shaders::Conv2d(typ) => conv2d::load_shader(typ.clone()),
		Shaders::Softmax(typ) => softmax::load_shader(typ.clone()),
		Shaders::Conv1d(typ) => conv1d::load_shader(typ.clone())        
        }
    }
} 

#[derive(Debug, Clone, PartialEq, Eq, Hash, std::marker::Copy)]
pub enum Constants {
    None,
	ConstIsStartoffsetZero1,
	Constv8,
	ConstDims1,
	Constv6,
	ConstDims2,
	ConstIsContiguous1,
	ConstIsStartoffsetZero3,
	Constv3,
	Constv5,
	Constv2,
	Constv4,
	ConstIsStartoffsetZero2,
	ConstIsContiguous3,
	ConstIsContiguous2,
	Constv1,
	Constv7,
	ConstDims3,
	Constv9,
	Constv0
}

impl crate::EntryPoint for Constants{
    fn get_entry_point(&self) -> &'static str{
        match self{
			Constants::ConstIsStartoffsetZero1 => "CONST_IS_STARTOFFSET_ZERO1",
			Constants::Constv8 => "CONSTV_8",
			Constants::ConstDims1 => "CONST_DIMS1",
			Constants::Constv6 => "CONSTV_6",
			Constants::ConstDims2 => "CONST_DIMS2",
			Constants::ConstIsContiguous1 => "CONST_IS_CONTIGUOUS1",
			Constants::ConstIsStartoffsetZero3 => "CONST_IS_STARTOFFSET_ZERO3",
			Constants::Constv3 => "CONSTV_3",
			Constants::Constv5 => "CONSTV_5",
			Constants::Constv2 => "CONSTV_2",
			Constants::Constv4 => "CONSTV_4",
			Constants::ConstIsStartoffsetZero2 => "CONST_IS_STARTOFFSET_ZERO2",
			Constants::ConstIsContiguous3 => "CONST_IS_CONTIGUOUS3",
			Constants::ConstIsContiguous2 => "CONST_IS_CONTIGUOUS2",
			Constants::Constv1 => "CONSTV_1",
			Constants::Constv7 => "CONSTV_7",
			Constants::ConstDims3 => "CONST_DIMS3",
			Constants::Constv9 => "CONSTV_9",
			Constants::Constv0 => "CONSTV_0",
            Constants::None => panic!("not expected")
        }
    } 
}

impl Default for Constants {
    fn default() -> Self {
        Constants::None
    }
}
pub mod binary {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{BinaryBufferFromBuffer,BinaryBufferInplace2ContiguousBoth,BinaryBufferFromBufferContiguousBoth,BinaryBufferInplace1ContiguousBoth}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::BinaryBufferFromBuffer => "ga",Functions::BinaryBufferInplace2ContiguousBoth => "gd",Functions::BinaryBufferFromBufferContiguousBoth => "gb",Functions::BinaryBufferInplace1ContiguousBoth => "gc"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/generated/binary.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/generated/binary.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/generated/binary.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        pub mod cmp {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{CmpBufferFromBuffer}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::CmpBufferFromBuffer => "ga"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/generated/cmp.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/generated/cmp.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/generated/cmp.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        pub mod conv1d {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Conv1dTranspose,Conv1d}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Conv1dTranspose => "gb",Functions::Conv1d => "ga"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/generated/conv1d.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/generated/conv1d.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/generated/conv1d.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        pub mod conv2d {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Conv2d2,Conv2d,Conv2dTranspose}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Conv2d2 => "gb",Functions::Conv2d => "ga",Functions::Conv2dTranspose => "gc"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/generated/conv2d.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/generated/conv2d.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/generated/conv2d.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        pub mod convert {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{ConvertF32ToU8,ConvertU32ToU8,ConvertU8ToF32,ConvertToU32,ConvertToF32}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::ConvertF32ToU8 => "gb",Functions::ConvertU32ToU8 => "gd",Functions::ConvertU8ToF32 => "ge",Functions::ConvertToU32 => "ga",Functions::ConvertToF32 => "gc"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/generated/convert.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/generated/convert.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/generated/convert.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        pub mod copy {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Copy3dPaddedNobatch,Copy,Copy2d2,Copy2d,Copy2dTranspose2,Copy3dPadded,CopyStrided,Copy2dTranspose,Copy3d}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Copy3dPaddedNobatch => "gi",Functions::Copy => "gb",Functions::Copy2d2 => "gd",Functions::Copy2d => "gc",Functions::Copy2dTranspose2 => "gf",Functions::Copy3dPadded => "gh",Functions::CopyStrided => "ga",Functions::Copy2dTranspose => "ge",Functions::Copy3d => "gg"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/generated/copy.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/generated/copy.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/generated/copy.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        pub mod gather {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{ScatterAddInplace,IndexAddInplace,Gather}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::ScatterAddInplace => "gb",Functions::IndexAddInplace => "gc",Functions::Gather => "ga"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/generated/gather.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/generated/gather.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/generated/gather.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        pub mod index_select {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{IndexSelect}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::IndexSelect => "ga"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/generated/index_select.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/generated/index_select.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/generated/index_select.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        pub mod matmul {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul5,Matmul1End,Matmul7,Matmul1}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul5 => "gc",Functions::Matmul1End => "gb",Functions::Matmul7 => "gd",Functions::Matmul1 => "ga"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/generated/matmul.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/generated/matmul.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/generated/matmul.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        pub mod pool2d {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{MaxPool2d,AvgPool2d}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::MaxPool2d => "ga",Functions::AvgPool2d => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/generated/pool2d.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/generated/pool2d.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/generated/pool2d.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        pub mod reduce {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{ReduceIndex,Reduce}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::ReduceIndex => "gb",Functions::Reduce => "ga"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/generated/reduce.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/generated/reduce.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/generated/reduce.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        pub mod rms_norm {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{RmsNorm}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::RmsNorm => "ga"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/generated/rms_norm.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/generated/rms_norm.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/generated/rms_norm.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        pub mod softmax {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Softmax}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Softmax => "ga"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/generated/softmax.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/generated/softmax.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/generated/softmax.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        pub mod unary {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{UnaryFromBuffer,UnaryFromBufferContiguous,UnaryInplaceContiguous}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::UnaryFromBuffer => "ga",Functions::UnaryFromBufferContiguous => "gb",Functions::UnaryInplaceContiguous => "gc"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/generated/unary.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/generated/unary.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/generated/unary.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        pub mod upsample {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Upsample1d,Upsample2d}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Upsample1d => "ga",Functions::Upsample2d => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/generated/upsample.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/generated/upsample.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/generated/upsample.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        pub mod where_cond {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{WhereCondIndexU32}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::WhereCondIndexU32 => "ga"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/generated/where_cond.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/generated/where_cond.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/generated/where_cond.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        