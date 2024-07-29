use crate::*; 

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Pipelines{
	IndexSelect(DType, index_select::Functions),
	Convert(DType, convert::Functions),
	Matmul(DType, matmul::Functions),
	Conv2d(DType, conv2d::Functions),
	Copy(DType, copy::Functions),
	Unary(DType, unary::Functions),
	RmsNorm(DType, rms_norm::Functions),
	WhereCond(DType, where_cond::Functions),
	Gather(DType, gather::Functions),
	Softmax(DType, softmax::Functions),
	Binary(DType, binary::Functions),
	Cmp(DType, cmp::Functions),
	Reduce(DType, reduce::Functions),
	Pool2d(DType, pool2d::Functions),
	Conv1d(DType, conv1d::Functions),
	Upsample(DType, upsample::Functions),
}
impl crate::EntryPoint for Pipelines{
    fn get_entry_point(&self) -> &'static str{
        match self{
			Pipelines::IndexSelect(_, f) => f.get_entry_point(),
			Pipelines::Convert(_, f) => f.get_entry_point(),
			Pipelines::Matmul(_, f) => f.get_entry_point(),
			Pipelines::Conv2d(_, f) => f.get_entry_point(),
			Pipelines::Copy(_, f) => f.get_entry_point(),
			Pipelines::Unary(_, f) => f.get_entry_point(),
			Pipelines::RmsNorm(_, f) => f.get_entry_point(),
			Pipelines::WhereCond(_, f) => f.get_entry_point(),
			Pipelines::Gather(_, f) => f.get_entry_point(),
			Pipelines::Softmax(_, f) => f.get_entry_point(),
			Pipelines::Binary(_, f) => f.get_entry_point(),
			Pipelines::Cmp(_, f) => f.get_entry_point(),
			Pipelines::Reduce(_, f) => f.get_entry_point(),
			Pipelines::Pool2d(_, f) => f.get_entry_point(),
			Pipelines::Conv1d(_, f) => f.get_entry_point(),
			Pipelines::Upsample(_, f) => f.get_entry_point()
        }
    } 
}
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Shaders{
	IndexSelect(DType),
	Convert(DType),
	Matmul(DType),
	Conv2d(DType),
	Copy(DType),
	Unary(DType),
	RmsNorm(DType),
	WhereCond(DType),
	Gather(DType),
	Softmax(DType),
	Binary(DType),
	Cmp(DType),
	Reduce(DType),
	Pool2d(DType),
	Conv1d(DType),
	Upsample(DType),
}
impl Pipelines {
    pub fn get_shader(&self) -> Shaders{
        match self{
			Pipelines::IndexSelect(typ, _) => Shaders::IndexSelect(typ.clone()),
			Pipelines::Convert(typ, _) => Shaders::Convert(typ.clone()),
			Pipelines::Matmul(typ, _) => Shaders::Matmul(typ.clone()),
			Pipelines::Conv2d(typ, _) => Shaders::Conv2d(typ.clone()),
			Pipelines::Copy(typ, _) => Shaders::Copy(typ.clone()),
			Pipelines::Unary(typ, _) => Shaders::Unary(typ.clone()),
			Pipelines::RmsNorm(typ, _) => Shaders::RmsNorm(typ.clone()),
			Pipelines::WhereCond(typ, _) => Shaders::WhereCond(typ.clone()),
			Pipelines::Gather(typ, _) => Shaders::Gather(typ.clone()),
			Pipelines::Softmax(typ, _) => Shaders::Softmax(typ.clone()),
			Pipelines::Binary(typ, _) => Shaders::Binary(typ.clone()),
			Pipelines::Cmp(typ, _) => Shaders::Cmp(typ.clone()),
			Pipelines::Reduce(typ, _) => Shaders::Reduce(typ.clone()),
			Pipelines::Pool2d(typ, _) => Shaders::Pool2d(typ.clone()),
			Pipelines::Conv1d(typ, _) => Shaders::Conv1d(typ.clone()),
			Pipelines::Upsample(typ, _) => Shaders::Upsample(typ.clone())
        }
    }

    pub fn load_shader(&self) -> &'static str{
        match self{
		Pipelines::IndexSelect(typ, _) => index_select::load_shader(typ.clone()),
		Pipelines::Convert(typ, _) => convert::load_shader(typ.clone()),
		Pipelines::Matmul(typ, _) => matmul::load_shader(typ.clone()),
		Pipelines::Conv2d(typ, _) => conv2d::load_shader(typ.clone()),
		Pipelines::Copy(typ, _) => copy::load_shader(typ.clone()),
		Pipelines::Unary(typ, _) => unary::load_shader(typ.clone()),
		Pipelines::RmsNorm(typ, _) => rms_norm::load_shader(typ.clone()),
		Pipelines::WhereCond(typ, _) => where_cond::load_shader(typ.clone()),
		Pipelines::Gather(typ, _) => gather::load_shader(typ.clone()),
		Pipelines::Softmax(typ, _) => softmax::load_shader(typ.clone()),
		Pipelines::Binary(typ, _) => binary::load_shader(typ.clone()),
		Pipelines::Cmp(typ, _) => cmp::load_shader(typ.clone()),
		Pipelines::Reduce(typ, _) => reduce::load_shader(typ.clone()),
		Pipelines::Pool2d(typ, _) => pool2d::load_shader(typ.clone()),
		Pipelines::Conv1d(typ, _) => conv1d::load_shader(typ.clone()),
		Pipelines::Upsample(typ, _) => upsample::load_shader(typ.clone())        
        }
    }
} 

impl Shaders {
    pub fn get_shader(&self) -> Shaders{
        match self{
			Shaders::IndexSelect(typ) => Shaders::IndexSelect(typ.clone()),
			Shaders::Convert(typ) => Shaders::Convert(typ.clone()),
			Shaders::Matmul(typ) => Shaders::Matmul(typ.clone()),
			Shaders::Conv2d(typ) => Shaders::Conv2d(typ.clone()),
			Shaders::Copy(typ) => Shaders::Copy(typ.clone()),
			Shaders::Unary(typ) => Shaders::Unary(typ.clone()),
			Shaders::RmsNorm(typ) => Shaders::RmsNorm(typ.clone()),
			Shaders::WhereCond(typ) => Shaders::WhereCond(typ.clone()),
			Shaders::Gather(typ) => Shaders::Gather(typ.clone()),
			Shaders::Softmax(typ) => Shaders::Softmax(typ.clone()),
			Shaders::Binary(typ) => Shaders::Binary(typ.clone()),
			Shaders::Cmp(typ) => Shaders::Cmp(typ.clone()),
			Shaders::Reduce(typ) => Shaders::Reduce(typ.clone()),
			Shaders::Pool2d(typ) => Shaders::Pool2d(typ.clone()),
			Shaders::Conv1d(typ) => Shaders::Conv1d(typ.clone()),
			Shaders::Upsample(typ) => Shaders::Upsample(typ.clone())
        }
    }

    pub fn load_shader(&self) -> &'static str{
        match self{
		Shaders::IndexSelect(typ) => index_select::load_shader(typ.clone()),
		Shaders::Convert(typ) => convert::load_shader(typ.clone()),
		Shaders::Matmul(typ) => matmul::load_shader(typ.clone()),
		Shaders::Conv2d(typ) => conv2d::load_shader(typ.clone()),
		Shaders::Copy(typ) => copy::load_shader(typ.clone()),
		Shaders::Unary(typ) => unary::load_shader(typ.clone()),
		Shaders::RmsNorm(typ) => rms_norm::load_shader(typ.clone()),
		Shaders::WhereCond(typ) => where_cond::load_shader(typ.clone()),
		Shaders::Gather(typ) => gather::load_shader(typ.clone()),
		Shaders::Softmax(typ) => softmax::load_shader(typ.clone()),
		Shaders::Binary(typ) => binary::load_shader(typ.clone()),
		Shaders::Cmp(typ) => cmp::load_shader(typ.clone()),
		Shaders::Reduce(typ) => reduce::load_shader(typ.clone()),
		Shaders::Pool2d(typ) => pool2d::load_shader(typ.clone()),
		Shaders::Conv1d(typ) => conv1d::load_shader(typ.clone()),
		Shaders::Upsample(typ) => upsample::load_shader(typ.clone())        
        }
    }
} 

pub enum Constants {
	ConstIsStartoffsetZero1,
	Constv2,
	Constv5,
	ConstIsContiguous1,
	Constv3,
	ConstIsStartoffsetZero2,
	Constv8,
	Constv6,
	Constv9,
	ConstIsContiguous2,
	ConstIsContiguous3,
	Constv0,
	Constv4,
	Constv7,
	Constv1,
	ConstDims1,
	ConstDims3,
	ConstDims2,
	ConstIsStartoffsetZero3
}

impl crate::EntryPoint for Constants{
    fn get_entry_point(&self) -> &'static str{
        match self{
			Constants::ConstIsStartoffsetZero1 => "CONST_IS_STARTOFFSET_ZERO1",
			Constants::Constv2 => "CONSTV_2",
			Constants::Constv5 => "CONSTV_5",
			Constants::ConstIsContiguous1 => "CONST_IS_CONTIGUOUS1",
			Constants::Constv3 => "CONSTV_3",
			Constants::ConstIsStartoffsetZero2 => "CONST_IS_STARTOFFSET_ZERO2",
			Constants::Constv8 => "CONSTV_8",
			Constants::Constv6 => "CONSTV_6",
			Constants::Constv9 => "CONSTV_9",
			Constants::ConstIsContiguous2 => "CONST_IS_CONTIGUOUS2",
			Constants::ConstIsContiguous3 => "CONST_IS_CONTIGUOUS3",
			Constants::Constv0 => "CONSTV_0",
			Constants::Constv4 => "CONSTV_4",
			Constants::Constv7 => "CONSTV_7",
			Constants::Constv1 => "CONSTV_1",
			Constants::ConstDims1 => "CONST_DIMS1",
			Constants::ConstDims3 => "CONST_DIMS3",
			Constants::ConstDims2 => "CONST_DIMS2",
			Constants::ConstIsStartoffsetZero3 => "CONST_IS_STARTOFFSET_ZERO3"
        }
    } 
}
