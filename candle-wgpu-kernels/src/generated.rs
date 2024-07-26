use crate::*; 

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Pipelines{
	Copy(DType, copy::Functions),
	Softmax(DType, softmax::Functions),
	Pool2d(DType, pool2d::Functions),
	IndexSelect(DType, index_select::Functions),
	Convert(DType, convert::Functions),
	Upsample(DType, upsample::Functions),
	RmsNorm(DType, rms_norm::Functions),
	Binary(DType, binary::Functions),
	Matmul(DType, matmul::Functions),
	WhereCond(DType, where_cond::Functions),
	Conv2d(DType, conv2d::Functions),
	Unary(DType, unary::Functions),
	Gather(DType, gather::Functions),
	Cmp(DType, cmp::Functions),
	Reduce(DType, reduce::Functions),
}
impl crate::EntryPoint for Pipelines{
    fn get_entry_point(&self) -> &'static str{
        match self{
			Pipelines::Copy(_, f) => f.get_entry_point(),
			Pipelines::Softmax(_, f) => f.get_entry_point(),
			Pipelines::Pool2d(_, f) => f.get_entry_point(),
			Pipelines::IndexSelect(_, f) => f.get_entry_point(),
			Pipelines::Convert(_, f) => f.get_entry_point(),
			Pipelines::Upsample(_, f) => f.get_entry_point(),
			Pipelines::RmsNorm(_, f) => f.get_entry_point(),
			Pipelines::Binary(_, f) => f.get_entry_point(),
			Pipelines::Matmul(_, f) => f.get_entry_point(),
			Pipelines::WhereCond(_, f) => f.get_entry_point(),
			Pipelines::Conv2d(_, f) => f.get_entry_point(),
			Pipelines::Unary(_, f) => f.get_entry_point(),
			Pipelines::Gather(_, f) => f.get_entry_point(),
			Pipelines::Cmp(_, f) => f.get_entry_point(),
			Pipelines::Reduce(_, f) => f.get_entry_point()
        }
    } 
}
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Shaders{
	Copy(DType),
	Softmax(DType),
	Pool2d(DType),
	IndexSelect(DType),
	Convert(DType),
	Upsample(DType),
	RmsNorm(DType),
	Binary(DType),
	Matmul(DType),
	WhereCond(DType),
	Conv2d(DType),
	Unary(DType),
	Gather(DType),
	Cmp(DType),
	Reduce(DType),
}
impl Pipelines {
    pub fn get_shader(&self) -> Shaders{
        match self{
			Pipelines::Copy(typ, _) => Shaders::Copy(typ.clone()),
			Pipelines::Softmax(typ, _) => Shaders::Softmax(typ.clone()),
			Pipelines::Pool2d(typ, _) => Shaders::Pool2d(typ.clone()),
			Pipelines::IndexSelect(typ, _) => Shaders::IndexSelect(typ.clone()),
			Pipelines::Convert(typ, _) => Shaders::Convert(typ.clone()),
			Pipelines::Upsample(typ, _) => Shaders::Upsample(typ.clone()),
			Pipelines::RmsNorm(typ, _) => Shaders::RmsNorm(typ.clone()),
			Pipelines::Binary(typ, _) => Shaders::Binary(typ.clone()),
			Pipelines::Matmul(typ, _) => Shaders::Matmul(typ.clone()),
			Pipelines::WhereCond(typ, _) => Shaders::WhereCond(typ.clone()),
			Pipelines::Conv2d(typ, _) => Shaders::Conv2d(typ.clone()),
			Pipelines::Unary(typ, _) => Shaders::Unary(typ.clone()),
			Pipelines::Gather(typ, _) => Shaders::Gather(typ.clone()),
			Pipelines::Cmp(typ, _) => Shaders::Cmp(typ.clone()),
			Pipelines::Reduce(typ, _) => Shaders::Reduce(typ.clone())
        }
    }

    pub fn load_shader(&self) -> &'static str{
        match self{
		Pipelines::Copy(typ, _) => copy::load_shader(typ.clone()),
		Pipelines::Softmax(typ, _) => softmax::load_shader(typ.clone()),
		Pipelines::Pool2d(typ, _) => pool2d::load_shader(typ.clone()),
		Pipelines::IndexSelect(typ, _) => index_select::load_shader(typ.clone()),
		Pipelines::Convert(typ, _) => convert::load_shader(typ.clone()),
		Pipelines::Upsample(typ, _) => upsample::load_shader(typ.clone()),
		Pipelines::RmsNorm(typ, _) => rms_norm::load_shader(typ.clone()),
		Pipelines::Binary(typ, _) => binary::load_shader(typ.clone()),
		Pipelines::Matmul(typ, _) => matmul::load_shader(typ.clone()),
		Pipelines::WhereCond(typ, _) => where_cond::load_shader(typ.clone()),
		Pipelines::Conv2d(typ, _) => conv2d::load_shader(typ.clone()),
		Pipelines::Unary(typ, _) => unary::load_shader(typ.clone()),
		Pipelines::Gather(typ, _) => gather::load_shader(typ.clone()),
		Pipelines::Cmp(typ, _) => cmp::load_shader(typ.clone()),
		Pipelines::Reduce(typ, _) => reduce::load_shader(typ.clone())        
        }
    }
} 

impl Shaders {
    pub fn get_shader(&self) -> Shaders{
        match self{
			Shaders::Copy(typ) => Shaders::Copy(typ.clone()),
			Shaders::Softmax(typ) => Shaders::Softmax(typ.clone()),
			Shaders::Pool2d(typ) => Shaders::Pool2d(typ.clone()),
			Shaders::IndexSelect(typ) => Shaders::IndexSelect(typ.clone()),
			Shaders::Convert(typ) => Shaders::Convert(typ.clone()),
			Shaders::Upsample(typ) => Shaders::Upsample(typ.clone()),
			Shaders::RmsNorm(typ) => Shaders::RmsNorm(typ.clone()),
			Shaders::Binary(typ) => Shaders::Binary(typ.clone()),
			Shaders::Matmul(typ) => Shaders::Matmul(typ.clone()),
			Shaders::WhereCond(typ) => Shaders::WhereCond(typ.clone()),
			Shaders::Conv2d(typ) => Shaders::Conv2d(typ.clone()),
			Shaders::Unary(typ) => Shaders::Unary(typ.clone()),
			Shaders::Gather(typ) => Shaders::Gather(typ.clone()),
			Shaders::Cmp(typ) => Shaders::Cmp(typ.clone()),
			Shaders::Reduce(typ) => Shaders::Reduce(typ.clone())
        }
    }

    pub fn load_shader(&self) -> &'static str{
        match self{
		Shaders::Copy(typ) => copy::load_shader(typ.clone()),
		Shaders::Softmax(typ) => softmax::load_shader(typ.clone()),
		Shaders::Pool2d(typ) => pool2d::load_shader(typ.clone()),
		Shaders::IndexSelect(typ) => index_select::load_shader(typ.clone()),
		Shaders::Convert(typ) => convert::load_shader(typ.clone()),
		Shaders::Upsample(typ) => upsample::load_shader(typ.clone()),
		Shaders::RmsNorm(typ) => rms_norm::load_shader(typ.clone()),
		Shaders::Binary(typ) => binary::load_shader(typ.clone()),
		Shaders::Matmul(typ) => matmul::load_shader(typ.clone()),
		Shaders::WhereCond(typ) => where_cond::load_shader(typ.clone()),
		Shaders::Conv2d(typ) => conv2d::load_shader(typ.clone()),
		Shaders::Unary(typ) => unary::load_shader(typ.clone()),
		Shaders::Gather(typ) => gather::load_shader(typ.clone()),
		Shaders::Cmp(typ) => cmp::load_shader(typ.clone()),
		Shaders::Reduce(typ) => reduce::load_shader(typ.clone())        
        }
    }
} 
