use crate::*; 

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
	Matmul128x128(DType, kernels::sgemm::matmul128x128::Functions),
	Matmul128x128Prefetch(DType, kernels::sgemm::matmul128x128_prefetch::Functions),
	Matmul16x64(DType, kernels::sgemm::matmul16x64::Functions),
	Matmul16x64Prefetch(DType, kernels::sgemm::matmul16x64_prefetch::Functions),
	Matmul1x128(DType, kernels::sgemm::matmul1x128::Functions),
	Matmul1x128Prefetch(DType, kernels::sgemm::matmul1x128_prefetch::Functions),
	Matmul1x256(DType, kernels::sgemm::matmul1x256::Functions),
	Matmul1x256Prefetch(DType, kernels::sgemm::matmul1x256_prefetch::Functions),
	Matmul24x24(DType, kernels::sgemm::matmul24x24::Functions),
	Matmul24x24Prefetch(DType, kernels::sgemm::matmul24x24_prefetch::Functions),
	Matmul24x48(DType, kernels::sgemm::matmul24x48::Functions),
	Matmul24x48Prefetch(DType, kernels::sgemm::matmul24x48_prefetch::Functions),
	Matmul32x32(DType, kernels::sgemm::matmul32x32::Functions),
	Matmul32x32Prefetch(DType, kernels::sgemm::matmul32x32_prefetch::Functions),
	Matmul64x1284x8(DType, kernels::sgemm::matmul64x128_4x8::Functions),
	Matmul64x1284x8Prefetch(DType, kernels::sgemm::matmul64x128_4x8_prefetch::Functions),
	Matmul64x1288x8(DType, kernels::sgemm::matmul64x128_8x8::Functions),
	Matmul64x1288x8Prefetch(DType, kernels::sgemm::matmul64x128_8x8_prefetch::Functions),
	Matmul64x64(DType, kernels::sgemm::matmul64x64::Functions),
	Matmul64x648x8(DType, kernels::sgemm::matmul64x64_8x8::Functions),
	Matmul64x648x8Prefetch(DType, kernels::sgemm::matmul64x64_8x8_prefetch::Functions),
	Matmul64x64Prefetch(DType, kernels::sgemm::matmul64x64_prefetch::Functions),
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
			Pipelines::Matmul128x128(_, f) => f.get_entry_point(),
			Pipelines::Matmul128x128Prefetch(_, f) => f.get_entry_point(),
			Pipelines::Matmul16x64(_, f) => f.get_entry_point(),
			Pipelines::Matmul16x64Prefetch(_, f) => f.get_entry_point(),
			Pipelines::Matmul1x128(_, f) => f.get_entry_point(),
			Pipelines::Matmul1x128Prefetch(_, f) => f.get_entry_point(),
			Pipelines::Matmul1x256(_, f) => f.get_entry_point(),
			Pipelines::Matmul1x256Prefetch(_, f) => f.get_entry_point(),
			Pipelines::Matmul24x24(_, f) => f.get_entry_point(),
			Pipelines::Matmul24x24Prefetch(_, f) => f.get_entry_point(),
			Pipelines::Matmul24x48(_, f) => f.get_entry_point(),
			Pipelines::Matmul24x48Prefetch(_, f) => f.get_entry_point(),
			Pipelines::Matmul32x32(_, f) => f.get_entry_point(),
			Pipelines::Matmul32x32Prefetch(_, f) => f.get_entry_point(),
			Pipelines::Matmul64x1284x8(_, f) => f.get_entry_point(),
			Pipelines::Matmul64x1284x8Prefetch(_, f) => f.get_entry_point(),
			Pipelines::Matmul64x1288x8(_, f) => f.get_entry_point(),
			Pipelines::Matmul64x1288x8Prefetch(_, f) => f.get_entry_point(),
			Pipelines::Matmul64x64(_, f) => f.get_entry_point(),
			Pipelines::Matmul64x648x8(_, f) => f.get_entry_point(),
			Pipelines::Matmul64x648x8Prefetch(_, f) => f.get_entry_point(),
			Pipelines::Matmul64x64Prefetch(_, f) => f.get_entry_point(),
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
	Matmul128x128(DType),
	Matmul128x128Prefetch(DType),
	Matmul16x64(DType),
	Matmul16x64Prefetch(DType),
	Matmul1x128(DType),
	Matmul1x128Prefetch(DType),
	Matmul1x256(DType),
	Matmul1x256Prefetch(DType),
	Matmul24x24(DType),
	Matmul24x24Prefetch(DType),
	Matmul24x48(DType),
	Matmul24x48Prefetch(DType),
	Matmul32x32(DType),
	Matmul32x32Prefetch(DType),
	Matmul64x1284x8(DType),
	Matmul64x1284x8Prefetch(DType),
	Matmul64x1288x8(DType),
	Matmul64x1288x8Prefetch(DType),
	Matmul64x64(DType),
	Matmul64x648x8(DType),
	Matmul64x648x8Prefetch(DType),
	Matmul64x64Prefetch(DType),
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
			Pipelines::Matmul128x128(typ, _) => Shaders::Matmul128x128(typ.clone()),
			Pipelines::Matmul128x128Prefetch(typ, _) => Shaders::Matmul128x128Prefetch(typ.clone()),
			Pipelines::Matmul16x64(typ, _) => Shaders::Matmul16x64(typ.clone()),
			Pipelines::Matmul16x64Prefetch(typ, _) => Shaders::Matmul16x64Prefetch(typ.clone()),
			Pipelines::Matmul1x128(typ, _) => Shaders::Matmul1x128(typ.clone()),
			Pipelines::Matmul1x128Prefetch(typ, _) => Shaders::Matmul1x128Prefetch(typ.clone()),
			Pipelines::Matmul1x256(typ, _) => Shaders::Matmul1x256(typ.clone()),
			Pipelines::Matmul1x256Prefetch(typ, _) => Shaders::Matmul1x256Prefetch(typ.clone()),
			Pipelines::Matmul24x24(typ, _) => Shaders::Matmul24x24(typ.clone()),
			Pipelines::Matmul24x24Prefetch(typ, _) => Shaders::Matmul24x24Prefetch(typ.clone()),
			Pipelines::Matmul24x48(typ, _) => Shaders::Matmul24x48(typ.clone()),
			Pipelines::Matmul24x48Prefetch(typ, _) => Shaders::Matmul24x48Prefetch(typ.clone()),
			Pipelines::Matmul32x32(typ, _) => Shaders::Matmul32x32(typ.clone()),
			Pipelines::Matmul32x32Prefetch(typ, _) => Shaders::Matmul32x32Prefetch(typ.clone()),
			Pipelines::Matmul64x1284x8(typ, _) => Shaders::Matmul64x1284x8(typ.clone()),
			Pipelines::Matmul64x1284x8Prefetch(typ, _) => Shaders::Matmul64x1284x8Prefetch(typ.clone()),
			Pipelines::Matmul64x1288x8(typ, _) => Shaders::Matmul64x1288x8(typ.clone()),
			Pipelines::Matmul64x1288x8Prefetch(typ, _) => Shaders::Matmul64x1288x8Prefetch(typ.clone()),
			Pipelines::Matmul64x64(typ, _) => Shaders::Matmul64x64(typ.clone()),
			Pipelines::Matmul64x648x8(typ, _) => Shaders::Matmul64x648x8(typ.clone()),
			Pipelines::Matmul64x648x8Prefetch(typ, _) => Shaders::Matmul64x648x8Prefetch(typ.clone()),
			Pipelines::Matmul64x64Prefetch(typ, _) => Shaders::Matmul64x64Prefetch(typ.clone()),
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
		Pipelines::Matmul128x128(typ, _) => kernels::sgemm::matmul128x128::load_shader(typ.clone()),
		Pipelines::Matmul128x128Prefetch(typ, _) => kernels::sgemm::matmul128x128_prefetch::load_shader(typ.clone()),
		Pipelines::Matmul16x64(typ, _) => kernels::sgemm::matmul16x64::load_shader(typ.clone()),
		Pipelines::Matmul16x64Prefetch(typ, _) => kernels::sgemm::matmul16x64_prefetch::load_shader(typ.clone()),
		Pipelines::Matmul1x128(typ, _) => kernels::sgemm::matmul1x128::load_shader(typ.clone()),
		Pipelines::Matmul1x128Prefetch(typ, _) => kernels::sgemm::matmul1x128_prefetch::load_shader(typ.clone()),
		Pipelines::Matmul1x256(typ, _) => kernels::sgemm::matmul1x256::load_shader(typ.clone()),
		Pipelines::Matmul1x256Prefetch(typ, _) => kernels::sgemm::matmul1x256_prefetch::load_shader(typ.clone()),
		Pipelines::Matmul24x24(typ, _) => kernels::sgemm::matmul24x24::load_shader(typ.clone()),
		Pipelines::Matmul24x24Prefetch(typ, _) => kernels::sgemm::matmul24x24_prefetch::load_shader(typ.clone()),
		Pipelines::Matmul24x48(typ, _) => kernels::sgemm::matmul24x48::load_shader(typ.clone()),
		Pipelines::Matmul24x48Prefetch(typ, _) => kernels::sgemm::matmul24x48_prefetch::load_shader(typ.clone()),
		Pipelines::Matmul32x32(typ, _) => kernels::sgemm::matmul32x32::load_shader(typ.clone()),
		Pipelines::Matmul32x32Prefetch(typ, _) => kernels::sgemm::matmul32x32_prefetch::load_shader(typ.clone()),
		Pipelines::Matmul64x1284x8(typ, _) => kernels::sgemm::matmul64x128_4x8::load_shader(typ.clone()),
		Pipelines::Matmul64x1284x8Prefetch(typ, _) => kernels::sgemm::matmul64x128_4x8_prefetch::load_shader(typ.clone()),
		Pipelines::Matmul64x1288x8(typ, _) => kernels::sgemm::matmul64x128_8x8::load_shader(typ.clone()),
		Pipelines::Matmul64x1288x8Prefetch(typ, _) => kernels::sgemm::matmul64x128_8x8_prefetch::load_shader(typ.clone()),
		Pipelines::Matmul64x64(typ, _) => kernels::sgemm::matmul64x64::load_shader(typ.clone()),
		Pipelines::Matmul64x648x8(typ, _) => kernels::sgemm::matmul64x64_8x8::load_shader(typ.clone()),
		Pipelines::Matmul64x648x8Prefetch(typ, _) => kernels::sgemm::matmul64x64_8x8_prefetch::load_shader(typ.clone()),
		Pipelines::Matmul64x64Prefetch(typ, _) => kernels::sgemm::matmul64x64_prefetch::load_shader(typ.clone()),
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
			Shaders::Matmul128x128(typ) => Shaders::Matmul128x128(typ.clone()),
			Shaders::Matmul128x128Prefetch(typ) => Shaders::Matmul128x128Prefetch(typ.clone()),
			Shaders::Matmul16x64(typ) => Shaders::Matmul16x64(typ.clone()),
			Shaders::Matmul16x64Prefetch(typ) => Shaders::Matmul16x64Prefetch(typ.clone()),
			Shaders::Matmul1x128(typ) => Shaders::Matmul1x128(typ.clone()),
			Shaders::Matmul1x128Prefetch(typ) => Shaders::Matmul1x128Prefetch(typ.clone()),
			Shaders::Matmul1x256(typ) => Shaders::Matmul1x256(typ.clone()),
			Shaders::Matmul1x256Prefetch(typ) => Shaders::Matmul1x256Prefetch(typ.clone()),
			Shaders::Matmul24x24(typ) => Shaders::Matmul24x24(typ.clone()),
			Shaders::Matmul24x24Prefetch(typ) => Shaders::Matmul24x24Prefetch(typ.clone()),
			Shaders::Matmul24x48(typ) => Shaders::Matmul24x48(typ.clone()),
			Shaders::Matmul24x48Prefetch(typ) => Shaders::Matmul24x48Prefetch(typ.clone()),
			Shaders::Matmul32x32(typ) => Shaders::Matmul32x32(typ.clone()),
			Shaders::Matmul32x32Prefetch(typ) => Shaders::Matmul32x32Prefetch(typ.clone()),
			Shaders::Matmul64x1284x8(typ) => Shaders::Matmul64x1284x8(typ.clone()),
			Shaders::Matmul64x1284x8Prefetch(typ) => Shaders::Matmul64x1284x8Prefetch(typ.clone()),
			Shaders::Matmul64x1288x8(typ) => Shaders::Matmul64x1288x8(typ.clone()),
			Shaders::Matmul64x1288x8Prefetch(typ) => Shaders::Matmul64x1288x8Prefetch(typ.clone()),
			Shaders::Matmul64x64(typ) => Shaders::Matmul64x64(typ.clone()),
			Shaders::Matmul64x648x8(typ) => Shaders::Matmul64x648x8(typ.clone()),
			Shaders::Matmul64x648x8Prefetch(typ) => Shaders::Matmul64x648x8Prefetch(typ.clone()),
			Shaders::Matmul64x64Prefetch(typ) => Shaders::Matmul64x64Prefetch(typ.clone()),
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
		Shaders::Matmul128x128(typ) => kernels::sgemm::matmul128x128::load_shader(typ.clone()),
		Shaders::Matmul128x128Prefetch(typ) => kernels::sgemm::matmul128x128_prefetch::load_shader(typ.clone()),
		Shaders::Matmul16x64(typ) => kernels::sgemm::matmul16x64::load_shader(typ.clone()),
		Shaders::Matmul16x64Prefetch(typ) => kernels::sgemm::matmul16x64_prefetch::load_shader(typ.clone()),
		Shaders::Matmul1x128(typ) => kernels::sgemm::matmul1x128::load_shader(typ.clone()),
		Shaders::Matmul1x128Prefetch(typ) => kernels::sgemm::matmul1x128_prefetch::load_shader(typ.clone()),
		Shaders::Matmul1x256(typ) => kernels::sgemm::matmul1x256::load_shader(typ.clone()),
		Shaders::Matmul1x256Prefetch(typ) => kernels::sgemm::matmul1x256_prefetch::load_shader(typ.clone()),
		Shaders::Matmul24x24(typ) => kernels::sgemm::matmul24x24::load_shader(typ.clone()),
		Shaders::Matmul24x24Prefetch(typ) => kernels::sgemm::matmul24x24_prefetch::load_shader(typ.clone()),
		Shaders::Matmul24x48(typ) => kernels::sgemm::matmul24x48::load_shader(typ.clone()),
		Shaders::Matmul24x48Prefetch(typ) => kernels::sgemm::matmul24x48_prefetch::load_shader(typ.clone()),
		Shaders::Matmul32x32(typ) => kernels::sgemm::matmul32x32::load_shader(typ.clone()),
		Shaders::Matmul32x32Prefetch(typ) => kernels::sgemm::matmul32x32_prefetch::load_shader(typ.clone()),
		Shaders::Matmul64x1284x8(typ) => kernels::sgemm::matmul64x128_4x8::load_shader(typ.clone()),
		Shaders::Matmul64x1284x8Prefetch(typ) => kernels::sgemm::matmul64x128_4x8_prefetch::load_shader(typ.clone()),
		Shaders::Matmul64x1288x8(typ) => kernels::sgemm::matmul64x128_8x8::load_shader(typ.clone()),
		Shaders::Matmul64x1288x8Prefetch(typ) => kernels::sgemm::matmul64x128_8x8_prefetch::load_shader(typ.clone()),
		Shaders::Matmul64x64(typ) => kernels::sgemm::matmul64x64::load_shader(typ.clone()),
		Shaders::Matmul64x648x8(typ) => kernels::sgemm::matmul64x64_8x8::load_shader(typ.clone()),
		Shaders::Matmul64x648x8Prefetch(typ) => kernels::sgemm::matmul64x64_8x8_prefetch::load_shader(typ.clone()),
		Shaders::Matmul64x64Prefetch(typ) => kernels::sgemm::matmul64x64_prefetch::load_shader(typ.clone()),
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
	Constv7,
	Constv3,
	Isoutputpadded,
	Constv0,
	ConstDims2,
	Constv2,
	Constv6,
	Preloada,
	ConstIsContiguous2,
	ConstIsStartoffsetZero3,
	Constv4,
	ConstIsStartoffsetZero1,
	Constv9,
	Constv5,
	UseZ,
	Constv1,
	ConstDims1,
	ConstIsStartoffsetZero2,
	Constv8,
	Preloadb,
	ConstIsContiguous1,
	ConstIsContiguous3,
	ConstDims3
}

impl crate::EntryPoint for Constants{
    fn get_entry_point(&self) -> &'static str{
        match self{
			Constants::Constv7 => "CONSTV_7",
			Constants::Constv3 => "CONSTV_3",
			Constants::Isoutputpadded => "IsOutputPadded",
			Constants::Constv0 => "CONSTV_0",
			Constants::ConstDims2 => "CONST_DIMS2",
			Constants::Constv2 => "CONSTV_2",
			Constants::Constv6 => "CONSTV_6",
			Constants::Preloada => "PreLoadA",
			Constants::ConstIsContiguous2 => "CONST_IS_CONTIGUOUS2",
			Constants::ConstIsStartoffsetZero3 => "CONST_IS_STARTOFFSET_ZERO3",
			Constants::Constv4 => "CONSTV_4",
			Constants::ConstIsStartoffsetZero1 => "CONST_IS_STARTOFFSET_ZERO1",
			Constants::Constv9 => "CONSTV_9",
			Constants::Constv5 => "CONSTV_5",
			Constants::UseZ => "USE_Z",
			Constants::Constv1 => "CONSTV_1",
			Constants::ConstDims1 => "CONST_DIMS1",
			Constants::ConstIsStartoffsetZero2 => "CONST_IS_STARTOFFSET_ZERO2",
			Constants::Constv8 => "CONSTV_8",
			Constants::Preloadb => "PreLoadB",
			Constants::ConstIsContiguous1 => "CONST_IS_CONTIGUOUS1",
			Constants::ConstIsContiguous3 => "CONST_IS_CONTIGUOUS3",
			Constants::ConstDims3 => "CONST_DIMS3",
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
        pub enum Functions{BinaryBufferInplace2ContiguousBoth,BinaryBufferInplace1ContiguousBoth,BinaryBufferFromBufferContiguousBoth,BinaryBufferFromBuffer}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::BinaryBufferInplace2ContiguousBoth => "gd",Functions::BinaryBufferInplace1ContiguousBoth => "gc",Functions::BinaryBufferFromBufferContiguousBoth => "gb",Functions::BinaryBufferFromBuffer => "ga"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/binary.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/binary.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/binary.pwgsl_generated_u8.wgsl"),
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
                crate::DType::F32 => include_str!("kernels//generated/cmp.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/cmp.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/cmp.pwgsl_generated_u8.wgsl"),
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
                crate::DType::F32 => include_str!("kernels//generated/conv1d.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/conv1d.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/conv1d.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod conv2d {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Conv2d,Conv2d2,Conv2dTranspose}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Conv2d => "ga",Functions::Conv2d2 => "gb",Functions::Conv2dTranspose => "gc"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/conv2d.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/conv2d.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/conv2d.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod convert {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{ConvertU32ToU8,ConvertU8ToF32,ConvertF32ToU8,ConvertToF32,ConvertToU32}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::ConvertU32ToU8 => "gd",Functions::ConvertU8ToF32 => "ge",Functions::ConvertF32ToU8 => "gb",Functions::ConvertToF32 => "gc",Functions::ConvertToU32 => "ga"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/convert.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/convert.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/convert.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod copy {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Copy3dPaddedNobatch,Copy2d2,Copy3dPadded,CopyStrided,Copy,Copy2dTranspose2,Copy2d,Copy3d,Copy4,Copy2dTranspose}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Copy3dPaddedNobatch => "gj",Functions::Copy2d2 => "ge",Functions::Copy3dPadded => "gi",Functions::CopyStrided => "ga",Functions::Copy => "gb",Functions::Copy2dTranspose2 => "gg",Functions::Copy2d => "gd",Functions::Copy3d => "gh",Functions::Copy4 => "gc",Functions::Copy2dTranspose => "gf"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/copy.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/copy.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/copy.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod gather {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{IndexAddInplace,ScatterAddInplace,Gather}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::IndexAddInplace => "gc",Functions::ScatterAddInplace => "gb",Functions::Gather => "ga"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/gather.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/gather.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/gather.pwgsl_generated_u8.wgsl"),
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
                crate::DType::F32 => include_str!("kernels//generated/index_select.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/index_select.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/index_select.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul116,Matmul1End,Matmul1,Matmul5,Matmul7}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul116 => "gb",Functions::Matmul1End => "gc",Functions::Matmul1 => "ga",Functions::Matmul5 => "gd",Functions::Matmul7 => "ge"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/matmul.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/matmul.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/matmul.pwgsl_generated_u8.wgsl"),
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
                crate::DType::F32 => include_str!("kernels//generated/pool2d.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/pool2d.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/pool2d.pwgsl_generated_u8.wgsl"),
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
                crate::DType::F32 => include_str!("kernels//generated/reduce.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/reduce.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/reduce.pwgsl_generated_u8.wgsl"),
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
                crate::DType::F32 => include_str!("kernels//generated/rms_norm.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/rms_norm.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/rms_norm.pwgsl_generated_u8.wgsl"),
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
                crate::DType::F32 => include_str!("kernels//generated/softmax.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/softmax.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/softmax.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod unary {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{UnaryFromBufferContiguous,UnaryFromBuffer,UnaryInplaceContiguous}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::UnaryFromBufferContiguous => "gb",Functions::UnaryFromBuffer => "ga",Functions::UnaryInplaceContiguous => "gc"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels//generated/unary.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/unary.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/unary.pwgsl_generated_u8.wgsl"),
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
                crate::DType::F32 => include_str!("kernels//generated/upsample.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/upsample.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/upsample.pwgsl_generated_u8.wgsl"),
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
                crate::DType::F32 => include_str!("kernels//generated/where_cond.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels//generated/where_cond.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels//generated/where_cond.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod sgemm {
    pub mod matmul128x128 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul128x128.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul128x128.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul128x128.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul128x128_prefetch {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul128x128_prefetch.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul128x128_prefetch.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul128x128_prefetch.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul16x64 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{MatmulNoPadded,Matmul}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::MatmulNoPadded => "gb",Functions::Matmul => "ga"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul16x64.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul16x64.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul16x64.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul16x64_prefetch {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul16x64_prefetch.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul16x64_prefetch.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul16x64_prefetch.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul1x128 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{MatmulNoPadded,Matmul}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::MatmulNoPadded => "gb",Functions::Matmul => "ga"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul1x128.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul1x128.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul1x128.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul1x128_prefetch {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul1x128_prefetch.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul1x128_prefetch.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul1x128_prefetch.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul1x256 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul1x256.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul1x256.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul1x256.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul1x256_prefetch {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{MatmulNoPadded,Matmul}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::MatmulNoPadded => "gb",Functions::Matmul => "ga"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul1x256_prefetch.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul1x256_prefetch.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul1x256_prefetch.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul24x24 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul24x24.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul24x24.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul24x24.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul24x24_prefetch {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul24x24_prefetch.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul24x24_prefetch.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul24x24_prefetch.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul24x48 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul24x48.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul24x48.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul24x48.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul24x48_prefetch {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul24x48_prefetch.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul24x48_prefetch.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul24x48_prefetch.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul32x32 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul32x32.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul32x32.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul32x32.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul32x32_prefetch {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul32x32_prefetch.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul32x32_prefetch.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul32x32_prefetch.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul64x128_4x8 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul64x128_4x8.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul64x128_4x8.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul64x128_4x8.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul64x128_4x8_prefetch {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{MatmulNoPadded,Matmul}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::MatmulNoPadded => "gb",Functions::Matmul => "ga"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul64x128_4x8_prefetch.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul64x128_4x8_prefetch.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul64x128_4x8_prefetch.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul64x128_8x8 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul64x128_8x8.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul64x128_8x8.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul64x128_8x8.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul64x128_8x8_prefetch {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul64x128_8x8_prefetch.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul64x128_8x8_prefetch.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul64x128_8x8_prefetch.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul64x64 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul64x64.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul64x64.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul64x64.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul64x64_8x8 {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul64x64_8x8.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul64x64_8x8.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul64x64_8x8.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul64x64_8x8_prefetch {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul64x64_8x8_prefetch.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul64x64_8x8_prefetch.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul64x64_8x8_prefetch.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    pub mod matmul64x64_prefetch {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Functions{Matmul,MatmulNoPadded}
        impl crate::EntryPoint for Functions{
            fn get_entry_point(&self) -> &'static str{
                match self{
                    Functions::Matmul => "ga",Functions::MatmulNoPadded => "gb"
                }
            } 
        }
        pub fn load_shader(typ : crate::DType) -> &'static str {
            match typ{
                crate::DType::F32 => include_str!("kernels/sgemm//generated/matmul64x64_prefetch.pwgsl_generated_f32.wgsl"),
                crate::DType::U32 => include_str!("kernels/sgemm//generated/matmul64x64_prefetch.pwgsl_generated_u32.wgsl"),
                crate::DType::U8 => include_str!("kernels/sgemm//generated/matmul64x64_prefetch.pwgsl_generated_u8.wgsl"),
            }
        }
    }
        
    }
}
