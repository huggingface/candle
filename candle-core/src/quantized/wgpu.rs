use super::GgmlDType;
use crate::{backend::{BackendDevice, BackendStorage}, DType, Result, Shape, WgpuDevice, WgpuStorage};
use crate::wgpu_backend::cache::BufferReferenceId;

pub struct QWgpuStorage {
    dtype: GgmlDType,
    storage: WgpuStorage,
}

fn read_to_vec<T: Clone>(buffer: &crate::CpuStorage, n: usize) -> Vec<T> {
    let ptr = buffer.as_slice::<u32>().unwrap().as_ptr() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}


impl QWgpuStorage {
    pub fn new(dtype: GgmlDType, storage : WgpuStorage) -> Self {
        Self { dtype, storage }
    }
    pub fn buffer(&self) -> &BufferReferenceId {
        &*self.storage.buffer()
    }
    pub fn zeros(device: &WgpuDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size = elem_count * dtype.type_size() / dtype.block_size();
        return  Ok(QWgpuStorage::new(dtype, device.zeros_impl(&(size as usize / 4,).into(), DType::U32)?));
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.storage.device()
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.storage.get_length() as usize
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<WgpuStorage> {
        use crate::quantized::k_quants::GgmlType;
        let dev = self.device();
        let dst = dev.alloc_uninit_size(DType::F32, elem_count);
      
        let mut queue = dev.get_queue();
        queue.add(elem_count);
        let pipeline = match self.dtype(){
            GgmlDType::Q4_0 => candle_wgpu_kernels::Pipelines::Q40(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q4_0::Functions::DequantizeToF32),
            GgmlDType::Q4_1 => candle_wgpu_kernels::Pipelines::Q41(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q4_1::Functions::DequantizeToF32),
            GgmlDType::Q5_0 => candle_wgpu_kernels::Pipelines::Q50(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q5_0::Functions::DequantizeToF32),
            GgmlDType::Q5_1 => candle_wgpu_kernels::Pipelines::Q51(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q5_1::Functions::DequantizeToF32),
            GgmlDType::Q8_0 => candle_wgpu_kernels::Pipelines::Q80(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q8_0::Functions::DequantizeToF32),
            GgmlDType::Q8_1 => candle_wgpu_kernels::Pipelines::Q81(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q8_1::Functions::DequantizeToF32),
            _ => {

                let buffer =  self.storage.to_cpu_storage()?;
       
                let mut out = vec![0.0; elem_count];
                let block_len = elem_count / self.dtype.block_size();
                match self.dtype {
                    GgmlDType::F32 => {
                        let vec: Vec<f32> = read_to_vec(&buffer, block_len);
                        f32::to_float(&vec, &mut out)?;
                    }
                    GgmlDType::F16 => {
                        let vec: Vec<half::f16> = read_to_vec(&buffer, block_len);
                        half::f16::to_float(&vec, &mut out)?;
                    }
                    GgmlDType::Q4_0 => {
                        let vec: Vec<crate::quantized::BlockQ4_0> = read_to_vec(&buffer, block_len);
                        crate::quantized::BlockQ4_0::to_float(&vec, &mut out)?;
                    }
                    GgmlDType::Q4_1 => {
                        let vec: Vec<crate::quantized::BlockQ4_1> = read_to_vec(&buffer, block_len);
                        crate::quantized::BlockQ4_1::to_float(&vec, &mut out)?;
                    }
                    GgmlDType::Q5_0 => {
                        let vec: Vec<crate::quantized::BlockQ5_0> = read_to_vec(&buffer, block_len);
                        crate::quantized::BlockQ5_0::to_float(&vec, &mut out)?;
                    }
                    GgmlDType::Q5_1 => {
                        let vec: Vec<crate::quantized::BlockQ5_1> = read_to_vec(&buffer, block_len);
                        crate::quantized::BlockQ5_1::to_float(&vec, &mut out)?;
                    }
                    GgmlDType::Q8_0 => {
                        let vec: Vec<crate::quantized::BlockQ8_0> = read_to_vec(&buffer, block_len);
                        crate::quantized::BlockQ8_0::to_float(&vec, &mut out)?;
                    }
                    GgmlDType::Q8_1 => {
                        let vec: Vec<crate::quantized::BlockQ8_1> = read_to_vec(&buffer, block_len);
                        crate::quantized::BlockQ8_1::to_float(&vec, &mut out)?;
                    }
                    GgmlDType::Q2K => {
                        let vec: Vec<crate::quantized::BlockQ2K> = read_to_vec(&buffer, block_len);
                        crate::quantized::BlockQ2K::to_float(&vec, &mut out)?;
                    }
                    GgmlDType::Q3K => {
                        let vec: Vec<crate::quantized::BlockQ3K> = read_to_vec(&buffer, block_len);
                        crate::quantized::BlockQ3K::to_float(&vec, &mut out)?;
                    }
                    GgmlDType::Q4K => {
                        let vec: Vec<crate::quantized::BlockQ4K> = read_to_vec(&buffer, block_len);
                        crate::quantized::BlockQ4K::to_float(&vec, &mut out)?;
                    }
                    GgmlDType::Q5K => {
                        let vec: Vec<crate::quantized::BlockQ5K> = read_to_vec(&buffer, block_len);
                        crate::quantized::BlockQ5K::to_float(&vec, &mut out)?;
                    }
                    GgmlDType::Q6K => {
                        let vec: Vec<crate::quantized::BlockQ6K> = read_to_vec(&buffer, block_len);
                        crate::quantized::BlockQ6K::to_float(&vec, &mut out)?;
                    }
                    GgmlDType::Q8K => {
                        let vec: Vec<crate::quantized::BlockQ8K> = read_to_vec(&buffer, block_len);
                        crate::quantized::BlockQ8K::to_float(&vec, &mut out)?;
                    }
                }
                
                return self.device().alloc_from_slice(DType::F32,&out);

            }
        };
        let pipeline = queue.get_pipeline(pipeline);
        let bind_group = dev.create_bind_group_input1(*dst.buffer(), *self.buffer(), DType::F32.into());

        queue.enqueue_64(pipeline, bind_group, elem_count as u32,elem_count);

        return Ok(dst);
    }

    pub fn quantize(&mut self, src: &WgpuStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu_storage()?;
        let elem_count = src.as_slice::<f32>()?.len();
        let src = crate::Storage::Cpu(src);
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let buffer = self.device().alloc_from_slice(DType::U32, &qcpu_storage.data()?)?;
        self.storage = buffer;
        Ok(())
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &WgpuStorage,
        layout: &crate::Layout,
    ) -> Result<(WgpuStorage, Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let src_shape = src_shape.dims().to_vec();

        
        let (b,m) = match src_shape.len() {
            3 => (src_shape[0], src_shape[1]),
            2 => (1, src_shape[0]),
            n => crate::bail!("Invalid rank {n} for quantized matmul wgpu"),
        };
        let mut dst_shape = src_shape;
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);

        let dst_shape = Shape::from(dst_shape);
        let dev = storage.device();
        let dst = dev.alloc_uninit_size(DType::F32, dst_shape.elem_count());
        
        let mut input1_stride = layout.stride().iter().rev();

        let input1_stride_k = *input1_stride.next().unwrap_or(&1);
        let input1_stride_m = *input1_stride.next().unwrap_or(&1);
        let input1_stride_b = *input1_stride.next().unwrap_or(&1);

        let mut queue = dev.get_queue();
        queue.add(b);
        queue.add(m);
        queue.add(k);
        queue.add(n);

        queue.add(input1_stride_b); //input1_stride_b
        queue.add(layout.start_offset()); //input1_offset
        queue.add(0); //input2_ofset
        queue.add(input1_stride_k);
        queue.add(input1_stride_m);
        let pipeline = match self.dtype(){
            GgmlDType::Q4_0 => candle_wgpu_kernels::Pipelines::Q40(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q4_0::Functions::MatmulNaive),
            GgmlDType::Q4_1 => candle_wgpu_kernels::Pipelines::Q41(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q4_1::Functions::MatmulNaive),
            GgmlDType::Q5_0 => candle_wgpu_kernels::Pipelines::Q50(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q5_0::Functions::MatmulNaive),
            GgmlDType::Q5_1 => candle_wgpu_kernels::Pipelines::Q51(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q5_1::Functions::MatmulNaive),
            GgmlDType::Q8_0 => candle_wgpu_kernels::Pipelines::Q80(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q8_0::Functions::MatmulNaive),
            GgmlDType::Q8_1 => candle_wgpu_kernels::Pipelines::Q81(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q8_1::Functions::MatmulNaive),
            _ => todo!()
        };
        let const_vec = vec![
            (input1_stride_k == 1) as usize,
            (input1_stride_m == 1) as usize,
            (b != 1) as usize,
        ];
        
        let pipeline = queue.get_pipeline_const(pipeline, const_vec);
        let bind_group = dev.create_bind_group_input2(*dst.buffer(), *storage.buffer(),*self.buffer(), DType::F32.into());

        queue.enqueue_workgroups(
            pipeline,
            bind_group,
            (n as u32 + 15) / 16,
            (m as u32+ 15) / 16,
            b as u32,
            k as usize * m as usize * n as usize,
        );
    
        Ok((dst, dst_shape))
    }
}

// pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
//     device: &WgpuDevice,
//     data: &[T],
// ) -> Result<QStorage> {
//     let storage = device.alloc_from_slice(DType::F32, data)?;
//     let device = device.clone();
//     Ok(QStorage::Wgpu(QWgpuStorage {
//         dtype: T::DTYPE,
//         storage,
//     }))
// }
