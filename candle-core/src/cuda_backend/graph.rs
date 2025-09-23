use std::fmt;
use cudarc::driver::{CudaGraph, sys};
use crate::{CudaDevice, Result, Shape, Error};
use crate::cuda_backend::CudaError;

pub struct CudaGraphHandle {
    graph: Option<CudaGraph>,
    device: CudaDevice,
    captured_shape: Shape,
}

impl fmt::Debug for CudaGraphHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::result::Result<(), fmt::Error> {
        #[derive(Debug)]
        #[allow(dead_code)]
        struct CudaGraphHandle<'a> {
            device: &'a CudaDevice,
            captured_shape: &'a Shape,
        }

        let Self {
            graph: _,
            device,
            captured_shape,
        } = self;

        fmt::Debug::fmt(&CudaGraphHandle { device, captured_shape }, f)
    }
}

impl CudaGraphHandle {
    pub fn capture<F>(
        device: &CudaDevice,
        shape: &Shape,
        capture_fn: F
    ) -> Result<Self>
    where
        F: FnOnce() -> Result<()>
    {
        let stream = device.cuda_stream();

        // Begin capture
        let flags = sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED;

        stream.begin_capture(flags).map_err(|e| CudaError::from(e))?;

        // Execute the capture function
        capture_fn()?;

        // End capture and create graph
        let flags = sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        let graph = stream.end_capture(flags).map_err(|e| CudaError::from(e))?;

        Ok(CudaGraphHandle {
            graph,
            device: device.clone(),
            captured_shape: shape.clone(),
        })
    }

    pub fn launch(&mut self) -> Result<()> {
        match &mut self.graph {
            Some(g) => {
                g.launch().map_err(|e| CudaError::from(e))
            }
            None => Err(CudaError::InternalError("failed to create graph during capture"))
        }.map_err(|e| Error::from(e))
    }

    pub fn shape(&self) -> &Shape {
        &self.captured_shape
    }

    pub fn device(&self) -> &CudaDevice { &self.device }
}

impl Drop for CudaGraphHandle {
    fn drop(&mut self) {
        if !self.graph.is_none() {
            std::mem::drop(self.graph.take());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, DType, Tensor};
    use crate::backend::BackendDevice;

    // Helper function to create a CUDA device for testing
    fn create_cuda_device() -> Result<CudaDevice> {
        CudaDevice::new_with_stream(0)
    }

    #[test]
    fn test_cuda_graph_capture_basic() -> Result<()> {

        let device = create_cuda_device()?;
        let shape = Shape::from_dims(&[2, 3]);
        
        // Create a simple capture function that performs a basic operation
        let graph = CudaGraphHandle::capture(&device, &shape, || {
            // In a real scenario, this would contain actual CUDA operations
            // For now, we just test that the capture mechanism works
            Ok(())
        })?;

        // Verify the shape was captured correctly
        assert_eq!(graph.shape(), &shape);
        Ok(())
    }

    #[test]
    fn test_cuda_graph_launch() -> Result<()> {
        let device = create_cuda_device()?;
        let shape = Shape::from_dims(&[4, 4]);
        
        // Capture a graph
        let mut graph = CudaGraphHandle::capture(&device, &shape, || {
            // Simulate some work during capture
            Ok(())
        })?;

        // Test that we can launch the graph without errors
        graph.launch()?;
        Ok(())
    }

    #[test]
    fn test_cuda_graph_multiple_launches() -> Result<()> {
        let device = create_cuda_device()?;
        let shape = Shape::from_dims(&[8, 8]);
        
        let mut graph = CudaGraphHandle::capture(&device, &shape, || {
            Ok(())
        })?;

        // Test multiple launches of the same graph
        for _ in 0..5 {
            graph.launch()?;
        }
        Ok(())
    }


    #[test]
    fn test_cuda_graph_error_handling() -> Result<()> {
        let device = create_cuda_device()?;
        let shape = Shape::from_dims(&[5, 5]);
        
        // Test that errors in capture function are properly propagated
        let result = CudaGraphHandle::capture(&device, &shape, || {
            Err(crate::Error::Msg("Test error during capture".into()))
        });

        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_cuda_graph_with_multiple_operations() -> Result<()> {
        let device = create_cuda_device()?;
        let cuda_device = Device::Cuda(device.clone());
        let shape = Shape::from_dims(&[16, 16]);
        
        // Create multiple tensors for complex operations
        let a = Tensor::ones((16, 16), DType::F32, &cuda_device)?;
        let b = Tensor::full(2.0f32, (16, 16), &cuda_device)?;

        // Capture a graph with multiple chained operations
        let mut graph = CudaGraphHandle::capture(&device, &shape, || {
            // Chain multiple operations together
            let step1 = a.add(&b)?;  // 1 + 2 = 3
            let step3 = step1.sum_keepdim(1)?;  // Sum along axis 1
            let _final_result = step3.relu()?;  // Apply ReLU
            Ok(())
        })?;

        // Verify graph creation and launch it
        assert_eq!(graph.shape(), &shape);
        
        // Test multiple launches to ensure stability
        for _ in 0..5 {
            graph.launch()?;
            device.synchronize()?;
        }

        Ok(())
    }
}