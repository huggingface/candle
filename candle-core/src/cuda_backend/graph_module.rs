use std::cell::RefCell;
use crate::{Result, Tensor, Shape, Module, CudaDevice};
use crate::cuda_backend::graph::CudaGraphHandle;
use std::collections::HashMap;

pub trait CudaGraphModule: Module {
    /// Replay the captured graph for the given input
    fn replay(&self, xs: &Tensor) -> Result<Tensor>;

    /// Check if a graph is captured for the given shape
    fn has_graph_for_shape(&self, shape: &Shape) -> bool;

    /// Clear captured graphs (useful for memory management)
    fn clear_graphs(&mut self);
}

pub struct CudaGraphWrapper<M: Module> {
    module: M,
    captured_graphs: RefCell<HashMap<Shape, CudaGraphHandle>>,
    warmup_iterations: usize,
}

impl<M: Module> CudaGraphWrapper<M> {
    pub fn new(module: M) -> Self {
        Self {
            module,
            captured_graphs: RefCell::new(HashMap::new()),
            warmup_iterations: 3, // Default warmup iterations
        }
    }

    /// Automatically capture graph for given input if not already captured
    pub fn capture_if_needed(&mut self, input: &Tensor) -> Result<()> {
        let shape = input.shape().clone();

        // Check if already captured
        {
            let graphs = self.captured_graphs.borrow_mut();
            if graphs.contains_key(&shape) {
                return Ok(());
            }
        }

        // Perform warmup iterations
        for _ in 0..self.warmup_iterations {
            let _ = self.module.forward(input)?;
        }

        // Capture the graph
        let device = CudaDevice::new_with_stream(0)?;
        let graph = CudaGraphHandle::capture(&device, &shape, || {
            self.module.forward(input)?;
            Ok(())
        })?;
        self.captured_graphs.borrow_mut().insert(shape, graph);

        Ok(())
    }
}

impl<M: Module> Module for CudaGraphWrapper<M> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let shape = xs.shape();

        // Try to use captured graph first
        if let Some(graph) = self.captured_graphs.borrow_mut().get_mut(&shape) {
            return self.replay_with_graph(xs, graph);
        }

        // Fallback
        self.module.forward(xs)
    }
}

impl<M: Module> CudaGraphWrapper<M> {
    fn replay_with_graph(&self, xs: &Tensor, graph: &mut CudaGraphHandle) -> Result<Tensor> {

        graph.launch()?;

        // For now, fallback to regular execution
        // In a full implementation, you'd need to track input/output tensor locations
        // and update them appropriately
        self.module.forward(xs)
    }
}

impl<M: Module> CudaGraphModule for CudaGraphWrapper<M> {
    fn replay(&self, xs: &Tensor) -> Result<Tensor> {
        let shape = xs.shape();
        let mut graphs = self.captured_graphs.borrow_mut();

        if let Some(graph) = graphs.get_mut(shape) {
            self.replay_with_graph(xs, graph)
        } else {
            Err(crate::Error::Msg("No captured graph for this shape".into()).bt())
        }
    }

    fn has_graph_for_shape(&self, shape: &Shape) -> bool {
        let graphs = self.captured_graphs.borrow();
        graphs.contains_key(shape)
    }

    fn clear_graphs(&mut self) {
        let mut graphs = self.captured_graphs.borrow_mut();
        graphs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, DType, Tensor};


    /// Simple Transformer Block implementation for testing
    pub struct MockTransformerBlock {
        device: Device,
        hidden_size: usize,
    }

    impl MockTransformerBlock {
        pub fn new(device: Device, hidden_size: usize) -> Result<Self> {
            Ok(Self {
                device,
                hidden_size,
            })
        }

        fn layer_norm(&self, x: &Tensor) -> Result<Tensor> {
            // Simplified layer normalization: just normalize along last dimension
            let mean = x.mean_keepdim(x.dims().len() - 1)?;
            let variance = x.var_keepdim(x.dims().len() - 1)?;
            let normalized = x.broadcast_sub(&mean)?.broadcast_div(&(variance + 1e-5)?)?;
            Ok(normalized)
        }

        fn multi_head_attention(&self, x: &Tensor) -> Result<Tensor> {
            // Simplified self-attention: just a linear transformation
            // In practice, this would involve Q, K, V projections and attention computation
            let batch_size = x.dim(0)?;
            let seq_len = x.dim(1)?;
            
            // Create a simple "attention" weight matrix
            let attention_weights = Tensor::ones((self.hidden_size, self.hidden_size), DType::F32, &self.device)?;
            
            // Apply attention (simplified as matrix multiplication)
            let reshaped = x.reshape((batch_size * seq_len, self.hidden_size))?;
            let attended = reshaped.matmul(&attention_weights)?;
            let output = attended.reshape((batch_size, seq_len, self.hidden_size))?;
            
            Ok(output)
        }

        fn mlp(&self, x: &Tensor) -> Result<Tensor> {
            // Simplified MLP: Linear -> ReLU -> Linear
            let batch_size = x.dim(0)?;
            let seq_len = x.dim(1)?;
            let intermediate_size = self.hidden_size * 4;
            
            // First linear layer (expand)
            let w1 = Tensor::randn(0f32, 0.1, (self.hidden_size, intermediate_size), &self.device)?;
            let reshaped = x.reshape((batch_size * seq_len, self.hidden_size))?;
            let hidden = reshaped.matmul(&w1)?.relu()?;
            
            // Second linear layer (project back)
            let w2 = Tensor::randn(0f32, 0.1, (intermediate_size, self.hidden_size), &self.device)?;
            let output = hidden.matmul(&w2)?;
            let final_output = output.reshape((batch_size, seq_len, self.hidden_size))?;
            
            Ok(final_output)
        }
    }

    impl Module for MockTransformerBlock {
        fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            // Standard Transformer block: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
            
            // Pre-attention layer norm
            let normed1 = self.layer_norm(xs)?;
            
            // Multi-head attention with residual connection
            let attended = self.multi_head_attention(&normed1)?;
            let residual1 = xs.add(&attended)?;
            
            // Pre-MLP layer norm
            let normed2 = self.layer_norm(&residual1)?;
            
            // MLP with residual connection
            let mlp_output = self.mlp(&normed2)?;
            let final_output = residual1.add(&mlp_output)?;
            
            Ok(final_output)
        }
    }

    #[test]
    fn test_multiple_shapes_capture() -> Result<()> {
        let device = Device::Cuda(CudaDevice::new_with_stream(0)?);
        let hidden_size = 128;

        let model = MockTransformerBlock::new(device.clone(), hidden_size)?;
        let mut model_with_graph = CudaGraphWrapper::new(model);
        
        // Test with different input shapes
        let shapes = vec![
            (2, 32, hidden_size),   // Small batch, short sequence
            (4, 64, hidden_size),   // Medium batch, medium sequence
            (1, 128, hidden_size),  // Single batch, long sequence
        ];
        
        for (batch_size, seq_len, hidden_size) in shapes {
            let input = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;
            let shape = input.shape().clone();

            // Capture graph for this shape
            model_with_graph.capture_if_needed(&input)?;
            
            // Verify capture
            assert!(model_with_graph.has_graph_for_shape(&shape));
            
            // Test replay
            let _output = model_with_graph.replay(&input)?;
            
        }
        
        // Test clearing graphs
        model_with_graph.clear_graphs();
        
        // Verify all graphs are cleared
        for (batch_size, seq_len, hidden_size) in [(2, 32, 128), (4, 64, 128), (1, 128, 128)] {
            let shape = Shape::from_dims(&[batch_size, seq_len, hidden_size]);
            assert!(!model_with_graph.has_graph_for_shape(&shape));
        }

        Ok(())
    }
}