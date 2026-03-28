#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::test_utils::{to_vec0_round, to_vec2_round};

use anyhow::Result;
use candle::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Linear, Module, Optimizer, ParamsAdamW, SGD};

#[test]
fn sgd_optim() -> Result<()> {
    let x = Var::new(0f32, &Device::Cpu)?;
    let mut sgd = SGD::new(vec![x.clone()], 0.1)?;
    let xt = x.as_tensor();
    for _step in 0..100 {
        let loss = ((xt - 4.2)? * (xt - 4.2)?)?;
        sgd.backward_step(&loss)?
    }
    assert_eq!(x.to_scalar::<f32>()?, 4.199999);
    Ok(())
}

/* The results of this test have been checked against the following PyTorch code.
    import torch
    from torch import optim

    w_gen = torch.tensor([[3., 1.]])
    b_gen = torch.tensor([-2.])

    sample_xs = torch.tensor([[2., 1.], [7., 4.], [-4., 12.], [5., 8.]])
    sample_ys = sample_xs.matmul(w_gen.t()) + b_gen

    m = torch.nn.Linear(2, 1)
    with torch.no_grad():
        m.weight.zero_()
        m.bias.zero_()
    optimizer = optim.SGD(m.parameters(), lr=0.004, momentum=0.)
    for _step in range(1000):
        optimizer.zero_grad()
        ys = m(sample_xs)
        loss = ((ys - sample_ys)**2).sum()
        loss.backward()
        optimizer.step()
    print(m.weight)
    print(m.bias)
*/
#[test]
fn sgd_linear_regression() -> Result<()> {
    // Generate some linear data, y = 3.x1 + x2 - 2.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    // Now use backprop to run a linear regression between samples and get the coefficients back.
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let mut sgd = SGD::new(vec![w.clone(), b.clone()], 0.004)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..1000 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        sgd.backward_step(&loss)?;
    }
    assert_eq!(w.to_vec2::<f32>()?, &[[2.9983196, 0.99790204]]);
    assert_eq!(b.to_scalar::<f32>()?, -1.9796902);
    Ok(())
}

/* The following test returns the same values as the PyTorch code below.
import torch
from torch import optim

w_gen = torch.tensor([[3., 1.]])
b_gen = torch.tensor([-2.])

sample_xs = torch.tensor([[2., 1.], [7., 4.], [-4., 12.], [5., 8.]])
sample_ys = sample_xs.matmul(w_gen.t()) + b_gen

m = torch.nn.Linear(2, 1)
with torch.no_grad():
    m.weight.zero_()
    m.bias.zero_()
optimizer = optim.AdamW(m.parameters(), lr=0.1)
for _step in range(100):
    optimizer.zero_grad()
    ys = m(sample_xs)
    loss = ((ys - sample_ys)**2).sum()
    loss.backward()
    optimizer.step()
print(m.weight)
print(m.bias)
*/
#[test]
fn adamw_linear_regression() -> Result<()> {
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    // Now use backprop to run a linear regression between samples and get the coefficients back.
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let params = ParamsAdamW {
        lr: 0.1,
        ..Default::default()
    };
    let mut opt = AdamW::new(vec![w.clone(), b.clone()], params)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..100 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
    }
    assert_eq!(to_vec2_round(w.as_tensor(), 4)?, &[[2.7257, 0.7097]]);
    assert_eq!(to_vec0_round(b.as_tensor(), 4)?, 0.7873);
    Ok(())
}

#[test]
fn adamw_linear_regression_varmap() -> Result<()> {
    use candle_nn::Init::Const;

    // Similar as the previous test but using a VarMap.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    let mut var_map = candle_nn::VarMap::new();

    let w = var_map.get((1, 2), "w", Const(0.), DType::F32, &Device::Cpu)?;
    let b = var_map.get((), "b", Const(0.), DType::F32, &Device::Cpu)?;
    let params = ParamsAdamW {
        lr: 0.1,
        ..Default::default()
    };
    let mut opt = AdamW::new(var_map.all_vars(), params)?;
    let lin = Linear::new(w, Some(b));
    for _step in 0..100 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
    }
    assert_eq!(to_vec2_round(lin.weight(), 4)?, &[[2.7257, 0.7097]]);
    assert_eq!(to_vec0_round(lin.bias().unwrap(), 4)?, 0.7873);

    var_map.set([("w", Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?)].into_iter())?;
    var_map.set([("b", Tensor::ones((), DType::F32, &Device::Cpu)?)].into_iter())?;

    assert_eq!(to_vec2_round(lin.weight(), 4)?, &[[0., 0.]]);
    assert_eq!(to_vec0_round(lin.bias().unwrap(), 4)?, 1.);
    Ok(())
}

// Tests for `ReduceLROnPlateau`
mod reduce_lr_on_plateau_tests {
    use candle::backprop;
    use candle_nn::optim::{LRPlateauMode, LRPlateauThresholdMode, ReduceLROnPlateauBuilder};

    use super::*;

    /// Assert that two floating point numbers are approximately equal
    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr) => {
            assert!(
                ($a - $b).abs() < 1e-8,
                "{} is not approximately equal to {}",
                $a,
                $b
            );
        };
    }

    // A mock optimizer for testing the scheduler
    pub struct MockOptimizer {
        learning_rate: f64,
    }

    impl Default for MockOptimizer {
        fn default() -> Self {
            Self::new(vec![], MockOptimizerConfig::default()).unwrap()
        }
    }

    pub struct MockOptimizerConfig {
        learning_rate: f64,
    }

    impl Default for MockOptimizerConfig {
        fn default() -> Self {
            Self { learning_rate: 0.1 }
        }
    }

    impl Optimizer for MockOptimizer {
        type Config = MockOptimizerConfig;
        fn new(_vars: Vec<Var>, config: Self::Config) -> candle::Result<Self> {
            Ok(Self {
                learning_rate: config.learning_rate,
            })
        }

        fn learning_rate(&self) -> f64 {
            self.learning_rate
        }

        fn set_learning_rate(&mut self, lr: f64) {
            self.learning_rate = lr;
        }

        fn step(&mut self, _grads: &backprop::GradStore) -> candle::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_reduce_lr_on_plateau_min_mode() -> Result<()> {
        // Initialize with a mock optimizer and min mode (default)
        let optimizer = MockOptimizer::default();
        let mut scheduler = ReduceLROnPlateauBuilder::new(optimizer)
            .mode(LRPlateauMode::Min)
            .patience(2)
            .factor(0.5)
            .build();

        // First metric sets the best value
        scheduler.step(10.0)?;
        assert_eq!(scheduler.opt().learning_rate(), 0.1);
        assert_eq!(scheduler.best(), Some(10.0));
        assert_eq!(scheduler.num_bad_epochs(), 0);

        // Improvement
        scheduler.step(9.0)?;
        assert_eq!(scheduler.opt().learning_rate(), 0.1);
        assert_eq!(scheduler.best(), Some(9.0));
        assert_eq!(scheduler.num_bad_epochs(), 0);

        // No improvement (first bad epoch)
        scheduler.step(9.0)?;
        assert_eq!(scheduler.opt().learning_rate(), 0.1);
        assert_eq!(scheduler.num_bad_epochs(), 1);

        // No improvement (second bad epoch)
        scheduler.step(9.1)?;
        assert_eq!(scheduler.opt().learning_rate(), 0.1);
        assert_eq!(scheduler.num_bad_epochs(), 2);

        // No improvement (third bad epoch - should reduce LR)
        scheduler.step(9.2)?;
        assert_approx_eq!(scheduler.opt().learning_rate(), 0.05); // 0.1 * 0.5
        assert_eq!(scheduler.num_bad_epochs(), 0); // Reset after LR reduction
        assert_eq!(scheduler.cooldown_counter(), 0); // No cooldown set

        Ok(())
    }

    #[test]
    fn test_reduce_lr_on_plateau_max_mode() -> Result<()> {
        // Initialize with max mode
        let optimizer = MockOptimizer::default();
        let starting_lr = optimizer.learning_rate();
        let decay_factor = 0.5;
        let mut scheduler = ReduceLROnPlateauBuilder::new(optimizer)
            .mode(LRPlateauMode::Max)
            .patience(2)
            .factor(decay_factor)
            .build();

        // First metric sets the best value
        scheduler.step(10.0)?;
        assert_eq!(scheduler.best(), Some(10.0));

        // No improvement
        scheduler.step(9.0)?;
        assert_eq!(scheduler.num_bad_epochs(), 1);

        // Improvement
        scheduler.step(11.0)?;
        assert_eq!(scheduler.best(), Some(11.0));
        assert_eq!(scheduler.num_bad_epochs(), 0);

        // No improvement (first bad epoch)
        scheduler.step(10.9)?;
        assert_eq!(scheduler.num_bad_epochs(), 1);

        // No improvement (second bad epoch)
        scheduler.step(10.8)?;
        assert_eq!(scheduler.num_bad_epochs(), 2);

        // No improvement (third bad epoch - should reduce LR)
        scheduler.step(10.7)?;
        assert_approx_eq!(scheduler.opt().learning_rate(), starting_lr * decay_factor);

        Ok(())
    }

    #[test]
    fn test_reduce_lr_on_plateau_threshold_rel() -> Result<()> {
        let optimizer = MockOptimizer::default();
        let mut scheduler = ReduceLROnPlateauBuilder::new(optimizer)
            .mode(LRPlateauMode::Min)
            .threshold(0.1) // 10% threshold
            .threshold_mode(LRPlateauThresholdMode::Rel)
            .patience(1)
            .build();

        // First step establishes baseline
        scheduler.step(10.0)?;
        assert_eq!(scheduler.best(), Some(10.0));

        // Improvement but within threshold (10.0 * (1 - 0.1) = 9.0)
        // 9.5 > 9.0, so this isn't enough improvement to count
        scheduler.step(9.5)?;
        assert_eq!(scheduler.num_bad_epochs(), 1);
        assert_eq!(scheduler.best(), Some(10.0)); // Best value doesn't change

        // Real improvement beyond threshold
        scheduler.step(8.9)?;
        assert_eq!(scheduler.best(), Some(8.9));
        assert_eq!(scheduler.num_bad_epochs(), 0);

        Ok(())
    }

    #[test]
    fn test_reduce_lr_on_plateau_threshold_abs() -> Result<()> {
        let optimizer = MockOptimizer::default();
        let mut scheduler = ReduceLROnPlateauBuilder::new(optimizer)
            .mode(LRPlateauMode::Min)
            .threshold(1.0) // Absolute threshold of 1.0
            .threshold_mode(LRPlateauThresholdMode::Abs)
            .patience(1)
            .build();

        // First step establishes baseline
        scheduler.step(10.0)?;
        assert_eq!(scheduler.best(), Some(10.0));

        // Improvement but within threshold (10.0 - 1.0 = 9.0)
        // 9.5 > 9.0, so this isn't enough improvement to count
        scheduler.step(9.5)?;
        assert_eq!(scheduler.num_bad_epochs(), 1);
        assert_eq!(scheduler.best(), Some(10.0)); // Best value doesn't change

        // Real improvement beyond threshold
        scheduler.step(8.9)?;
        assert_eq!(scheduler.best(), Some(8.9));
        assert_eq!(scheduler.num_bad_epochs(), 0);

        Ok(())
    }

    #[test]
    fn test_reduce_lr_on_plateau_cooldown() -> Result<()> {
        let optimizer = MockOptimizer::default();
        let starting_lr = optimizer.learning_rate();
        let decay_factor = 0.5;
        let mut scheduler = ReduceLROnPlateauBuilder::new(optimizer)
            .mode(LRPlateauMode::Min)
            .patience(1)
            .cooldown(2) // 2 epochs cooldown
            .factor(decay_factor)
            .build();

        // First metric sets the best value
        scheduler.step(10.0)?;
        assert_eq!(scheduler.best(), Some(10.0));

        // No improvement (first bad epoch)
        scheduler.step(10.1)?;
        assert_eq!(scheduler.num_bad_epochs(), 1);

        // No improvement (second bad epoch - should reduce LR)
        scheduler.step(10.2)?;
        assert_approx_eq!(scheduler.opt().learning_rate(), starting_lr * decay_factor);
        assert_eq!(scheduler.cooldown_counter(), 2); // Cooldown starts

        // Even though this is worse, we're in cooldown so no reduction
        scheduler.step(10.3)?;
        assert_approx_eq!(scheduler.opt().learning_rate(), starting_lr * decay_factor);
        assert_eq!(scheduler.cooldown_counter(), 1); // Cooldown decreases
        assert_eq!(scheduler.num_bad_epochs(), 0); // Bad epochs reset during cooldown

        // Still in cooldown
        scheduler.step(10.4)?;
        assert_approx_eq!(scheduler.opt().learning_rate(), starting_lr * decay_factor);
        assert_eq!(scheduler.cooldown_counter(), 0); // Cooldown over after this step

        // Cooldown over, this will count as a bad epoch
        scheduler.step(10.5)?;
        assert_approx_eq!(scheduler.opt().learning_rate(), starting_lr * decay_factor);
        assert_eq!(scheduler.num_bad_epochs(), 1); // Bad epochs increment again

        // Another bad epoch, should reduce LR again
        scheduler.step(10.6)?;
        assert_approx_eq!(
            scheduler.opt().learning_rate(),
            starting_lr * decay_factor * decay_factor
        );
        assert_eq!(scheduler.cooldown_counter(), 2); // Cooldown starts again

        Ok(())
    }

    #[test]
    fn test_reduce_lr_on_plateau_min_lr() -> Result<()> {
        let optimizer = MockOptimizer::default();
        let starting_lr = optimizer.learning_rate();
        let mut scheduler = ReduceLROnPlateauBuilder::new(optimizer)
            .patience(0) // No patience, so we should reduce LR immediately
            .min_lr(0.01) // Set minimum learning rate
            .build();

        // First step sets best
        scheduler.step(10.0)?;
        assert_approx_eq!(scheduler.opt().learning_rate(), starting_lr);

        // Reduce LR from 0.1 to 0.01
        scheduler.step(10.1)?;
        assert_approx_eq!(scheduler.opt().learning_rate(), 0.01);

        // No improvement, reached min_lr
        scheduler.step(10.2)?;
        assert_approx_eq!(scheduler.opt().learning_rate(), 0.01);

        Ok(())
    }

    #[test]
    fn test_reduce_lr_on_plateau_epsilon() -> Result<()> {
        let optimizer = MockOptimizer::default();
        let starting_lr = optimizer.learning_rate();
        let mut scheduler = ReduceLROnPlateauBuilder::new(optimizer)
            .mode(LRPlateauMode::Min)
            .patience(1)
            .factor(0.999) // Very small reduction factor
            .eps(1e-5) // Minimum meaningful change
            .build();

        // First step sets best
        let mut data = 10.0;
        scheduler.step(10.0)?;

        // Add an eps that is very small
        data += 1e-9;
        scheduler.step(data)?;
        // No change
        assert_approx_eq!(scheduler.opt().learning_rate(), starting_lr);

        // Try with a bigger factor that exceeds eps
        data += 1e-2;
        scheduler.step(data)?;
        assert_approx_eq!(scheduler.opt().learning_rate(), 0.0999); // Change applied

        Ok(())
    }

    #[test]
    fn test_reduce_lr_on_plateau_builder() {
        let optimizer = MockOptimizer::default();

        // Test that all builder methods work
        let scheduler = ReduceLROnPlateauBuilder::new(optimizer)
            .mode(LRPlateauMode::Max)
            .factor(0.2)
            .patience(5)
            .threshold(0.05)
            .threshold_mode(LRPlateauThresholdMode::Abs)
            .cooldown(3)
            .min_lr(0.001)
            .eps(0.00001)
            .build();

        // Verify all properties were set correctly
        assert!(matches!(scheduler.mode(), LRPlateauMode::Max));
        assert_eq!(scheduler.factor(), 0.2);
        assert_eq!(scheduler.patience(), 5);
        assert_eq!(scheduler.threshold(), 0.05);
        assert!(matches!(
            scheduler.threshold_mode(),
            LRPlateauThresholdMode::Abs
        ));
        assert_eq!(scheduler.cooldown(), 3);
        assert_eq!(scheduler.min_lr(), 0.001);
        assert_eq!(scheduler.eps(), 0.00001);
    }

    #[test]
    fn test_with_real_optimizer() -> Result<()> {
        // Create a tensor to optimize
        let device = Device::Cpu;
        let tensor = Tensor::new(&[0.0f32, 0.0], &device)?;
        let tensor = Var::from_tensor(&tensor)?;

        // Set up a real AdamW optimizer
        let optimizer = AdamW::new(vec![tensor], ParamsAdamW::default())?;
        let initial_lr = optimizer.learning_rate();

        let mut scheduler = ReduceLROnPlateauBuilder::new(optimizer)
            .patience(1)
            .factor(0.1)
            .build();

        // First step to establish baseline
        scheduler.step(1.0)?;

        // Two consecutive non-improvements should trigger LR reduction
        scheduler.step(1.1)?;
        scheduler.step(1.2)?;

        assert_approx_eq!(scheduler.opt().learning_rate(), initial_lr * 0.1);

        Ok(())
    }

    /// Test the struct prints as expected
    #[test]
    fn test_display() {
        let optimizer = MockOptimizer::default();
        let scheduler = ReduceLROnPlateauBuilder::new(optimizer)
            .factor(0.5)
            .patience(5)
            .threshold(1e-3)
            .threshold_mode(LRPlateauThresholdMode::Abs)
            .cooldown(2)
            .min_lr(1e-6)
            .eps(1e-9)
            .build();

        let display_output = format!("{}", scheduler);

        // Verify that all fields are included in the display output
        assert!(display_output.contains("mode: Min"));
        assert!(display_output.contains("factor: 0.5"));
        assert!(display_output.contains("patience: 5"));
        assert!(display_output.contains("threshold: 0.001"));
        assert!(display_output.contains("threshold_mode: Abs"));
        assert!(display_output.contains("cooldown: 2"));
        assert!(display_output.contains("min_lr: 0.000001"));
        assert!(display_output.contains("eps: 0.000000001"));
        assert!(display_output.contains("best: None"));
        assert!(display_output.contains("num_bad_epochs: 0"));
        assert!(display_output.contains("current_lr: 0.1"));
    }
}
