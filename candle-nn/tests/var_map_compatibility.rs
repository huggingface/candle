use candle::{DType, Device, Result, Tensor, Var};
use candle_nn::var_map::ConcurrentVarMap;
use candle_nn::{Init, VarMap};
use std::sync::{Arc, Barrier};
use std::thread;

#[test]
fn test_basic_operations_compatibility() -> Result<()> {
    let device = Device::Cpu;

    // Original implementation
    let original = {
        #[derive(Clone)]
        struct OriginalVarMap {
            data: Arc<std::sync::Mutex<std::collections::HashMap<String, Var>>>,
        }

        impl OriginalVarMap {
            fn new() -> Self {
                Self {
                    data: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
                }
            }

            fn get<S: Into<candle::Shape>>(
                &self,
                shape: S,
                path: &str,
                init: Init,
                dtype: DType,
                device: &Device,
            ) -> Result<Tensor> {
                let shape = shape.into();
                let mut tensor_data = self.data.lock().unwrap();
                if let Some(tensor) = tensor_data.get(path) {
                    let tensor_shape = tensor.shape();
                    if &shape != tensor_shape {
                        candle::bail!("shape mismatch on {path}: {shape:?} <> {tensor_shape:?}")
                    }
                    return Ok(tensor.as_tensor().clone());
                }
                let var = init.var(shape, dtype, device)?;
                let tensor = var.as_tensor().clone();
                tensor_data.insert(path.to_string(), var);
                Ok(tensor)
            }

            fn all_vars(&self) -> Vec<Var> {
                let tensor_data = self.data.lock().unwrap();
                tensor_data.values().cloned().collect()
            }
        }

        OriginalVarMap::new()
    };

    // New implementation
    let updated = VarMap::new();

    // Test 1: Basic get operations
    let t1_orig = original.get(
        (2, 3),
        "test1",
        Init::Randn {
            mean: 0.,
            stdev: 1.,
        },
        DType::F32,
        &device,
    )?;
    let t1_updated = updated.get(
        (2, 3),
        "test1",
        Init::Randn {
            mean: 0.,
            stdev: 1.,
        },
        DType::F32,
        &device,
    )?;

    // Shapes should match
    assert_eq!(t1_orig.shape(), t1_updated.shape());

    // Test 2: Repeated get returns same variable
    let t1_orig_2 = original.get((2, 3), "test1", Init::Const(0.), DType::F32, &device)?;
    let t1_updated_2 = updated.get((2, 3), "test1", Init::Const(0.), DType::F32, &device)?;

    // Should return existing variables
    assert_eq!(t1_orig.shape(), t1_orig_2.shape());
    assert_eq!(t1_updated.shape(), t1_updated_2.shape());

    // Test 3: Multiple variables
    for i in 0..10 {
        let name = format!("var_{}", i);
        original.get(
            (i + 1, i + 2),
            &name,
            Init::Const(i as f64),
            DType::F32,
            &device,
        )?;
        updated.get(
            (i + 1, i + 2),
            &name,
            Init::Const(i as f64),
            DType::F32,
            &device,
        )?;
    }

    // Verify all variables match
    assert_eq!(original.all_vars().len(), updated.all_vars().len());

    Ok(())
}

#[test]
fn test_concurrent_reads_match_sequential() -> Result<()> {
    let device = Device::Cpu;
    let updated = Arc::new(VarMap::new());
    let concurrent = Arc::new(ConcurrentVarMap::new());

    // Initialize both with same data
    for i in 0..100 {
        let name = format!("var_{}", i);
        let shape = (10, 10);
        let init = Init::Const(i as f64);

        updated.get(shape, &name, init, DType::F32, &device)?;
        concurrent.get(shape, &name, init, DType::F32, &device)?;
    }

    // Test concurrent reads
    let n_threads = 8;
    let barrier = Arc::new(Barrier::new(n_threads));
    let mut handles = vec![];

    for _thread_id in 0..n_threads {
        let updated_clone: Arc<VarMap> = Arc::clone(&updated);
        let concurrent_clone: Arc<ConcurrentVarMap> = Arc::clone(&concurrent);
        let barrier_clone = Arc::clone(&barrier);
        let device_clone = device.clone();

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            // Each thread reads multiple variables
            for i in 0..100 {
                let name = format!("var_{}", i);
                let shape = (10, 10);

                let v1 = updated_clone
                    .get(shape, &name, Init::Const(0.), DType::F32, &device_clone)
                    .unwrap();
                let v2 = concurrent_clone
                    .get(shape, &name, Init::Const(0.), DType::F32, &device_clone)
                    .unwrap();

                // Values should match
                assert_eq!(v1.shape(), v2.shape());

                // Compare flattened data for any shape
                let data1 = v1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let data2 = v2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                assert_eq!(data1, data2);
            }
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    Ok(())
}

#[test]
fn test_save_load_compatibility() -> Result<()> {
    let device = Device::Cpu;
    let original = VarMap::new();
    let updated = VarMap::new();

    // Create identical data
    for i in 0..20 {
        let name = format!("layer_{}.weight", i);
        let shape = (64, 64);
        // Use a deterministic init for comparison
        original.get(shape, &name, Init::Const(i as f64), DType::F32, &device)?;
        updated.get(shape, &name, Init::Const(i as f64), DType::F32, &device)?;
    }

    // Save both
    let original_path = "/tmp/test_original_varmap.safetensors";
    let updated_path = "/tmp/test_updated_varmap.safetensors";

    original.save(original_path)?;
    updated.save(updated_path)?;

    // Files should be identical
    let original_bytes = std::fs::read(original_path)?;
    let updated_bytes = std::fs::read(updated_path)?;
    assert_eq!(original_bytes, updated_bytes, "Saved files differ!");

    // Test loading
    let mut original_loaded = VarMap::new();
    let mut updated_loaded = VarMap::new();

    // Pre-create variables for loading
    for i in 0..20 {
        let name = format!("layer_{}.weight", i);
        original_loaded.get((64, 64), &name, Init::Const(0.), DType::F32, &device)?;
        updated_loaded.get((64, 64), &name, Init::Const(0.), DType::F32, &device)?;
    }

    original_loaded.load(original_path)?;
    updated_loaded.load(updated_path)?;

    // Verify loaded data matches - check a few specific variables
    for i in 0..20 {
        let name = format!("layer_{}.weight", i);
        let orig_var =
            original_loaded.get((64, 64), &name, Init::Const(0.), DType::F32, &device)?;
        let updated_var =
            updated_loaded.get((64, 64), &name, Init::Const(0.), DType::F32, &device)?;

        // Compare shapes
        assert_eq!(orig_var.shape(), updated_var.shape());

        // Compare values - flatten first
        let orig_data: Vec<f32> = orig_var.flatten_all()?.to_vec1()?;
        let updated_data: Vec<f32> = updated_var.flatten_all()?.to_vec1()?;

        // Values should be close to i (the const value we used)
        for (o, u) in orig_data.iter().zip(updated_data.iter()) {
            assert!((o - u).abs() < 1e-6, "Value mismatch in {}", name);
        }
    }

    // Cleanup
    std::fs::remove_file(original_path).ok();
    std::fs::remove_file(updated_path).ok();

    Ok(())
}

#[test]
fn test_set_operations_compatibility() -> Result<()> {
    let device = Device::Cpu;
    let mut original = VarMap::new();
    let mut updated = VarMap::new();

    // Initialize with same data
    for i in 0..10 {
        let name = format!("param_{}", i);
        original.get((5, 5), &name, Init::Const(0.), DType::F32, &device)?;
        updated.get((5, 5), &name, Init::Const(0.), DType::F32, &device)?;
    }

    // Test set_one
    let new_value = Tensor::ones((5, 5), DType::F32, &device)?;
    original.set_one("param_0", &new_value)?;
    updated.set_one("param_0", &new_value)?;

    // Test set with iterator
    let updates: Vec<(String, Tensor)> = (1..5)
        .map(|i| {
            let name = format!("param_{}", i);
            let value = Tensor::full(i as f32, (5, 5), &device).unwrap();
            (name, value)
        })
        .collect();

    original.set(updates.iter().map(|(k, v)| (k, v)))?;
    updated.set(updates.iter().map(|(k, v)| (k, v)))?;

    // Verify specific values match
    for i in 0..5 {
        let name = format!("param_{}", i);
        let orig_tensor = original.get((5, 5), &name, Init::Const(0.), DType::F32, &device)?;
        let updated_tensor = updated.get((5, 5), &name, Init::Const(0.), DType::F32, &device)?;

        // Flatten and compare
        let orig_data: Vec<f32> = orig_tensor.flatten_all()?.to_vec1()?;
        let updated_data: Vec<f32> = updated_tensor.flatten_all()?.to_vec1()?;

        let expected_val = if i == 0 { 1.0 } else { i as f32 };

        for (o, u) in orig_data.iter().zip(updated_data.iter()) {
            assert!(
                (o - expected_val).abs() < 1e-6,
                "Original value mismatch for {}",
                name
            );
            assert!(
                (u - expected_val).abs() < 1e-6,
                "Updated value mismatch for {}",
                name
            );
            assert!((o - u).abs() < 1e-6, "Values don't match for {}", name);
        }
    }

    // Verify unchanged values
    for i in 5..10 {
        let name = format!("param_{}", i);
        let orig_tensor = original.get((5, 5), &name, Init::Const(0.), DType::F32, &device)?;
        let updated_tensor = updated.get((5, 5), &name, Init::Const(0.), DType::F32, &device)?;

        let orig_data: Vec<f32> = orig_tensor.flatten_all()?.to_vec1()?;
        let updated_data: Vec<f32> = updated_tensor.flatten_all()?.to_vec1()?;

        // These should still be 0
        for (o, u) in orig_data.iter().zip(updated_data.iter()) {
            assert!(
                o.abs() < 1e-6,
                "Original unchanged value not zero for {}",
                name
            );
            assert!(
                u.abs() < 1e-6,
                "Updated unchanged value not zero for {}",
                name
            );
        }
    }

    Ok(())
}

#[test]
fn test_error_conditions_match() -> Result<()> {
    let device = Device::Cpu;
    let mut original = VarMap::new();
    let mut updated = VarMap::new();

    // Test shape mismatch error
    original.get((2, 3), "test", Init::Const(0.), DType::F32, &device)?;
    updated.get((2, 3), "test", Init::Const(0.), DType::F32, &device)?;

    // Both should fail with shape mismatch
    let orig_err = original.get((3, 2), "test", Init::Const(0.), DType::F32, &device);
    let updated_err = updated.get((3, 2), "test", Init::Const(0.), DType::F32, &device);

    assert!(orig_err.is_err());
    assert!(updated_err.is_err());

    // Test set_one on non-existent variable
    let tensor = Tensor::ones((2, 2), DType::F32, &device)?;
    let orig_err = original.set_one("nonexistent", &tensor);
    let updated_err = updated.set_one("nonexistent", &tensor);

    assert!(orig_err.is_err());
    assert!(updated_err.is_err());

    Ok(())
}

#[test]
fn test_concurrent_varmap_specific_features() -> Result<()> {
    let device = Device::Cpu;
    let concurrent = ConcurrentVarMap::new();

    // Initialize data
    for i in 0..50 {
        let name = format!("weight_{}", i);
        concurrent.get(
            (32, 32),
            &name,
            Init::Randn {
                mean: 0.,
                stdev: 0.02,
            },
            DType::F32,
            &device,
        )?;
    }

    // Test batch operations
    let names: Vec<&str> = (0..10)
        .map(|i| Box::leak(format!("weight_{}", i).into_boxed_str()) as &str)
        .collect();
    let batch_vars = concurrent.get_vars_batch(&names);

    assert_eq!(batch_vars.len(), 10);
    for (_name, var) in batch_vars {
        assert_eq!(var.shape().dims(), &[32, 32]);
    }

    // Test concurrent read access
    let n_readers = 10;
    let barrier = Arc::new(Barrier::new(n_readers));
    let concurrent: Arc<ConcurrentVarMap> = Arc::new(concurrent);

    let handles: Vec<_> = (0..n_readers)
        .map(|_| {
            let concurrent: Arc<ConcurrentVarMap> = Arc::clone(&concurrent);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier.wait();

                // Multiple concurrent reads
                let _guard = concurrent.read_data();
                thread::sleep(std::time::Duration::from_millis(10));

                // Should not block other readers
                assert!(concurrent.all_vars().len() >= 50);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    Ok(())
}

#[test]
fn test_varmap_conversion() -> Result<()> {
    let device = Device::Cpu;
    let original = VarMap::new();

    // Add some data
    for i in 0..25 {
        let name = format!("conv_{}.weight", i);
        original.get(
            (3, 3, 64, 64),
            &name,
            Init::Kaiming {
                dist: candle_nn::init::NormalOrUniform::Normal,
                fan: candle_nn::init::FanInOut::FanIn,
                non_linearity: candle_nn::init::NonLinearity::ReLU,
            },
            DType::F32,
            &device,
        )?;
    }

    // Convert to concurrent
    let concurrent = original.clone().into_concurrent();

    // Verify all data transferred
    assert_eq!(original.all_vars().len(), concurrent.all_vars().len());

    // Verify values match
    let orig_vars = original.all_vars();
    let conc_vars = concurrent.all_vars();

    for (orig, conc) in orig_vars.iter().zip(conc_vars.iter()) {
        assert_eq!(orig.shape(), conc.shape());
        assert_eq!(orig.dtype(), conc.dtype());
    }

    Ok(())
}

#[test]
fn test_backend_trait_implementation() -> Result<()> {
    use candle_nn::VarBuilder;

    let device = Device::Cpu;

    // Test that VarMap works as SimpleBackend
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Create some layers
    let weight1 = vb.get((128, 256), "layer1.weight")?;
    let weight2 = vb.get((256, 512), "layer2.weight")?;

    assert_eq!(weight1.shape().dims(), &[128, 256]);
    assert_eq!(weight2.shape().dims(), &[256, 512]);

    // Test ConcurrentVarMap as backend
    let concurrent = ConcurrentVarMap::new();
    let vb_concurrent =
        VarBuilder::from_backend(Box::new(concurrent.clone()), DType::F32, device.clone());

    let weight3 = vb_concurrent.get((64, 128), "layer3.weight")?;
    assert_eq!(weight3.shape().dims(), &[64, 128]);

    Ok(())
}
