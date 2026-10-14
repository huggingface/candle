#[cfg(feature = "metal")]
#[test]
fn test_metal_device_ordinal_bounds() {
    use candle_core::Device;

    // This test verifies that creating a Metal device with an invalid ordinal
    // returns a proper error instead of panicking.
    // See issue #3566: https://github.com/huggingface/candle/issues/3566

    // Try to create a device with a very large ordinal that definitely doesn't exist
    let result = Device::new_metal(999);

    match result {
        Ok(_) => {
            // This is unexpected unless the system actually has 1000+ Metal devices
            println!("Unexpectedly created Metal device with ordinal 999");
        }
        Err(e) => {
            // This is the expected behavior - should return an error, not panic
            let error_msg = e.to_string();
            assert!(
                error_msg.contains("no Metal device at ordinal") ||
                error_msg.contains("MTLCopyAllDevices"),
                "Error message should mention device ordinal or MTLCopyAllDevices, got: {}",
                error_msg
            );
            println!("✓ Correctly returned error: {}", error_msg);
        }
    }

    // Also test ordinal 0 - it might fail if no Metal devices are available,
    // but it should return an error, not panic
    let result = Device::new_metal(0);
    match result {
        Ok(device) => {
            println!("✓ Successfully created Metal device: {:?}", device);
        }
        Err(e) => {
            println!("✓ Metal device creation failed with error (this is OK): {}", e);
        }
    }
}
