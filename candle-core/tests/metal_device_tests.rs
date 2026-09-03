use candle_core::{Device, Error, Result};

#[cfg(feature = "metal")]
mod metal_device_tests {
    use super::{
        Device,
        Error,
        Result,
    };
    use candle_core::metal_backend::MetalError;

    #[test]
    fn test_metal_device_unavailability() -> Result<()> {
        // Temporarily disable Metal feature if it's enabled for this test to simulate unavailability.
        // This is a placeholder and might require more sophisticated feature management
        // if candle-core doesn't provide a direct way to disable Metal at runtime.
        // For now, we assume that on systems without Metal, Device::metal_if_available will fail.

        // Test with an invalid ordinal (e.g., a very high number) to ensure it returns an error.
        let result = Device::metal_if_available(999);
        match result {
            Ok(_) => panic!("Metal device initialization unexpectedly succeeded with invalid ordinal."),
            Err(Error::Metal(MetalError::Message(e))) => {
                assert!(e.contains("No Metal devices found.") || e.contains("Metal device ordinal 999 is out of bounds"));
                eprintln!("Caught expected Metal initialization error: {}", e);
            },
            Err(e) => panic!("Caught unexpected error type during Metal initialization: {:?}", e),
        }

        // Test on a system where Metal is generally unavailable. This is harder to test directly
        // without specific environment setup or mocking. The above invalid ordinal test is a good proxy.
        // If run on a system without Metal, Device::metal_if_available(0) should also return an error.
        let result_zero_ordinal = Device::metal_if_available(0);
        if let Err(Error::Metal(MetalError::Message(e))) = result_zero_ordinal {
            // This message depends on the actual MetalError generated when no devices are found.
            // It should contain "No Metal devices found." based on our change.
            assert!(e.contains("No Metal devices found.") || e.contains("Metal device ordinal 0 is out of bounds"));
            eprintln!("Caught expected Metal initialization error for ordinal 0: {}", e);
        } else if result_zero_ordinal.is_ok() {
            eprintln!("Metal device initialization for ordinal 0 succeeded, this system has Metal.");
        }
        Ok(())
    }
}