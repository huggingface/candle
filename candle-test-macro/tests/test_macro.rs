use candle_test_macro::test_device;
use candle::{Device, Result};

#[test_device]
fn test_example(dev: &Device) -> Result<()>{
    println!("Current device is {:?}", dev);
    Ok(())
}