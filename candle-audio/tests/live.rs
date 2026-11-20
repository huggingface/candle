#![cfg(feature = "live-io")]
use candle_audio::live::{enumerate_devices, Direction};
#[test]
#[ignore = "queries native audio device metadata only; never opens or starts a stream"]
fn enumerate_native_audio_devices_without_capture() {
    for direction in [Direction::Input, Direction::Output] {
        let devices = enumerate_devices(direction).unwrap();
        eprintln!("{direction:?}: {} devices", devices.len());
        for device in devices {
            assert!(!device.id.is_empty());
            eprintln!(
                "{} {} default={} format={:?}",
                device.id, device.name, device.is_default, device.default_format
            );
        }
    }
}
