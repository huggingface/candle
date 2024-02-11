#[cfg(feature = "microphone")]
mod whisper_micro;

#[cfg(feature = "microphone")]
mod multilingual;

fn main() {
    #[cfg(feature = "microphone")]
    {
        let _ = whisper_micro::main();
    }

    #[cfg(not(feature = "microphone"))]
    {
        println!("Please enable `--features microphone` to run this example");
    }
}
