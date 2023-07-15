use std::str::FromStr;

pub fn get_num_threads() -> usize {
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
    {
        Some(x) if x > 0 => x,
        Some(_) | None => num_cpus::get(),
    }
}

pub fn has_mkl() -> bool {
    #[cfg(feature = "mkl")]
    return true;
    #[cfg(not(feature = "mkl"))]
    return false;
}

pub fn cuda_is_available() -> bool {
    #[cfg(feature = "cuda")]
    return true;
    #[cfg(not(feature = "cuda"))]
    return false;
}
