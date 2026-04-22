pub fn get_anyres_image_grid_shape(
    image_size: (u32, u32),
    grid_pinpoints: &[(u32, u32)],
    patch_size: u32,
) -> (u32, u32) {
    let (width, height) = select_best_resolution(image_size, grid_pinpoints);
    (width / patch_size, height / patch_size)
}

pub fn select_best_resolution(
    original_size: (u32, u32),
    possible_resolutions: &[(u32, u32)],
) -> (u32, u32) {
    let (original_width, original_height) = original_size;
    let mut best_fit = (0, 0);
    let original_width_f = original_width as f32;
    let original_height_f = original_height as f32;
    let mut max_effective_resolution = 0_u32;
    let mut min_wasted_resolution = u32::MAX;
    for (width, height) in possible_resolutions {
        let width_f = *width as f32;
        let height_f = *height as f32;
        let scale = (width_f / original_width_f).min(height_f / original_height_f);
        let (downscaled_width, downscaled_height) = (
            (original_width_f * scale) as u32,
            (original_height_f * scale) as u32,
        );
        let effective_resolution =
            std::cmp::min((*width) * (*height), downscaled_width * downscaled_height);
        let wasted_resolution = (*width) * (*height) - effective_resolution;
        if effective_resolution > max_effective_resolution
            || (effective_resolution == max_effective_resolution
                && wasted_resolution < min_wasted_resolution)
        {
            best_fit = (*width, *height);
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
        }
    }
    best_fit
}
