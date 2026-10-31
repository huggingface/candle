//! Image preprocessing for LFM2-VL, ported from `Lfm2VlImageProcessor` and `Lfm2VlProcessor`
//! in transformers.
//!
//! Choosing the sizes and the tile grid, resizing and cutting the image, turning the tiles into
//! padded patches, and building the image tokens of the prompt.

use candle::{DType, Device, Result, Tensor};
use image::imageops::FilterType;
use image::RgbImage;

/// Preprocessing settings, as found in `preprocessor_config.json`, or under `image_processor`
/// in `processor_config.json` for newer checkpoints.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct Lfm2VlProcessorConfig {
    pub downsample_factor: usize,
    pub do_image_splitting: bool,
    pub min_tiles: usize,
    pub max_tiles: usize,
    pub use_thumbnail: bool,
    pub min_image_tokens: usize,
    pub max_image_tokens: usize,
    pub encoder_patch_size: usize,
    pub tile_size: usize,
    pub max_pixels_tolerance: f64,
    /// PIL resampling filter for the resizes: 2 for bilinear, 3 for bicubic.
    pub resample: usize,
}

impl Default for Lfm2VlProcessorConfig {
    fn default() -> Self {
        Self {
            downsample_factor: 2,
            do_image_splitting: true,
            min_tiles: 2,
            max_tiles: 10,
            use_thumbnail: true,
            min_image_tokens: 64,
            max_image_tokens: 256,
            encoder_patch_size: 16,
            tile_size: 512,
            max_pixels_tolerance: 2.0,
            resample: 2,
        }
    }
}

/// `(pixel_values, spatial_shapes, pixel_attention_mask)`, see [`Lfm2VlProcessorConfig::preprocess`].
pub type ProcessedImages = (Tensor, Vec<(usize, usize)>, Tensor);

/// How an image is cut before going through the vision tower.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageLayout {
    /// Grid of `tile_size` tiles as `(rows, cols)`, `(1, 1)` when the image is not split.
    pub grid: (usize, usize),
    /// `(height, width)` of the whole image when not split, or of the thumbnail otherwise.
    pub resized: (usize, usize),
    /// Whether a thumbnail follows the tiles.
    pub thumbnail: bool,
}

impl ImageLayout {
    pub fn is_split(&self) -> bool {
        self.grid != (1, 1)
    }

    /// `(height, width)` to resize the whole image to before cutting it into tiles.
    pub fn grid_size(&self, tile_size: usize) -> (usize, usize) {
        (self.grid.0 * tile_size, self.grid.1 * tile_size)
    }
}

/// Like Python's `round(number / factor) * factor`, which rounds halves to even.
fn round_by_factor(number: usize, factor: usize) -> usize {
    (number as f64 / factor as f64).round_ties_even() as usize * factor
}

impl Lfm2VlProcessorConfig {
    /// Side of a square of pixels that ends up as a single image token (32).
    fn total_factor(&self) -> usize {
        self.encoder_patch_size * self.downsample_factor
    }

    /// Number of pixels covered by `tokens` image tokens.
    fn pixels_for_tokens(&self, tokens: usize) -> usize {
        tokens * self.total_factor().pow(2)
    }

    /// Patch rows of the vision tower input, so the padded length of `pixel_values` (1024).
    pub fn max_num_patches(&self) -> usize {
        let thumbnail_patches = self.max_image_tokens * self.downsample_factor.pow(2);
        let tile_patches = if self.do_image_splitting {
            (self.tile_size / self.encoder_patch_size).pow(2)
        } else {
            0
        };
        thumbnail_patches.max(tile_patches)
    }

    /// Size `(height, width)` close to the original, divisible by 32, and worth between
    /// `min_image_tokens` and `max_image_tokens` tokens.
    pub fn smart_resize(&self, height: usize, width: usize) -> (usize, usize) {
        let factor = self.total_factor();
        let min_pixels = self.pixels_for_tokens(self.min_image_tokens) as f64;
        let max_pixels = self.pixels_for_tokens(self.max_image_tokens) as f64;
        let (h, w) = (height as f64, width as f64);

        let h_bar = factor.max(round_by_factor(height, factor));
        let w_bar = factor.max(round_by_factor(width, factor));

        if (h_bar * w_bar) as f64 > max_pixels {
            let beta = (h * w / max_pixels).sqrt();
            let shrink = |x: f64| factor.max((x / beta / factor as f64).floor() as usize * factor);
            (shrink(h), shrink(w))
        } else if ((h_bar * w_bar) as f64) < min_pixels {
            let beta = (min_pixels / (h * w)).sqrt();
            let grow = |x: f64| (x * beta / factor as f64).ceil() as usize * factor;
            (grow(h), grow(w))
        } else {
            (h_bar, w_bar)
        }
    }

    /// Whether the image has too many pixels to be squeezed into a single tile.
    pub fn is_image_too_large(&self, height: usize, width: usize) -> bool {
        let factor = self.total_factor();
        let h_bar = self.encoder_patch_size.max(round_by_factor(height, factor));
        let w_bar = self.encoder_patch_size.max(round_by_factor(width, factor));
        let limit =
            self.pixels_for_tokens(self.max_image_tokens) as f64 * self.max_pixels_tolerance;
        (h_bar * w_bar) as f64 > limit
    }

    /// Every `(cols, rows)` grid allowed by `min_tiles..=max_tiles`, fewest tiles first.
    fn target_ratios(&self) -> Vec<(usize, usize)> {
        let mut ratios = Vec::new();
        for cols in 1..=self.max_tiles {
            for rows in 1..=self.max_tiles {
                if (self.min_tiles..=self.max_tiles).contains(&(cols * rows)) {
                    ratios.push((cols, rows));
                }
            }
        }
        // Stable sort, so grids with the same number of tiles keep their order.
        ratios.sort_by_key(|&(cols, rows)| cols * rows);
        ratios
    }

    /// Grid `(rows, cols)` whose shape is closest to the image's; on a tie, the larger grid
    /// wins if the image covers more than half of it.
    pub fn grid_layout(&self, height: usize, width: usize) -> (usize, usize) {
        let aspect_ratio = width as f64 / height as f64;
        let area = (width * height) as f64;
        let mut best_diff = f64::INFINITY;
        let mut best = (1, 1);
        for (cols, rows) in self.target_ratios() {
            let diff = (aspect_ratio - cols as f64 / rows as f64).abs();
            if diff < best_diff {
                best_diff = diff;
                best = (cols, rows);
            } else if diff == best_diff {
                let target_area = (self.tile_size.pow(2) * cols * rows) as f64;
                if area > 0.5 * target_area {
                    best = (cols, rows);
                }
            }
        }
        (best.1, best.0)
    }

    /// Decides whether the image is split into tiles, and the sizes to resize it to.
    pub fn layout(&self, height: usize, width: usize) -> ImageLayout {
        let resized = self.smart_resize(height, width);
        let can_split = self.do_image_splitting && !(self.min_tiles == 1 && self.max_tiles == 1);
        if can_split && self.is_image_too_large(height, width) {
            let grid = self.grid_layout(height, width);
            ImageLayout {
                grid,
                resized,
                thumbnail: self.use_thumbnail && grid != (1, 1),
            }
        } else {
            ImageLayout {
                grid: (1, 1),
                resized,
                thumbnail: false,
            }
        }
    }

    /// Image tokens produced by a resized image of `(height, width)` pixels.
    fn num_tokens(&self, height: usize, width: usize) -> usize {
        let p = self.encoder_patch_size;
        let f = self.downsample_factor;
        (height / p).div_ceil(f) * (width / p).div_ceil(f)
    }

    /// The text that replaces one image in the prompt, e.g. for a 2x3 grid:
    /// `<|image_start|><|img_row_1_col_1|><image>...<|img_thumbnail|><image>...<|image_end|>`.
    pub fn image_tokens(&self, layout: &ImageLayout) -> String {
        const IMAGE: &str = "<image>";
        let (height, width) = layout.resized;
        let mut text = String::from("<|image_start|>");
        if layout.is_split() {
            let per_tile = self.num_tokens(self.tile_size, self.tile_size);
            for row in 1..=layout.grid.0 {
                for col in 1..=layout.grid.1 {
                    text.push_str(&format!("<|img_row_{row}_col_{col}|>"));
                    text.push_str(&IMAGE.repeat(per_tile));
                }
            }
            if layout.thumbnail {
                text.push_str("<|img_thumbnail|>");
                text.push_str(&IMAGE.repeat(self.num_tokens(height, width)));
            }
        } else {
            text.push_str(&IMAGE.repeat(self.num_tokens(height, width)));
        }
        text.push_str("<|image_end|>");
        text
    }

    /// Resizes an image to a `(3, height, width)` u8 tensor.
    ///
    /// transformers resizes with torchvision (antialiased); the filters of the `image` crate
    /// are close but not identical, so pixels can differ by a few levels. This is enough to
    /// change some words of long generations, but not the quality of the output.
    pub fn resize(&self, image: &RgbImage, (height, width): (usize, usize)) -> Result<Tensor> {
        let filter = match self.resample {
            2 => FilterType::Triangle,
            3 => FilterType::CatmullRom,
            r => candle::bail!(
                "unsupported resample filter {r}, expected 2 (bilinear) or 3 (bicubic)"
            ),
        };
        let resized = image::imageops::resize(image, width as u32, height as u32, filter);
        Tensor::from_vec(resized.into_raw(), (height, width, 3), &Device::Cpu)?.permute((2, 0, 1))
    }

    /// Resizes and cuts an image as `layout` says: the tiles row by row, then the thumbnail,
    /// in the order of the image tokens of the prompt.
    pub fn image_tiles(&self, image: &RgbImage, layout: &ImageLayout) -> Result<Vec<Tensor>> {
        if !layout.is_split() {
            return Ok(vec![self.resize(image, layout.resized)?]);
        }
        let tile = self.tile_size;
        let grid = self.resize(image, layout.grid_size(tile))?;
        let mut tiles = Vec::new();
        for row in 0..layout.grid.0 {
            for col in 0..layout.grid.1 {
                tiles.push(
                    grid.narrow(1, row * tile, tile)?
                        .narrow(2, col * tile, tile)?,
                );
            }
        }
        if layout.thumbnail {
            tiles.push(self.resize(image, layout.resized)?);
        }
        Ok(tiles)
    }

    /// Turns one resized tile `(C, H, W)` with values in `0..=255` into its normalized patches
    /// `(H/p * W/p, p * p * C)`, each patch flattened in `(row, col, channel)` order.
    fn patchify(&self, tile: &Tensor) -> Result<Tensor> {
        let p = self.encoder_patch_size;
        let (c, h, w) = tile.dims3()?;
        tile.to_dtype(DType::F32)?
            // Rescale by 1/255 then normalize with mean = std = 0.5: x / 127.5 - 1.
            .affine(1. / 127.5, -1.)?
            .reshape((c, h / p, p, w / p, p))?
            // (C, Hp, py, Wp, px) -> (Hp, Wp, py, px, C)
            .permute((1, 3, 2, 4, 0))?
            .reshape((h / p * (w / p), p * p * c))
    }

    /// Builds the vision tower inputs from the resized tiles of all the images, in prompt
    /// order: `pixel_values` `(N, max_num_patches, p * p * C)`, `spatial_shapes` in patches,
    /// and `pixel_attention_mask` `(N, max_num_patches)` with 1 for real patches.
    pub fn preprocess(&self, tiles: &[Tensor]) -> Result<ProcessedImages> {
        let max_len = self.max_num_patches();
        let p = self.encoder_patch_size;
        let mut pixel_values = Vec::with_capacity(tiles.len());
        let mut spatial_shapes = Vec::with_capacity(tiles.len());
        let mut masks = Vec::with_capacity(tiles.len());
        for tile in tiles {
            let (_, h, w) = tile.dims3()?;
            let patches = self.patchify(tile)?;
            let n = patches.dim(0)?;
            if n > max_len {
                candle::bail!("tile of {h}x{w} pixels has {n} patches, more than {max_len}")
            }
            // Pad with zeros at the end; the mask tells the attention to ignore those rows.
            pixel_values.push(patches.pad_with_zeros(0, 0, max_len - n)?);
            spatial_shapes.push((h / p, w / p));
            let mask: Vec<u32> = (0..max_len).map(|i| u32::from(i < n)).collect();
            masks.push(Tensor::new(mask, tile.device())?);
        }
        Ok((
            Tensor::stack(&pixel_values, 0)?,
            spatial_shapes,
            Tensor::stack(&masks, 0)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn small_image_is_resized_in_one_piece() {
        let cfg = Lfm2VlProcessorConfig::default();
        let layout = cfg.layout(256, 384);
        assert_eq!(layout.grid, (1, 1));
        assert_eq!(layout.resized, (256, 384));
        let text = cfg.image_tokens(&layout);
        assert_eq!(text.matches("<image>").count(), 96);
        assert!(text.starts_with("<|image_start|><image>"));
        assert!(text.ends_with("<image><|image_end|>"));
    }

    #[test]
    fn large_image_is_split_with_thumbnail() {
        let cfg = Lfm2VlProcessorConfig::default();
        let layout = cfg.layout(768, 1024);
        assert_eq!(layout.grid, (2, 3));
        assert_eq!(layout.resized, (416, 576));
        assert!(layout.thumbnail);
        assert_eq!(layout.grid_size(cfg.tile_size), (1024, 1536));
        let text = cfg.image_tokens(&layout);
        assert_eq!(text.matches("<image>").count(), 6 * 256 + 234);
        assert!(text.contains("<|img_row_2_col_3|>"));
        assert!(text.contains("<|img_thumbnail|>"));
    }

    #[test]
    fn tiles_follow_the_layout() -> Result<()> {
        let cfg = Lfm2VlProcessorConfig::default();
        let image = RgbImage::new(1024, 768);
        let layout = cfg.layout(768, 1024);
        let tiles = cfg.image_tiles(&image, &layout)?;
        let dims: Vec<_> = tiles.iter().map(|t| t.dims().to_vec()).collect();
        let mut expected = vec![vec![3, 512, 512]; 6];
        expected.push(vec![3, 416, 576]);
        assert_eq!(dims, expected);
        Ok(())
    }

    #[test]
    fn layouts_match_transformers() {
        let cfg = Lfm2VlProcessorConfig::default();
        // (height, width) -> (grid, resized), computed with transformers' Lfm2VlImageProcessor.
        let cases = [
            ((256, 384), (1, 1), (256, 384)),
            ((768, 1024), (2, 3), (416, 576)),
            ((400, 400), (1, 1), (384, 384)),
            ((1080, 1920), (2, 4), (384, 672)),
            ((100, 3000), (1, 1), (64, 2784)),
            ((3000, 100), (1, 1), (2784, 64)),
            ((720, 1280), (1, 2), (384, 672)),
            ((50, 50), (1, 1), (256, 256)),
            ((2000, 2000), (3, 3), (512, 512)),
            ((600, 900), (2, 3), (416, 608)),
            ((500, 520), (1, 1), (512, 512)),
            ((1200, 300), (1, 1), (1024, 256)),
        ];
        for ((h, w), grid, resized) in cases {
            let layout = cfg.layout(h, w);
            assert_eq!((layout.grid, layout.resized), (grid, resized), "{h}x{w}");
        }
    }

    #[test]
    fn patches_are_padded_and_ordered() -> Result<()> {
        let cfg = Lfm2VlProcessorConfig::default();
        // Pixel value = channel index, so the last dim must read c0, c1, c2, c0, ...
        let tile = Tensor::arange(0f32, 3., &Device::Cpu)?
            .reshape((3, 1, 1))?
            .broadcast_as((3, 32, 48))?
            .contiguous()?;
        let (pixel_values, shapes, mask) = cfg.preprocess(&[tile])?;
        assert_eq!(pixel_values.dims(), &[1, 1024, 768]);
        assert_eq!(shapes, vec![(2, 3)]);
        assert_eq!(mask.sum_all()?.to_scalar::<u32>()?, 6);
        let first = pixel_values
            .get(0)?
            .get(0)?
            .narrow(0, 0, 4)?
            .to_vec1::<f32>()?;
        let expected = [-1., 1. / 127.5 - 1., 2. / 127.5 - 1., -1.];
        for (a, b) in first.iter().zip(expected) {
            assert!((a - b).abs() < 1e-6);
        }
        Ok(())
    }
}
