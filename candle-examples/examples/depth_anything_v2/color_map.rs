use enterpolation::linear::ConstEquidistantLinear;
use enterpolation::Generator;
use palette::LinSrgb;

use candle::Tensor;

pub struct SpectralRColormap {
    gradient: ConstEquidistantLinear<f32, LinSrgb, 9>,
}

impl SpectralRColormap {
    pub(crate) fn new() -> Self {
        // Define a colormap similar to 'Spectral_r' by specifying key colors.
        // got the colors from ChatGPT-4o
        let gradient = ConstEquidistantLinear::<f32, _, 9>::equidistant_unchecked([
            LinSrgb::new(0.3686, 0.3098, 0.6353), // Dark blue
            LinSrgb::new(0.1961, 0.5333, 0.7412), // Blue
            LinSrgb::new(0.4000, 0.7608, 0.6471), // Cyan
            LinSrgb::new(0.6706, 0.8667, 0.6431), // Green
            LinSrgb::new(0.9020, 0.9608, 0.5961), // Yellow
            LinSrgb::new(0.9961, 0.8784, 0.5451), // Orange
            LinSrgb::new(0.9922, 0.6824, 0.3804), // Red
            LinSrgb::new(0.9569, 0.4275, 0.2627), // Dark red
            LinSrgb::new(0.8353, 0.2431, 0.3098), // Dark purple
        ]);
        Self { gradient }
    }

    fn get_color(&self, value: f32) -> LinSrgb {
        self.gradient.gen(value)
    }

    pub fn gray2color(&self, gray: &Tensor) -> candle::Result<Tensor> {
        println!("Gray: {:?}", gray.dims());
        let gray_values: Vec<f32> = gray.flatten_all()?.to_vec1()?;
        let rgb_values: Vec<f32> = gray_values
            .iter()
            .map(|g| self.get_color(*g))
            .flat_map(|rgb| [rgb.red, rgb.green, rgb.blue])
            .collect();

        let [.., height, width] = gray.dims() else {
            candle::bail!("Not enough dims!")
        };

        let color = Tensor::from_vec(rgb_values, (*height, *width, 3), gray.device())?;

        color.permute((2, 0, 1))
    }
}
