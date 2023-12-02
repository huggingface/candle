use candle::{Device, Result, Tensor};

pub fn linspace(start: f64, stop: f64, steps: usize) -> Result<Tensor> {
    if steps == 0 {
        Tensor::from_vec(Vec::<f64>::new(), steps, &Device::Cpu)
    } else if steps == 1 {
        Tensor::from_vec(vec![start], steps, &Device::Cpu)
    } else {
        let delta = (stop - start) / (steps - 1) as f64;
        let vs = (0..steps)
            .map(|step| start + step as f64 * delta)
            .collect::<Vec<_>>();
        Tensor::from_vec(vs, steps, &Device::Cpu)
    }
}

/// A linear interpolator for a sorted array of x and y values.
struct LinearInterpolator<'x, 'y> {
    xp: &'x [f64],
    fp: &'y [f64],
    cache: usize,
}

impl<'x, 'y> LinearInterpolator<'x, 'y> {
    fn accel_find(&mut self, x: f64) -> usize {
        let xidx = self.cache;
        if x < self.xp[xidx] {
            self.cache = self.xp[0..xidx].partition_point(|o| *o < x);
            self.cache = self.cache.saturating_sub(1);
        } else if x >= self.xp[xidx + 1] {
            self.cache = self.xp[xidx..self.xp.len()].partition_point(|o| *o < x) + xidx;
            self.cache = self.cache.saturating_sub(1);
        }

        self.cache
    }

    fn eval(&mut self, x: f64) -> f64 {
        if x < self.xp[0] || x > self.xp[self.xp.len() - 1] {
            return f64::NAN;
        }

        let idx = self.accel_find(x);

        let x_l = self.xp[idx];
        let x_h = self.xp[idx + 1];
        let y_l = self.fp[idx];
        let y_h = self.fp[idx + 1];
        let dx = x_h - x_l;
        if dx > 0.0 {
            y_l + (x - x_l) / dx * (y_h - y_l)
        } else {
            f64::NAN
        }
    }
}

pub fn interp(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    let mut interpolator = LinearInterpolator { xp, fp, cache: 0 };
    x.iter().map(|&x| interpolator.eval(x)).collect()
}
