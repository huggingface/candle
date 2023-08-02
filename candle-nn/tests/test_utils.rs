#![allow(dead_code)]
use candle::{Result, Tensor};

pub fn to_vec0_round(t: &Tensor, digits: i32) -> Result<f32> {
    let b = 10f32.powi(digits);
    let t = t.to_vec0::<f32>()?;
    Ok(f32::round(t * b) / b)
}

pub fn to_vec1_round(t: &Tensor, digits: i32) -> Result<Vec<f32>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec1::<f32>()?;
    let t = t.iter().map(|t| f32::round(t * b) / b).collect();
    Ok(t)
}

pub fn to_vec2_round(t: &Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec2::<f32>()?;
    let t = t
        .iter()
        .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
        .collect();
    Ok(t)
}

pub fn to_vec3_round(t: Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3::<f32>()?;
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    Ok(t)
}
