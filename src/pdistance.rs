use arrow::array::PrimitiveArray;
use arrow::compute::kernels::aggregate::sum;
use arrow::compute::kernels::numeric::{mul, sub};
use arrow::datatypes::Float64Type;
use arrow::error::{ArrowError, Result};

pub fn euclidean_distance(
    a: &PrimitiveArray<Float64Type>,
    b: &PrimitiveArray<Float64Type>,
) -> Result<f64> {
    let differences = sub(a, b)?;
    let squared = mul(&differences, &differences)?;

    let squared = squared
        .as_any()
        .downcast_ref::<PrimitiveArray<Float64Type>>()
        .ok_or_else(|| {
            ArrowError::ComputeError("Failed to downcast to PrimitiveArray".to_string())
        })?;
    let squared_sum = sum(squared);
    squared_sum.map(|sum| sum.sqrt()).ok_or_else(|| {
        arrow::error::ArrowError::ComputeError("Failed to compute Euclidean distance".to_string())
    })
}

pub fn manhattan_distance(
    a: &PrimitiveArray<Float64Type>,
    b: &PrimitiveArray<Float64Type>,
) -> Result<f64> {
    let differences = sub(a, b)?;
    let differences = differences
        .as_any()
        .downcast_ref::<PrimitiveArray<Float64Type>>()
        .ok_or_else(|| {
            ArrowError::ComputeError("Failed to downcast to PrimitiveArray".to_string())
        })?;

    // Use map to compute absolute values and then sum
    let absolute_sum = differences
        .iter()
        .map(|v| {
            match v {
                Some(val) => val.abs(),
                None => 0.0, // Handle null values, if any
            }
        })
        .sum::<f64>();

    Ok(absolute_sum)
}

pub fn inner_product(
    a: &PrimitiveArray<Float64Type>,
    b: &PrimitiveArray<Float64Type>,
) -> Result<f64> {
    let products = mul(a, b)?;
    let products = products
        .as_any()
        .downcast_ref::<PrimitiveArray<Float64Type>>()
        .ok_or_else(|| {
            ArrowError::ComputeError("Failed to downcast to PrimitiveArray".to_string())
        })?;

    let product_sum = sum(products);
    product_sum.ok_or_else(|| {
        arrow::error::ArrowError::ComputeError("Failed to compute inner product".to_string())
    })
}

pub fn cosine_similarity(
    a: &PrimitiveArray<Float64Type>,
    b: &PrimitiveArray<Float64Type>,
) -> Result<f64> {
    let dot_product = inner_product(a, b)?;

    let squared_a = mul(a, a)?;
    let squared_a = squared_a
        .as_any()
        .downcast_ref::<PrimitiveArray<Float64Type>>()
        .ok_or_else(|| {
            ArrowError::ComputeError("Failed to downcast to PrimitiveArray".to_string())
        })?;

    let squared_b = mul(b, b)?;
    let squared_b = squared_b
        .as_any()
        .downcast_ref::<PrimitiveArray<Float64Type>>()
        .ok_or_else(|| {
            ArrowError::ComputeError("Failed to downcast to PrimitiveArray".to_string())
        })?;

    let norm_a = sum(squared_a).map(|sum| sum.sqrt()).ok_or_else(|| {
        arrow::error::ArrowError::ComputeError("Failed to compute norm_a".to_string())
    })?;
    let norm_b = sum(squared_b).map(|sum| sum.sqrt()).ok_or_else(|| {
        arrow::error::ArrowError::ComputeError("Failed to compute norm_b".to_string())
    })?;

    Ok(dot_product / (norm_a * norm_b))
}
