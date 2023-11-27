/*
This file defines the function that allows for nearest-neighbor search
on a PyArrow table column. The function is called from Python.
 */

pub mod pdistance;

use std::sync::Arc;

use arrow::array::{Array, ArrayData, Float64Array, ListArray, PrimitiveArray};
use arrow::datatypes::Float64Type;
use arrow::error::ArrowError;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::PyArrowType;
use pdistance::{cosine_similarity, euclidean_distance, inner_product, manhattan_distance};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use rayon::prelude::*;

// Helper function to calculate distances
fn compute_distance_batch(
    reader: PyArrowType<ArrowArrayStreamReader>,
    column_name: &PyString,
    query_point: Vec<f64>,
    metric: &str,
) -> PyResult<Vec<Vec<(usize, Result<f64, ArrowError>)>>> {
    let query_primitive = Arc::new(PrimitiveArray::<Float64Type>::from(query_point));

    let distance_fn: Arc<
        dyn Fn(
                &PrimitiveArray<Float64Type>,
                &PrimitiveArray<Float64Type>,
            ) -> Result<f64, arrow::error::ArrowError>
            + Sync
            + Send,
    > = Arc::new(match metric {
        "euclidean" => {
            euclidean_distance
                as fn(
                    &PrimitiveArray<Float64Type>,
                    &PrimitiveArray<Float64Type>,
                ) -> Result<f64, arrow::error::ArrowError>
        }
        "manhattan" => {
            manhattan_distance
                as fn(
                    &PrimitiveArray<Float64Type>,
                    &PrimitiveArray<Float64Type>,
                ) -> Result<f64, arrow::error::ArrowError>
        }
        "inner_product" => {
            inner_product
                as fn(
                    &PrimitiveArray<Float64Type>,
                    &PrimitiveArray<Float64Type>,
                ) -> Result<f64, arrow::error::ArrowError>
        }
        "cosine_similarity" => {
            cosine_similarity
                as fn(
                    &PrimitiveArray<Float64Type>,
                    &PrimitiveArray<Float64Type>,
                ) -> Result<f64, arrow::error::ArrowError>
        }
        _ => {
            // Raise an error if the metric is not supported
            return Err(PyValueError::new_err(
                "Metric must be one of: euclidean, manhattan, inner_product, cosine_similarity",
            ));
        }
    });

    let reader = reader.0;
    let column_name_str = column_name.to_string_lossy();

    let batches: Result<Vec<_>, _> = reader
        .map(|batch_result| {
            batch_result
                .map_err(|e| PyValueError::new_err(format!("Error processing batch: {}", e)))
                .and_then(|batch| {
                    let column_index = batch.schema().index_of(&column_name_str).map_err(|e| {
                        PyValueError::new_err(format!("Error finding column: {}", e))
                    })?;
                    let distance_fn = Arc::clone(&distance_fn);
                    let query_primitive = Arc::clone(&query_primitive);

                    let column = batch.column(column_index);
                    let list_array = column
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .ok_or_else(|| PyValueError::new_err("Column must be a ListArray"))?;

                    // Parallel process each list in the list array
                    let batch_distances: Result<Vec<_>, _> = (0..list_array.len())
                        .into_par_iter()
                        .map(move |i| {
                            let value_array = list_array.value(i);
                            let value =
                                value_array.as_any().downcast_ref::<Float64Array>().unwrap();

                            let distance = distance_fn(&value, &query_primitive);
                            Ok((i, distance))
                        })
                        .collect();

                    batch_distances
                })
        })
        .collect();

    Ok(batches?)
}

#[pyfunction]
pub fn knn(
    reader: PyArrowType<ArrowArrayStreamReader>,
    column_name: &PyString,
    query_point: Vec<f64>,
    k: usize,
    metric: &str,
) -> PyResult<(Vec<usize>, Vec<f64>)> {
    let batch_results = compute_distance_batch(reader, column_name, query_point, metric)?;
    let ascending = matches!(metric, "euclidean" | "manhattan");

    // Process the batch results in a single pipeline
    let mut results: Vec<_> = batch_results
        .into_par_iter()
        .flatten() // Flatten the nested results
        .filter_map(|(index, dist_result)| {
            dist_result.ok().map(|distance| (index, distance)) // Filter and unwrap the Ok results
        })
        .collect();

    // Check if results are empty after filtering
    if results.is_empty() {
        return Err(PyValueError::new_err(
            "All distance calculations failed, possibly due to a dimensionality mismatch",
        ));
    }

    // Sort the results based on distance
    if ascending {
        results.par_sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    } else {
        results.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    }

    // Separate the indices and distances into two vectors
    let (nearest_neighbors, distances): (Vec<usize>, Vec<f64>) =
        results.into_iter().take(k).unzip();

    Ok((nearest_neighbors, distances))
}

#[pyfunction]
pub fn distance(
    reader: PyArrowType<ArrowArrayStreamReader>,
    column_name: &PyString,
    query_point: Vec<f64>,
    metric: &str,
) -> PyResult<PyArrowType<ArrayData>> {
    let batch_results = compute_distance_batch(reader, column_name, query_point, metric)?;

    let distances: Vec<f64> = batch_results
        .into_par_iter() // Use Rayon's parallel iterator
        .flatten() // Flatten to get a single iterator over all results
        .filter_map(|(_, dist_result)| dist_result.ok()) // Keep only Ok values
        .collect();

    // Turn this into a Float64Array
    let distance_array = Float64Array::from(distances);

    Ok(PyArrowType(distance_array.into_data()))
}

#[pymodule]
fn fastvs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(knn, m)?)?;
    m.add_function(wrap_pyfunction!(distance, m)?)?;
    Ok(())
}
