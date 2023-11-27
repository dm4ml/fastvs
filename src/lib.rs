/*
This file defines the function that allows for nearest-neighbor search
on a PyArrow table column. The function is called from Python.
 */

pub mod pdistance;

use std::sync::Arc;

use arrow::array::{Array, Float64Array, ListArray, PrimitiveArray, PrimitiveBuilder};
use arrow::datatypes::Float64Type;
use arrow::error::ArrowError;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::PyArrowType;
use pdistance::{cosine_similarity, euclidean_distance, inner_product, manhattan_distance};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use rayon::prelude::*;

use arrow::array::ArrayData;
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, ToByteSlice};
use pyo3::types::PyAny;

// // Define the distance function type
// type DistanceFn = dyn Fn(&PrimitiveArray<Float64Type>, &PrimitiveArray<Float64Type>) -> Result<f64, ArrowError>
//     + Sync
//     + Send;

// // Function to create the distance function based on metric
// fn create_distance_fn(metric: &str) -> Result<Arc<DistanceFn>, PyValueError> {
//     match metric {
//         "euclidean" => Ok(Arc::new(euclidean_distance)),
//         "manhattan" => Ok(Arc::new(manhattan_distance)),
//         "inner_product" => Ok(Arc::new(inner_product)),
//         "cosine_similarity" => Ok(Arc::new(cosine_similarity)),
//         _ => Err(PyValueError::new_err("Invalid distance metric")),
//     }
// }

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

    let mut results = batch_results.into_par_iter().flatten().collect::<Vec<_>>();

    results.retain(|&(_, ref distance)| distance.is_ok());

    // Check if results are empty after filtering
    if results.is_empty() {
        return Err(PyValueError::new_err(
            "All distance calculations failed, possibly due to a dimensionality mismatch",
        ));
    }

    // Sort the results
    // TODO: See if we can use Arrow for partial sorting
    if ascending {
        results.par_sort_by(|a, b| {
            a.1.as_ref()
                .unwrap()
                .partial_cmp(&b.1.as_ref().unwrap())
                .unwrap()
        });
    } else {
        results.par_sort_by(|a, b| {
            b.1.as_ref()
                .unwrap()
                .partial_cmp(&a.1.as_ref().unwrap())
                .unwrap()
        });
    }

    // Separate the indices and distances into two vectors
    let (nearest_neighbors, distances): (Vec<usize>, Vec<f64>) = results
        .into_iter()
        .take(k)
        .filter_map(|(index, distance_result)| {
            distance_result.ok().map(|distance| (index, distance))
        })
        .unzip();

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

    // // Extract and collect the distances into a Vec<f64>, filtering out errors
    // let distances: Vec<f64> = computed_distances
    //     .into_iter()
    //     .filter_map(|(_, dist)| dist.ok())
    //     .collect();

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
