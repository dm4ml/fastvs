import numpy as np
import pyarrow as pa
import pandas as pd
import pytest
from fastvs import search_arrow
from scipy.spatial import distance

DIM = 1536


def create_dataset(num_rows=1_000, num_dims=DIM):
    data = np.random.rand(num_rows, num_dims)

    offsets = np.arange(0, num_rows * num_dims + 1, num_dims, dtype=np.int32)
    flat_values = pa.array(data.reshape(-1), type=pa.float64())
    arrow_array = pa.ListArray.from_arrays(pa.array(offsets), flat_values)

    table = pa.Table.from_arrays([arrow_array], names=["points"])
    return table


# Define a helper function for calculating nearest neighbors using NumPy
def numpy_knn(data, query_point, k, metric):
    if metric == "euclidean":
        dists = distance.cdist(data, [query_point], metric="euclidean").flatten()
        indices = np.argsort(dists)[:k]
    elif metric == "manhattan":
        dists = distance.cdist(data, [query_point], metric="cityblock").flatten()
        indices = np.argsort(dists)[:k]
    elif metric == "cosine_similarity":
        dists = distance.cdist(data, [query_point], metric="cosine").flatten()

        # Return complement
        dists = 1 - dists

        indices = np.argsort(dists)[-k:]
    elif metric == "inner_product":
        dists = np.dot(data, query_point).flatten()
        indices = np.argsort(dists)[-k:]

    return indices


# Define test functions for each metric
@pytest.mark.parametrize(
    "metric", ["euclidean", "manhattan", "cosine_similarity", "inner_product"]
)
def test_knn_consistency(metric):
    num_rows = 1000  # Reduced number for testing
    k = 10
    query_point = np.random.rand(DIM)

    # Create dataset
    table = create_dataset(num_rows, DIM)
    df = table.to_pandas()
    data = df["points"].to_numpy()
    data = np.vstack(data)  # Convert list of arrays to 2D array

    # Get indices from Rust implementation
    rust_indices, _ = search_arrow(table, "points", query_point.tolist(), k, metric)

    # Get indices from NumPy implementation
    numpy_indices = numpy_knn(data, query_point, k, metric)

    # Sort both indices lists
    rust_indices.sort()
    numpy_indices.sort()

    print(rust_indices)
    print(numpy_indices)

    assert all(
        rust_index == numpy_index
        for rust_index, numpy_index in zip(rust_indices, numpy_indices)
    )
