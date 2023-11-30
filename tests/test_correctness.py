import numpy as np
import pyarrow as pa
import pandas as pd
import pytest
from fastvs import search_arrow

from .utils import scipy_knn, create_dataset, numpy_knn

DIM = 1536


# Define test functions for each metric
@pytest.mark.parametrize(
    "metric", ["euclidean", "manhattan", "cosine_similarity", "inner_product"]
)
def test_knn_consistency_scipy(metric):
    num_rows = 1000  # Reduced number for testing
    k = 10
    query_point = np.random.rand(DIM)

    # Create dataset
    table = create_dataset(num_rows, DIM)

    # Get indices from Rust implementation
    rust_indices, _ = search_arrow(table, "points", query_point.tolist(), k, metric)

    # Get indices from SciPy implementation
    data = table["points"].to_numpy()
    scipy_indices = scipy_knn(data, query_point, k, metric)

    # Sort both indices lists
    rust_indices.sort()
    scipy_indices.sort()

    print(rust_indices)
    print(scipy_indices)

    assert all(
        rust_index == scipy_index
        for rust_index, scipy_index in zip(rust_indices, scipy_indices)
    )


@pytest.mark.parametrize(
    "metric", ["euclidean", "manhattan", "cosine_similarity", "inner_product"]
)
def test_knn_consistency_numpy(metric):
    num_rows = 1000  # Reduced number for testing
    k = 10
    query_point = np.random.rand(DIM)

    # Create dataset
    table = create_dataset(num_rows, DIM)

    # Get indices from Rust implementation
    rust_indices, _ = search_arrow(table, "points", query_point.tolist(), k, metric)

    # Get indices from np implementation
    data = table["points"].to_numpy()
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
