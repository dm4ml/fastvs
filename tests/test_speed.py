import numpy as np
import pytest
from fastvs import search_arrow
import time
from .utils import numpy_knn, create_dataset

DIM = 1536


def execute_with_timer(func, *args, **kwargs):
    start = time.time()
    func(*args, **kwargs)
    end = time.time()
    return end - start


# Define test functions for each metric
@pytest.mark.parametrize(
    "metric", ["euclidean", "manhattan", "cosine_similarity", "inner_product"]
)
def test_knn_speed(metric):
    num_rows = 100000
    k = 10
    query_point = np.random.rand(DIM)

    # Create dataset
    table = create_dataset(num_rows, DIM)

    # Get time from Rust implementation
    rust_time = execute_with_timer(
        search_arrow, table, "points", query_point.tolist(), k, metric
    )

    # Get time from NumPy implementation
    numpy_time = execute_with_timer(numpy_knn, table, query_point, k, metric)

    print(f"Rust time: {rust_time}")
    print(f"NumPy time: {numpy_time}")
    assert rust_time < numpy_time
