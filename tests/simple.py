import pyarrow as pa
import pandas
import numpy as np
from fastvs import knn
import random
import time
import pyarrow.compute as pc
from sklearn.metrics.pairwise import cosine_similarity as cosine

DIM = 1536


def create_dataset(num_rows=1_000_000, num_dims=DIM):
    # Generate a large dataset with num_rows rows and num_dims dimensions
    data = np.random.rand(num_rows, num_dims)

    offsets = np.arange(0, num_rows * num_dims + 1, num_dims, dtype=np.int32)
    flat_values = pa.array(data.reshape(-1), type=pa.float64())
    arrow_array = pa.ListArray.from_arrays(pa.array(offsets), flat_values)

    return arrow_array


def create_table():
    arrow_array = create_dataset(num_rows=1000)

    table = pa.Table.from_arrays([arrow_array], names=["points"])
    return table


def test_basic_functionality(reader, query_point, k, metric="euclidean"):
    # Query point should have the same dimensions as the dataset
    results, dists = knn(reader, "points", query_point, k, metric)
    return results
    # The assert might need to be adjusted depending on the random data
    # assert results == [expected indices]


def test_pure_numpy_euclidean(reader, query_point, k):
    # Convert the reader to a numpy array
    df = reader.read_pandas()
    data = df["points"].to_numpy()

    # Transform the data into a 2D array
    data = np.vstack(data)

    # Compute euclidean distance with numpy
    euclidean_dists = np.linalg.norm(data - query_point, axis=1)

    # Get the indices of the k smallest distances
    indices = np.argsort(euclidean_dists)[:k]
    return indices


def test_pure_numpy_cosine(reader, query_point, k):
    # Convert the reader to a numpy array
    df = reader.read_pandas()
    data = df["points"].to_numpy()

    # Transform the data into a 2D array
    data = np.vstack(data)

    # Normalize the data and the query point
    data_norm = data / np.linalg.norm(data, axis=1, keepdims=True)
    query_point_norm = query_point / np.linalg.norm(query_point)

    # Compute cosine similarity with numpy
    cosine_dists = np.dot(data_norm, query_point_norm)

    # Get the indices of the k smallest distances
    indices = np.argsort(cosine_dists)[-k:]
    return indices


def test_sklearn_cosine(reader, query_point, k):
    # Convert the reader to a numpy array
    df = reader.read_pandas()
    data = df["points"].to_numpy()

    # Transform the data into a 2D array
    data = np.vstack(data)

    # Run sklearn cosine similarity
    cosine_dists = cosine(data, query_point.reshape(1, -1))

    # Get the indices of the k smallest distances
    indices = np.argsort(cosine_dists)[-k:]
    return indices


if __name__ == "__main__":
    table = create_table()
    query_point = np.array([0.1] * DIM)
    k = 3
    metric = "cosine_similarity"

    # Start timer
    print("Starting our test")
    start = time.time()
    our_indices = test_basic_functionality(table.to_reader(), query_point, k, metric)
    print("Time taken: ", time.time() - start)

    print("Starting numpy test")
    start = time.time()
    numpy_indices = test_pure_numpy_cosine(table.to_reader(), query_point, k)
    print("Time taken: ", time.time() - start)

    print("Starting sklearn test")
    start = time.time()
    sklearn_indices = test_sklearn_cosine(table.to_reader(), query_point, k)
    print("Time taken: ", time.time() - start)

    # Sort and compare the results
    our_indices.sort()
    numpy_indices.sort()
    sklearn_indices.sort()
    print(our_indices == numpy_indices)
    print(our_indices == sklearn_indices)
