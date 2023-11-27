from scipy.spatial import distance
import numpy as np
import pyarrow as pa

DIM = 1536


def create_dataset(num_rows=1_000, num_dims=DIM):
    data = np.random.rand(num_rows, num_dims)

    offsets = np.arange(0, num_rows * num_dims + 1, num_dims, dtype=np.int32)
    flat_values = pa.array(data.reshape(-1), type=pa.float64())
    arrow_array = pa.ListArray.from_arrays(pa.array(offsets), flat_values)

    table = pa.Table.from_arrays([arrow_array], names=["points"])

    return table


# Define a helper function for calculating nearest neighbors using NumPy
def numpy_knn(table, query_point, k, metric):
    df = table.to_pandas()
    data = df["points"].values
    data = np.vstack(data)  # Convert list of arrays to 2D array

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
