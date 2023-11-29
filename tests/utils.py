from scipy.spatial import distance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

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
    # Convert to numpy
    data = table["points"].to_numpy()
    data = np.vstack(data)  # Convert list of arrays to 2D array

    if metric == "euclidean":
        dists = distance.cdist(data, [query_point], metric="euclidean").flatten()
        indices = np.argpartition(dists, k)[:k]
    elif metric == "manhattan":
        dists = distance.cdist(data, [query_point], metric="cityblock").flatten()
        indices = np.argpartition(dists, k)[:k]
    elif metric == "cosine_similarity":
        dists = distance.cdist(data, [query_point], metric="cosine").flatten()

        # Return complement
        dists = 1 - dists

        indices = np.argpartition(dists, -k)[-k:]
    elif metric == "inner_product":
        dists = np.dot(data, query_point).flatten()
        indices = np.argpartition(dists, -k)[-k:]

    return indices


# TODO: fix this because it's not working

# def _flatten_data(table):
#     # Assuming 'points' is a column of lists, we flatten it
#     flat_data = pc.list_flatten(table.column("points"))
#     # Calculate the number of elements in each list (assuming uniform length)
#     num_elements = pc.list_value_length(table.column("points")[0]).as_py()
#     return flat_data, num_elements

# def pyarrow_knn(table, query_point, k, metric):
#     # Get the pyarrow array
#     data = table["points"]
#     flat_data, num_elements = _flatten_data(table)
#     # query_point_array = pa.array(query_point * (len(flat_data) // num_elements))

#     query_point_array = pa.array(query_point * len(flat_data))

#     print(query_point_array.shape)
#     print(flat_data.shape)

#     if metric == "euclidean":
#         # Compute distances with pyarrow array functions
#         dists = pc.power(
#             pc.sum(pc.power(pc.subtract(flat_data, query_point_array), 2), axis=1), 0.5
#         )

#     elif metric == "manhattan":
#         dists = pc.sum(pc.abs(pc.subtract(flat_data, query_point_array)), axis=1)

#     elif metric == "cosine_similarity":
#         # Dot product of each vector in data with the query point
#         dot_product = pc.sum(pc.multiply(flat_data, query_point_array), axis=1)

#         # Magnitude (Euclidean norm) of each vector in data
#         magnitude_data = pc.sum(pc.power(flat_data, 2), axis=1) ** 0.5

#         # Magnitude of the query point
#         magnitude_query = pc.sum(pc.power(query_point_array, 2), axis=1) ** 0.5

#         # Cosine similarity
#         dists = pc.divide(dot_product, pc.multiply(magnitude_data, magnitude_query))
#     elif metric == "inner_product":
#         # Calculate inner product
#         dists = pc.sum(pc.multiply(flat_data, query_point_array), axis=1)

#     # Find the indices of the k smallest distances
#     pivot = (
#         k - 1
#         if metric not in ["inner_product", "cosine_similarity"]
#         else len(dists) - k
#     )
#     partition_indices = pc.partition_nth_indices(dists, pivot)
#     top_k_indices = pc.slice(partition_indices, 0, k)

#     return top_k_indices.to_pylist()
