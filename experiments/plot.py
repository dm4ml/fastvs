import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time
from fastvs import search_arrow
from tests.utils import numpy_knn, create_dataset


def execute_with_timer(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return end - start, result


# Experiment settings for various dimensions
dataset_size = 100000  # Fixed dataset size
dimensions = [1536, 2048, 4096, 8192]  # Varying dimensions
k = 10
metric = "cosine_similarity"
results_dims = []
num_trials = 10  # Number of trials

# Run the experiment for various dimensions
for dim in dimensions:
    for _ in range(num_trials):
        query_point = np.random.rand(dim)
        table = create_dataset(dataset_size, dim)

        # Time Rust implementation
        rust_time, _ = execute_with_timer(
            search_arrow, table, "points", query_point.tolist(), k, metric
        )
        rust_qps = 1 / rust_time

        # Time NumPy implementation
        numpy_time, _ = execute_with_timer(numpy_knn, table, query_point, k, metric)
        numpy_qps = 1 / numpy_time

        # Record results for dimensions
        results_dims.append(
            {"Dimension": dim, "Implementation": "FastVS", "QPS": rust_qps}
        )
        results_dims.append(
            {"Dimension": dim, "Implementation": "SciPy", "QPS": numpy_qps}
        )

# Convert results to DataFrame and calculate average QPS for each dimension and implementation
df_dims = (
    pd.DataFrame(results_dims)
    # .groupby(["Dimension", "Implementation"])
    # .mean()
    # .reset_index()
)

# Plotting for various dimensions
sns.lineplot(data=df_dims, x="Dimension", y="QPS", hue="Implementation")
plt.title("Average QPS Across Different Dimensions (Dataset Size = 100K)")
plt.xlabel("Vector Dimension/Length")
plt.ylabel("Queries Per Second")
# plt.tight_layout()
plt.savefig("experiments/plots/dimensions.png")

# Experiment settings for various dataset sizes
dimensions = [1536]  # Fixed dimension
dataset_sizes = [
    1000,
    5000,
    10000,
    50000,
    100000,
]  # Varying dataset sizes
results_sizes = []

# Run the experiment for various dataset sizes
for num_rows in dataset_sizes:
    for _ in range(num_trials):
        query_point = np.random.rand(dimensions[0])
        table = create_dataset(num_rows, dimensions[0])

        # Time Rust implementation
        rust_time, _ = execute_with_timer(
            search_arrow, table, "points", query_point.tolist(), k, metric
        )
        rust_qps = 1 / rust_time

        # Time NumPy implementation
        numpy_time, _ = execute_with_timer(numpy_knn, table, query_point, k, metric)
        numpy_qps = 1 / numpy_time

        # Record results for dataset sizes
        results_sizes.append(
            {"Dataset Size": num_rows, "Implementation": "FastVS", "QPS": rust_qps}
        )
        results_sizes.append(
            {"Dataset Size": num_rows, "Implementation": "SciPy", "QPS": numpy_qps}
        )

# Convert results to DataFrame and calculate average QPS for each dataset size and implementation
df_sizes = (
    pd.DataFrame(results_sizes)
    # .groupby(["Dataset Size", "Implementation"])
    # .mean()
    # .reset_index()
)

# Clear
plt.clf()

# Plotting for various dataset sizes
sns.lineplot(data=df_sizes, x="Dataset Size", y="QPS", hue="Implementation")
plt.title("Average QPS Across Different Dataset Sizes (Dimension = 1536)")
plt.xlabel("Dataset Size")
plt.ylabel("Queries Per Second")
# plt.tight_layout()
plt.savefig("experiments/plots/datasetsizes.png")
