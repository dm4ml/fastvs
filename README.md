# FastVS

FastVS (Fast Vector Search) is a Python library designed for exact vector search in dataframes or tables. It provides functionality to work with both PyArrow Tables and Pandas DataFrames, allowing users to perform nearest neighbor searches using various distance metrics. It is most optimized for PyArrow Tables, as it uses the Rust Arrow library under the hood for fast computation.

## Installation

FastVS can be installed using pip:

```bash
pip install fastvs
```

## Functions

FastVS offers the following main functions:

### `search_arrow`

Searches a PyArrow Table for the k nearest neighbors of a query point.

#### Parameters

- `table` (pyarrow.Table): The table to search.
- `column_name` (str): This column should be a list or np array type column, where each element is a vector of floats.
- `query_point` (list): The query point.
- `k` (int): The number of nearest neighbors to return.
- `metric` (str): The metric to use for the search (e.g., "euclidean", "manhattan", "cosine_similarity", "inner_product").

#### Returns

- Tuple[List[int], List[float]]: The indices and distances of the k nearest neighbors.

#### Usage

```python
import pyarrow as pa
from fastvs import search_arrow

indices, distances = search_arrow(your_pyarrow_table, "your_column", [1.0, 2.0], 5, "cosine_similarity")
```

### `search_pandas`

Searches a Pandas DataFrame for the k nearest neighbors of a query point. This function uses `search_table` under the hood. Note that this function is slower than `search_arrow` due to the copying of data from the DataFrame to the Arrow table format.

#### Parameters

- `df` (pandas.DataFrame): The DataFrame to search.
- `column_name` (str): The column name to search. This column should be a list or np array type column, where each element is a vector of floats.
- `query_point` (list): The query point.
- `k` (int): The number of nearest neighbors to return.
- `metric` (str): The metric to use for the search.

#### Returns

- Tuple[List[int], List[float]]: The indices and distances of the k nearest neighbors.

#### Usage

```python
import pandas as pd
from fastvs import search_pandas

df = pd.read_csv("your_dataset.csv")
indices, distances = search_pandas(df, "your_column", [1.0, 2.0], 5, "cosine_similarity")
```

### `apply_distance_arrow`

Applies a distance function to a PyArrow table and returns an array of distances.

#### Parameters

- `table` (pyarrow.Table): The table to search.
- `column_name` (str): The column name to search. This column should be a list or np array type column, where each element is a vector of floats.
- `query_point` (list): The query point.
- `metric` (str): The metric to use for the search.

#### Returns

- pyarrow.Array: The distances in the order of the table.

#### Usage

```python
import pyarrow as pa
from fastvs import apply_distance_arrow

table = pa.Table.from_pandas(your_dataframe)
distances = apply_distance_arrow(table, "your_column", [1.0, 2.0], "euclidean")
```

### `apply_distance_pandas`

Applies a distance function to a Pandas DataFrame and returns a Series of distances. Uses `apply_distance_arrow` under the hood.

#### Parameters

- `df` (pandas.DataFrame): The DataFrame to search.
- `column_name` (str): The column name to search. This column should be a list or np array type column, where each element is a vector of floats.
- `query_point` (list): The query point.
- `metric` (str): The metric to use for the search.

#### Returns

- pandas.Series: The distances as a pandas Series.

#### Usage

```python
import pandas as pd
from fastvs import apply_distance_pandas

df = pd.read_csv("your_dataset.csv")
distances = apply_distance_pandas(df, "your_column", [1.0, 2.0], "euclidean")
```

## Supported Metrics

FastVS supports various distance metrics, including:

- Euclidean ("euclidean")
- Manhattan ("manhattan")
- Inner Product ("inner_product")
- Cosine Similarity ("cosine_similarity")

Please ensure your data is appropriate for the chosen metric.

## Contribution

Contributions to FastVS are welcome! Please submit your pull requests to the repository or open an issue for any bugs or feature requests.

To Dos:

- [ ] Clean up rust code
- [ ] Add CI and linting

## License

FastVS is released under the MIT License. See the LICENSE file in the repository for more details.
