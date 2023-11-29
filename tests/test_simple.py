import pytest
import pyarrow as pa
import pandas as pd
from fastvs import (
    search_arrow,
    search_pandas,
    apply_distance_arrow,
    apply_distance_pandas,
)

# Sample data for testing
sample_data = {"points": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}
query_point = [2.0, 3.0]
column_name = "points"
metric = "euclidean"
k = 2

sample_table = pa.Table.from_pandas(pd.DataFrame(sample_data))
sample_df = sample_table.to_pandas()


def test_search_arrow():
    indices, distances = search_arrow(sample_table, column_name, query_point, k, metric)
    assert len(indices) == k
    assert len(distances) == k


def test_error_with_int_array():
    sample_data = {"points": [[1, 2], [3, 4], [5, 6]]}
    table = pa.Table.from_pandas(pd.DataFrame(sample_data))
    with pytest.raises(TypeError):
        search_arrow(table, "points", query_point, k, metric)


def test_search_pandas():
    indices, distances = search_pandas(sample_df, column_name, query_point, k, metric)
    assert len(indices) == k
    assert len(distances) == k


def test_apply_distance_arrow():
    distances = apply_distance_arrow(sample_table, column_name, query_point, metric)
    assert isinstance(distances, pa.Array)
    assert len(distances) == len(sample_data["points"])


def test_apply_distance_pandas():
    distances = apply_distance_pandas(sample_df, column_name, query_point, metric)
    assert isinstance(distances, pd.Series)
    assert len(distances) == len(sample_data["points"])
