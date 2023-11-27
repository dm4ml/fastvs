"""
This file contains helper functions to do nearest neighbor search on
PyArrow Tables and Pandas DataFrames.
"""

from .fastvs import nearest_neighbor_search
import pyarrow as pa
import pandas as pd
from typing import List, Tuple


def search_table(
    table: pa.Table,
    column_name: str,
    query_point: list,
    k: int,
    metric: str,
) -> Tuple[List[int], List[float]]:
    """
    Search a PyArrow Table for the k nearest neighbors of a query point.

    Parameters
    ----------
    table : pyarrow.Table
        The table to search.
    column_name : str
        The column name to search.
    query_point : list
        The query point.
    k : int
        The number of nearest neighbors to return.
    metric : str
        The metric to use for the search.

    Returns
    -------
    list
        The indices of the k nearest neighbors.
    list
        The distances of the k nearest neighbors.

    """
    # Turn table into a RecordBatchReader.
    reader = table.to_reader()
    return nearest_neighbor_search(reader, column_name, query_point, k, metric)


def search_df(
    df: pd.DataFrame,
    column_name: str,
    query_point: list,
    k: int,
    metric: str,
) -> Tuple[List[int], List[float]]:
    """
    Search a Pandas DataFrame for the k nearest neighbors of a query point.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to search.
    column_name : str
        The column name to search.
    query_point : list
        The query point.
    k : int
        The number of nearest neighbors to return.
    metric : str
        The metric to use for the search.

    Returns
    -------
    list
        The indices of the k nearest neighbors.
    list
        The distances of the k nearest neighbors.
    """
    # Turn DataFrame into pyarrow.Table. Only select the column we need.
    table = pa.Table.from_pandas(df[[column_name]])
    return search_table(table, column_name, query_point, k, metric)
