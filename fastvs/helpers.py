"""
This file contains functions to do nearest neighbor search on
PyArrow Tables and Pandas DataFrames.
"""

from .fastvs import knn, distance
import pyarrow as pa
import pandas as pd
from typing import List, Tuple


def search_arrow(
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
    return knn(reader, column_name, query_point, k, metric)


def search_pandas(
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
    return search_arrow(table, column_name, query_point, k, metric)


def apply_distance_arrow(
    table: pa.Table,
    column_name: str,
    query_point: list,
    metric: str,
) -> pa.Array:
    """
    Applies distance function to a PyArrow table and returns an array
    of distances in the order of the table.

    Parameters
    ----------
    table : pyarrow.Table
        The table to search.
    column_name : str
        The column name to search.
    query_point : list
        The query point.
    metric : str
        The metric to use for the search.

    Returns
    -------
    pyarrow.Array
        The distances of the k nearest neighbors.

    """
    # Turn table into a RecordBatchReader.
    reader = table.to_reader()
    return distance(reader, column_name, query_point, metric)


def apply_distance_pandas(
    df: pd.DataFrame,
    column_name: str,
    query_point: list,
    metric: str,
) -> pd.Series:
    """
    Applies distance function to a Pandas DataFrame and returns an array
    of distances in the order of the table.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to search.
    column_name : str
        The column name to search.
    query_point : list
        The query point.
    metric : str
        The metric to use for the search.

    Returns
    -------
    pyarrow.Array
        The distances of the k nearest neighbors.

    """
    # Turn DataFrame into pyarrow.Table. Only select the column we need.
    table = pa.Table.from_pandas(df[[column_name]])
    distances = apply_distance_arrow(table, column_name, query_point, metric)

    # Turn the array into a pandas Series
    return pd.Series(distances.to_pandas())
