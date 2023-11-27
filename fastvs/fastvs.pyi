from typing import List, Tuple
import numpy as np
import pyarrow as pa

def knn(
    reader: pa.lib.RecordBatchReader,
    column_name: str,
    query_point: np.ndarray,
    k: int,
    metric: str,
) -> Tuple[List[int], List[float]]: ...
def distance(
    reader: pa.lib.RecordBatchReader,
    column_name: str,
    query_point: np.ndarray,
    k: int,
    metric: str,
) -> pa.Array: ...
