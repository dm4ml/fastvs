from typing import List, Tuple
from pyarrow.lib import RecordBatchReader
import numpy as np

def knn(
    reader: RecordBatchReader,
    column_name: str,
    query_point: np.ndarray,
    k: int,
    metric: str,
) -> Tuple[List[int], List[float]]: ...
