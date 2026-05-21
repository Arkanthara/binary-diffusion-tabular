from typing import List, Tuple, Literal, Optional, Dict, Union
from joblib import Parallel, delayed
import math
import joblib

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import torch

from binary_diffusion_tabular import TASK, PathOrStr


__all__ = [
    "FixedSizeBinaryTableTransformation",
]


COLUMN_DTYPE = Literal["numerical", "categorical"]
LABELS = Union[np.ndarray, pd.Series, torch.Tensor]


def column_to_fixed_size_binary(
    column: pd.Series,
    dtype: COLUMN_DTYPE,
    metadata: Optional[Dict] = None,
    size: Optional[int] = None,
) -> Tuple[pd.Series, Dict, int]:
    """
    Convert a pandas DataFrame column to fixed-size binary representation with automatic size calculation,
    and return relevant metadata.

    Args:
        column (pd.Series): DataFrame column to be converted.
        dtype (COLUMN_DTYPE): The type of data ('numerical', 'categorical').
        metadata (dict): Metadata necessary for conversion (min-max for numerical, mapping for categorical).
        size (int): The size of the fixed size binary.

    Returns:
        tuple: A tuple containing the converted column and metadata (min-max for numerical, mapping for categorical).

    Notes:
        size is calculated automatically. For categorical data, the size is calculated as the log2 of the number of
        unique values. For numerical data, the size is set to 32.
    """

    if dtype == "categorical":
        unique_values = len(column.unique())
        if size is None:
            size = math.ceil(math.log2(unique_values)) if unique_values > 1 else 1
    else:
        # default size for numerical columns
        size = 32

    def numerical_to_binary(val, min_val: float, max_val: float):
        return format(
            int((val - min_val) / (max_val - min_val) * (2**size - 1)),
            f"0{size}b",
        )

    if dtype == "numerical":
        if not pd.api.types.is_numeric_dtype(column):
            raise ValueError(
                "Column must contain numeric values for numerical data type."
            )
        if metadata is None:
            metadata = {"min": column.min(), "max": column.max()}

        min_val = metadata["min"]
        max_val = metadata["max"]

        converted_column = column.apply(numerical_to_binary, args=(min_val, max_val))
    else:
        if metadata is None:
            category_map = {
                category: index for index, category in enumerate(column.unique())
            }
            metadata = {"category_map": category_map}
        else:
            category_map = metadata["category_map"]

        def get_category_index(val):
            cat_idx = category_map.get(val, None)
            if cat_idx is not None:
                return format(cat_idx, f"0{size}b")
            else:
                return None

        converted_column = column.apply(get_category_index)

    return converted_column, metadata, size


def column_from_fixed_size_binary(
    binary_column: pd.Series, metadata: Dict, dtype: COLUMN_DTYPE
) -> pd.Series:
    """
    Convert a binary representation back to its original form using provided metadata.

    Args:
        binary_column (pd.Series): Column with binary representations.
        metadata (dict): Metadata necessary for conversion (min-max for numerical, mapping for categorical).
        dtype (COLUMN_DTYPE): The type of data ('numerical', 'categorical').

    Returns:
        pd.Series: A column with original values.
    """

    if dtype == "numerical":
        min_val = metadata["min"]
        max_val = metadata["max"]
        return binary_column.apply(
            lambda x: int(x, 2) / (2 ** len(x) - 1) * (max_val - min_val) + min_val
        )

    elif dtype == "categorical":
        category_map = metadata["category_map"]
        inverse_map = {v: k for k, v in category_map.items()}
        return binary_column.apply(lambda x: inverse_map.get(int(x, 2), None))
    else:
        raise ValueError(
            "Data type not recognized. Choose 'numerical' or 'categorical'."
        )


def pandas_row_to_tensor(row: pd.Series) -> torch.Tensor:
    row_str = "".join(row.astype(str))
    row_np = np.array(list(row_str))
    row_np = row_np.astype(int)
    row_binary = torch.tensor(row_np, dtype=torch.float)
    return row_binary


class FixedSizeBinaryTableTransformation:
    """Transformation to convert pandas dataframe to fixed size binary tensor and back"""

    def __init__(
        self,
        task: TASK,
        numerical_columns: List[str],
        categorical_columns: List[str],
        parallel: bool = False,
    ):
        self.task = task
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.parallel = parallel
        self.label_encoder = (
            LabelEncoder() if self.task == "classification" else MinMaxScaler()
        )

        self.fitted = False
        self.fitted_label = False
        self.metadata = None
        self.size = None

    def save_checkpoint(self, path_checkpoint: PathOrStr) -> None:
        """
        Save the current state of the transformation to a file.

        Args:
            path_checkpoint: Path to the file where the state will be saved.
        """
        joblib.dump(self, path_checkpoint)

    @classmethod
    def from_checkpoint(
        cls, path_checkpoint: PathOrStr
    ) -> "FixedSizeBinaryTableTransformation":
        """Loads the transformation from a .joblib file.

        Args:
            path_checkpoint: Path to the .joblib file.

        Returns:
            FixedSizeBinaryTableTransformation: The loaded transformation.
        """

        transformer = joblib.load(path_checkpoint)
        return transformer

    @property
    def row_size(self) -> int:
        if not self.fitted:
            raise RuntimeError(
                "FixedSizeBinaryTableTransformation has not been fitted."
            )
        return sum(self.size.values())

    @property
    def n_classes(self) -> int:
        if not self.fitted:
            raise RuntimeError(
                "FixedSizeBinaryTableTransformation has not been fitted."
            )

        if self.task != "classification":
            raise ValueError("Task must be 'classification'.")

        return len(self.label_encoder.classes_)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits transformation and transforms the input dataframe

        Transformation doesn't handle the empty values
        All handling of empty values, dropping columns, etc should be done beforehand.

        Args:
            X: input dataframe.
            y: target dataframe. Can be None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: transformed X and y. If y is provided

            OR

            torch.Tensor: transformed X, if y is not provided
        """

        if self.fitted:
            raise RuntimeError(
                "Transformation already fitted. Use transform() instead."
            )

        x_binary, metadata, size = self._convert_df_to_fixed_size_binary_tensor(X)
        self.metadata = metadata
        self.size = size

        self.fitted = True

        if y is not None:
            y_trans = self.fit_transform_label(y)
            self.fitted_label = True
            y_trans = torch.tensor(y_trans, dtype=torch.float)
            return x_binary, y_trans

        return x_binary

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Transforms the input dataframe into a fixed size binary tensor.

        Args:
            X: input dataframe.
            y: target dataframe. Can be None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: transformed X and y. If y is provided

            torch.Tensor: transformed X, if y is not provided
        """

        if not self.fitted:
            raise RuntimeError("Fit before transform. Use fit_transform() instead.")

        x_binary, *_ = self._convert_df_to_fixed_size_binary_tensor(X)

        if y is not None:
            if not self.fitted_label:
                raise RuntimeError("Label encoder not fitted.")

            y_trans = self.transform_label(y)
            return x_binary, y_trans

        return x_binary

    def fit_transform_label(self, y: LABELS) -> torch.Tensor:
        """Fits encoder for labels and transforms the labels

        Args:
            y: labels to transform. Can be np.ndarray, pd.Series.

        Returns:
            torch.Tensor: transformed labels
        """

        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        y_trans = self.label_encoder.fit_transform(y.ravel())
        y_trans = torch.tensor(y_trans, dtype=torch.float)

        self.fitted_label = True
        return y_trans

    def transform_label(self, y: LABELS) -> torch.Tensor:
        """Transforms the labels

        Args:
            y: labels to transform. Can be np.ndarray, pd.Series.

        Returns:
            torch.Tensor: transformed labels
        """

        if not self.fitted_label:
            raise RuntimeError("Label encoder not fitted.")

        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        y_trans = self.label_encoder.transform(y.ravel())
        y_trans = torch.tensor(y_trans, dtype=torch.float)
        return y_trans

    def inverse_transform(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        """Inverse transformation to convert binary fixed size tensor and labels back to original dataframe.

        Args:
            X: input binary fixed size tensor

            y: target dataframe. Can be None.

        Returns:
            pd.DataFrame: transformed dataframe, if y is not provided

            or

            pd.DataFrame, np.ndarray: transformed dataframe and labels, if y is provided
        """

        if not self.fitted:
            raise RuntimeError("Fit before transform. Use fit_transform() instead.")

        df = self._convert_fixed_size_binary_tensor_to_df(X)

        if y is not None:
            if not self.fitted_label:
                raise RuntimeError("Label encoder not fitted.")

            y_trans = self.inverse_transform_label(y)
            return df, y_trans

        return df

    def inverse_transform_label(self, y: LABELS) -> np.ndarray:
        """Inverse transformation for labels

        Args:
            y: labels to transform. Can be np.ndarray, pd.Series or torch.Tensor.

        Returns:
            np.ndarray: transformed labels
        """

        if not self.fitted_label:
            raise RuntimeError("Label encoder not fitted.")

        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        if self.task == "classification":
            y = y.astype(int)

        y_trans = self.label_encoder.inverse_transform(y.ravel())
        return y_trans

    def _convert_fixed_size_binary_tensor_to_df(
        self, rows_binary: torch.Tensor
    ) -> pd.DataFrame:
        rows_np = rows_binary.detach().cpu().numpy().astype(int)
        rows_str = rows_np.astype(str)

        df_bin = {}
        start = 0
        for col, size in self.size.items():
            end = start + size
            df_bin[col] = ["".join(row) for row in rows_str[:, start:end]]
            start = end

        df_bin = pd.DataFrame(df_bin)

        df = pd.DataFrame()
        for col in df_bin.columns:
            df[col] = column_from_fixed_size_binary(
                df_bin[col],
                metadata=self.metadata[col],
                dtype="numerical" if col in self.numerical_columns else "categorical",
            )

        return df

    def _convert_df_to_fixed_size_binary_tensor(self, df: pd.DataFrame):
        df_binary = pd.DataFrame()
        metadata = {}
        size = {}

        columns = df.columns
        for col in columns:
            col_binary, metadat_col, size_col = column_to_fixed_size_binary(
                column=df[col],
                dtype="numerical" if col in self.numerical_columns else "categorical",
                metadata=None if self.metadata is None else self.metadata[col],
                size=None if self.size is None else self.size[col],
            )
            df_binary[col] = col_binary
            metadata[col] = metadat_col
            size[col] = size_col

        if self.parallel:
            # rows_binary = parallelize_dataframe(df, pandas_row_to_tensor, 4)
            n_jobs = -1
            rows_binary = Parallel(n_jobs=n_jobs)(
                delayed(pandas_row_to_tensor)(row) for _, row in df_binary.iterrows()
            )
        else:
            rows_binary = df_binary.apply(pandas_row_to_tensor, axis=1).tolist()

        rows_binary = torch.stack(rows_binary, dim=0)
        return rows_binary, metadata, size
