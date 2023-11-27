import os
from typing import List, Tuple
import silence_tensorflow.auto

import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences

TrainTestData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
TrainTestValData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def load_csv(fp_train: str, fp_test: str, label_col: str, drop_cols: List[str], normalization: bool = False) -> TrainTestData:
    X_train = read_csv(fp_train).astype(np.float32)  # DataFrames directly converted to float32
    X_test = read_csv(fp_test).astype(np.float32)

    y_train = X_train[label_col].astype(np.int32)
    y_test = X_test[label_col].astype(np.int32)
    X_train.drop(columns=drop_cols + [label_col], inplace=True)  # Dropping cols and label column
    X_test.drop(columns=drop_cols + [label_col], inplace=True)

    # Other data sources are already normalized. RGB values are always in range 0 to 255.
    if normalization:
        mini, maxi = X_train.min(axis=0), X_train.max(axis=0)
        X_train -= mini
        X_train /= maxi - mini
        X_test -= mini
        X_test /= maxi - mini

    return X_train.values, y_train.values, X_test.values, y_test.values  # Numpy arrays

def get_train_test_val(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, val_frac: float = 0.25) -> TrainTestValData:
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_frac, stratify=y_train)
    return X_train, y_train, X_test, y_test, X_val, y_val

