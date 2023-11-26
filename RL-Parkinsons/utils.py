import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_csv(fp: str = "./data/creditcard.csv", fp_dest: str = "./data",
              name: str = "credit", test_size: int = 0.5, strat_col: str = "Class") -> None:
    if not os.path.isfile(fp):
        raise FileNotFoundError(f"File at {fp} does not exist.")
    if not os.path.isdir(fp_dest):
        raise ValueError(f"Directory at {fp_dest} does not exist.")
    if not 0 < test_size < 1:
        raise ValueError(f"{test_size} is not in interval 0 < x < 1.")

    df = pd.read_csv(fp)

    if not (strat_col in df.columns):
        raise ValueError(f"Stratify column {strat_col} not found in DataFrame.")

    train, test = train_test_split(df, test_size=test_size, stratify=df[strat_col])

    train.to_csv(f"{fp_dest}/{name}0.csv", index=False)
    test.to_csv(f"{fp_dest}/{name}1.csv", index=False)


def rounded_dict(d: dict, precision: int = 6) -> dict:
    return {k: round(v, precision) for k, v in d.items()}
