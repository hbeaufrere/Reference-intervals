"""
Built-in example dataset for the Reference Interval Calculator.

Contains triglyceride and cholesterol values (mg/dL) from 80 mixed-sex
cockatiels (Nymphicus hollandicus).  A few values are intentionally
missing to demonstrate how the tool handles incomplete data.

These are simulated data generated to be realistic for the species based
on published psittacine lipid reference ranges.
"""

import numpy as np
import pandas as pd


def get_example_dataframe() -> pd.DataFrame:
    """Return the example cockatiel lipid dataset as a DataFrame.

    Returns
    -------
    pd.DataFrame
        80 rows, columns: Bird_ID, Triglycerides_mg_dL, Cholesterol_mg_dL.
        Contains a small number of intentional missing values (NaN).
    """
    rng = np.random.RandomState(2024)
    n = 80

    ids = [f"CKT-{i+1:03d}" for i in range(n)]

    # Triglycerides: right-skewed (gamma), mean ~85 mg/dL, range ~35-200
    trig = rng.gamma(shape=4.5, scale=19.0, size=n)
    trig = np.round(trig, 1)

    # Cholesterol: roughly normal, mean ~210 mg/dL, SD ~28
    chol = rng.normal(loc=210, scale=28, size=n)
    chol = np.round(chol, 1)

    # Introduce a few missing values (3 in triglycerides, 2 in cholesterol)
    trig_float = trig.astype(float)
    chol_float = chol.astype(float)

    missing_trig = rng.choice(n, size=3, replace=False)
    missing_chol = rng.choice(n, size=2, replace=False)
    trig_float[missing_trig] = np.nan
    chol_float[missing_chol] = np.nan

    df = pd.DataFrame({
        "Bird_ID": ids,
        "Triglycerides_mg_dL": trig_float,
        "Cholesterol_mg_dL": chol_float,
    })
    return df
