file_path = r"D:\My Things\Eng. Things\My Machine Learning Journey\Mostafa Saad\ML\Projects\creditcard-fraud-detection\src\data\creditcard.csv"

import pandas as pd

def load_data(path=file_path):
    df = pd.read_csv(path)
    return df
