import pandas as pd

def load_data(path):
    df = pd.read_csv(path, encoding="latin-1", header=None)
    df.columns = ["sentiment", "id", "date", "query", "user", "text"]
    return df