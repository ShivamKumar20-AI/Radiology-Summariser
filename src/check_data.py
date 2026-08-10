import pandas as pd


files = [
    "train_data.csv",
    "test_data.csv",
    "predictions_data.csv"
]


for filename in files:

    path = "data/processed/" + filename
    data = pd.read_csv(path)

    print("\nFILE:", filename)
    print("Rows:", len(data))
    print("Columns:", data.columns.tolist())
    print(data.head(3).to_string(index=False))