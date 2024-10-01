"""Module to create the matrix with the node features and the new dataframe with the needed columns."""

from pathlib import Path

import numpy as np
import pandas as pd
from dataset.utils import read_csv_chunked, save_csv_chunked

FILE_PATH = Path("dygnn/dataset/processed_data") / "citations.csv"
OUT_FOLDER = Path("dygnn/DyGLib/processed_data/citations")


def main():
    # Open the edge index file
    print(f"Reading the edge index from {FILE_PATH}...")
    assert FILE_PATH.exists(), f"{FILE_PATH} does not exist!"
    df = read_csv_chunked(file_path=FILE_PATH)
    print("Done!")
    print(f"Dataframe shape: {df.shape}")
    print(f"Dataframe columns: {df.columns}\n")

    # Create a new dataframe with the needed columns
    print("Creating a new dataframe with the needed columns...")
    u = df["source"]
    i = df["target"]
    ts = df["time_step"]
    label = np.zeros(df.shape[0])
    idx = df["index"]
    final_df = pd.DataFrame(
        {"": idx, "u": u, "i": i, "ts": ts, "label": label, "idx": idx}
    )
    final_df = final_df.reset_index(drop=True)
    print("Done!")
    print(f"Dataframe shape: {final_df.shape}")

    # Save the new dataframe
    print(f"Saving the new dataframe in {OUT_FOLDER}...")
    save_csv_chunked(df=final_df, file_path=OUT_FOLDER / "...")
    print("Done!")
    print(f"New dataframe saved in: {OUT_FOLDER / '...'}\n")

    # Create the matrix with the node features
    print("Creating the matrix with the node features...")
    unique_values = np.unique(np.concatenate((final_df["u"], final_df["i"])))
    n_rows = unique_values.shape[0]
    n_col = 172
    nodes_features = np.zeros((n_rows, n_col))
    print("Done!")
    print(f"Nodes features shape: {nodes_features.shape}")

    # Save the matrix with the node features
    print(f"Saving the matrix with the node features in {OUT_FOLDER}...")
    OUT_FOLDER.mkdir(parents=True, exist_ok=True)
    np.save(OUT_FOLDER / "nodes_features.npy", nodes_features)
    print("Done!")
    print(f"Nodes features saved in: {OUT_FOLDER / 'nodes_features.npy'}\n")


if __name__ == "__main__":
    main()
