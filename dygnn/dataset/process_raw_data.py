"""Module to create the edge index from the raw data."""

import pickle
from pathlib import Path

from dataset.utils import (
    create_years_subset,
    extract_keys,
    get_edge_index,
    map_and_normalize,
    save_csv_chunked,
)

FILE_PATH = Path("dygnn/dataset/raw_data") / "dblp_v14.json"
OUT_FOLDER = Path("dygnn/dataset/processed_data")
N = 10000000
YEAR_START = 2010


def main():
    # Extract keys from the file to create the initial dataframe
    print(f"Creating the initial dataframe from {FILE_PATH}...")
    assert FILE_PATH.exists(), f"{FILE_PATH} does not exist!"
    df = extract_keys(file_path=FILE_PATH, N=10000000, year_start=YEAR_START)
    print("Done!")
    print(f"Dataframe shape: {df.shape}")
    print(f"Dataframe columns: {df.columns}\n")

    # Create a subset of the dataframe for each year
    print("Creating a subset of the dataframe for each year...")
    dfs = create_years_subset(df)
    print("Done!")
    print(f"Total number of years: {len(dfs)}")
    print(f"Years: {list(dfs.keys())}\n")

    # Get the edge index where each node is an author and each edge is a co-authorship
    print("Creating the edge index...")
    edge_index = get_edge_index(dfs)
    print("Done!")
    print(f"Edge index shape: {edge_index.shape}\n")

    # Map node's name, year and normalize the citations' number
    edge_index, mapping_names, mapping_years = map_and_normalize(edge_index)
    print("Done!\n")

    # Save the edge index, mapping names and mapping years
    print("Saving the edge index, mapping names and mapping years...")
    OUT_FOLDER.mkdir(parents=True, exist_ok=True)
    save_csv_chunked(df=edge_index, file_path=OUT_FOLDER / "citations.csv")
    with open(OUT_FOLDER / "mapping_names.pkl", "wb") as f:
        pickle.dump(mapping_names, f)
    with open(OUT_FOLDER / "mapping_years.pkl", "wb") as f:
        pickle.dump(mapping_years, f)
    print("Done!")
    print(f"Files saved in: {OUT_FOLDER}")


if __name__ == "__main__":
    main()
