"""Module to create the edge index from the raw data."""

import argparse
import pickle
from pathlib import Path

from dataset.utils import (
    create_years_subset,
    extract_keys,
    get_edge_index,
    map_and_normalize,
    save_csv_chunked,
)


def main(file_path, out_folder, n, year_start):
    # Extract keys from the file to create the initial dataframe
    print(f"Creating the initial dataframe from {file_path}...")
    assert file_path.exists(), f"{file_path} does not exist!"
    df = extract_keys(file_path=file_path, N=n, year_start=year_start)
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
    out_folder.mkdir(parents=True, exist_ok=True)
    save_csv_chunked(df=edge_index, file_path=out_folder / "citations.csv")
    with open(out_folder / "mapping_names.pkl", "wb") as f:
        pickle.dump(mapping_names, f)
    with open(out_folder / "mapping_years.pkl", "wb") as f:
        pickle.dump(mapping_years, f)
    print("Done!")
    print(f"Files saved in: {out_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create edge index from raw data")
    parser.add_argument(
        "--file_path",
        type=Path,
        default=Path("dygnn/dataset/raw_data") / "dblp_v14.json",
        help="Path to the input JSON file",
    )
    parser.add_argument(
        "--out_folder",
        type=Path,
        default=Path("dygnn/dataset/processed_data"),
        help="Path to the output folder",
    )
    parser.add_argument(
        "--n", type=int, default=10000000, help="Number of entries to process"
    )
    parser.add_argument(
        "--year_start", type=int, default=2010, help="Starting year for data processing"
    )

    args = parser.parse_args()

    main(args.file_path, args.out_folder, args.n, args.year_start)
