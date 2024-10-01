"""Utility functions for dataset module."""

from pathlib import Path
from typing import Optional

import ijson
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from pyarrow import csv
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

KEYS_TO_EXTRACT = [
    "id",
    "title",
    "keywords",
    "year",
    "n_citation",
    "abstract",
    "authors",
    "doc_type",
    "references",
]


def extract_author_fields(
    authors_list: list[dict],
) -> tuple[list[Optional[str]], list[Optional[str]], list[Optional[str]]]:
    """
    Extract author fields from a list of authors.

    Args:
        authors_list: A list of dictionaries containing author information.

    Returns:
        author_ids: List of author IDs.
        author_names: List of author names.
        author_orgs: List of author organizations.
    """
    if not authors_list:
        return None, None, None

    author_ids = []
    author_names = []
    author_orgs = []

    for author in authors_list:
        author_ids.append(author.get("id", None))
        author_names.append(author.get("name", None))
        author_orgs.append(author.get("org", None))

    return author_ids, author_names, author_orgs


def extract_keys(
    file_path: Path, N: int = 10000000, year_start: int = 2010
) -> pd.DataFrame:
    """
    Extract specific keys from a JSON file and create a DataFrame.

    Args:
        file_path: The path to the JSON file to be processed.

    Returns:
        A DataFrame containing the extracted and filtered data,
        sorted by year in ascending order.
    """
    with open(file_path, "r") as f:
        # Iterate over objects in the JSON file
        objects = ijson.items(f, "item")
        partial_data = []

        # Iterate over the objects and extract the required keys
        for _, obj in tqdm(
            zip(range(N), objects), total=N, desc="Processing JSON Data"
        ):
            filtered_obj = {k: obj[k] for k in KEYS_TO_EXTRACT if k in obj}

            # Filter out objects with year <= year_start
            if "year" in filtered_obj and filtered_obj["year"] >= year_start:
                # Extract author fields
                if "authors" in filtered_obj:
                    author_ids, author_names, author_orgs = extract_author_fields(
                        filtered_obj["authors"]
                    )
                    filtered_obj["author_ids"] = author_ids if author_ids else None
                    filtered_obj["author_names"] = (
                        author_names if author_names else None
                    )
                    filtered_obj["author_orgs"] = author_orgs if author_orgs else None
                    del filtered_obj["authors"]

                partial_data.append(filtered_obj)

    return pd.DataFrame(partial_data).sort_values(by="year", ascending=True)


def create_years_subset(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """
    Create a subset of DataFrames for each year.

    Args:
        df: The DataFrame to be split.

    Returns:
        A dictionary containing DataFrames for each year.
    """
    # get unique years
    years = df["year"].dropna().astype(int).unique()
    return {year: df[df["year"] == year] for year in years}


def get_edge_index(dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Create edge index DataFrame from the subset of DataFrames.

    Args:
        dfs: A dictionary containing DataFrames for each year.

    Returns:
        A DataFrame containing the edge index.
    """
    edge_index = []
    counter = 0

    # Iterate over the years
    for year in dfs:
        print(f"Processing year {year}")
        df = dfs[year]

        # Iterate over the rows in the DataFrame
        for idx in tqdm(range(df.shape[0]), desc="Processing Rows"):
            edge_text_list = []
            author_id = df["author_ids"].iloc[idx]

            if author_id:
                # Map document type to integer
                doc_type = 1 if df["doc_type"].iloc[idx] == "Journal" else 0

                # Get number of citations
                citations = int(df["n_citation"].iloc[idx])

                # Create a list of text fields
                edge_text_list.append(df["title"].iloc[idx])
                edge_text_list.extend(df["keywords"].iloc[idx])
                edge_text_list.append(df["abstract"].iloc[idx])

                # Concatenate the text fields
                author_text = ". ".join(edge_text_list)

                # Create edges between authors
                for j in range(len(author_id)):
                    for k in range(j + 1, len(author_id)):
                        edge_index.append(
                            (
                                author_id[j],
                                author_id[k],
                                year,
                                counter,
                                author_text,
                                doc_type,
                                citations,
                            )
                        )
                        counter += 1

    return pd.DataFrame(
        edge_index,
        columns=[
            "source",
            "target",
            "time_step",
            "index",
            "text",
            "doc_type",
            "citations",
        ],
    )


def map_and_normalize(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[int, int]]:
    """
    Map unique alphanumeric strings to integers and normalize the 'citations' column.

    Args:
        df: The DataFrame to be processed.

    Returns:
        A tuple containing the processed DataFrame, mapping of names to integers,
        and mapping of years to integers.
    """

    # Combine unique alphanumeric strings from both 'source' and 'target'
    unique_values = pd.concat([df["source"], df["target"]]).unique()

    # Create a mapping for names and years
    mapping_names = {val: idx for idx, val in enumerate(unique_values)}
    mapping_years = {val: idx for idx, val in enumerate(df["time_step"].unique())}

    # Replace the values with corresponding integers
    df["source"] = df["source"].map(mapping_names)
    df["target"] = df["target"].map(mapping_names)
    df["time_step"] = df["time_step"].map(mapping_years)

    def z_score_normalize(x, mean, std):
        return (x - mean) / std

    mean = df["citations"].mean()
    std = df["citations"].std()
    df["citations"] = df["citations"].apply(lambda x: z_score_normalize(x, mean, std))

    return df, mapping_names, mapping_years


def save_csv_chunked(
    df: pd.DataFrame, file_path: Path, chunksize: int = 100000
) -> None:
    """
    Save a DataFrame to a CSV file in chunks.

    Args:
        df: The DataFrame to be saved.
        file_path: The path to the CSV file.
        chunksize: The size of each chunk.

    Returns:
        None
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert DataFrame to PyArrow Table
    table = pa.Table.from_pandas(df)

    # Calculate total number of chunks
    total_chunks = (len(df) + chunksize - 1) // chunksize

    with csv.CSVWriter(file_path, table.schema) as writer:
        for i in tqdm(
            range(0, len(df), chunksize), total=total_chunks, desc="Saving CSV"
        ):
            chunk = table.slice(i, chunksize)
            writer.write_table(chunk)


def read_csv_chunked(file_path: Path, chunksize: int = 100000) -> pd.DataFrame:
    """
    Read a CSV file in chunks and return a single DataFrame.

    Args:
        file_path: The path to the CSV file.
        chunksize: The size of each chunk.

    Returns:
        A single DataFrame containing the data from the CSV file
    """

    # Get total number of rows in the CSV file
    total_rows = sum(1 for _ in open(file_path)) - 1

    chunks = []
    with tqdm(total=total_rows, desc="Reading CSV") as pbar:
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            chunks.append(chunk)
            pbar.update(len(chunk))

    return pd.concat(chunks, ignore_index=True)


def get_text_embeddings(
    df: pd.DataFrame, model: SentenceTransformer, device: torch.device
) -> np.ndarray:
    """
    Generate text embeddings for the 'text' column in a DataFrame using a Sentence-BERT model.

    Args:
        df: The DataFrame containing the text data.
        model: The Sentence-BERT model to be used for encoding.
        device: The device to move the model to.

    Returns:
        A numpy array containing the text embeddings.
    """
    # Lists to store embeddings for each row
    text_embeddings = []
    num_rows = len(df)

    # Loop through each row in the DataFrame
    for text in tqdm(df["text"], total=num_rows, desc="Generating embeddings"):
        # Encode the text and organization using the Sentence-BERT model
        text_embeddings.append(model.encode(text, device=device))

    return np.array(text_embeddings)


def dim_reduction(
    X: np.ndarray, n_components: int, seed: int = 123, chunk_size: int = 100000
) -> np.ndarray:
    """
    Apply dimensionality reduction using TruncatedSVD.

    Args:
        X: The input data.
        n_components: The number of components to keep.
        seed: The random seed for reproducibility.
        chunk_size: The size of each chunk.

    Returns:
        The reduced data.
    """
    reducer = TruncatedSVD(n_components=n_components, random_state=seed)
    num_chunks = X.shape[0] // chunk_size + (1 if X.shape[0] % chunk_size != 0 else 0)

    X_reduced = np.empty((X.shape[0], n_components), dtype=X.dtype)
    for i in tqdm(range(num_chunks), desc="Reducing dimensionality"):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, X.shape[0])
        chunk = X[start:end]
        if i == 0:
            reducer.fit(chunk)
        X_reduced[start:end] = reducer.transform(chunk)

    return X_reduced
