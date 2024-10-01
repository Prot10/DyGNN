"""Module to create the edge index from the dataset."""

from pathlib import Path

import numpy as np
import torch
from dataset.utils import dim_reduction, get_text_embeddings, read_csv_chunked
from sentence_transformers import SentenceTransformer

FILE_PATH = Path("dygnn/dataset/processed_data") / "citation.csv"
OUT_FOLDER = Path("dygnn/DyGLib/processed_data/citations")
N_COMPONENTS = 170


def main():
    # Open the edge index file
    print(f"Reading the edge index from {FILE_PATH}...")
    assert FILE_PATH.exists(), f"{FILE_PATH} does not exist!"
    df = read_csv_chunked(file_path=FILE_PATH)
    print("Done!")
    print(f"Dataframe shape: {df.shape}")
    print(f"Dataframe columns: {df.columns}\n")

    # Get the embedding for the text
    print("Getting the embedding for the text...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load pretrained Sentence-BERT model and move it to the device
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    sbert_model.to(device)
    embeddings = get_text_embeddings(df=df, model=sbert_model, device=device)
    print("Done!\n")

    # Apply the TruncatedSVD to the embeddings
    print("Performing dimensionality reduction...")
    embeddings = dim_reduction(X=embeddings, n_components=N_COMPONENTS)
    print("Done!")
    print(f"Embeddings shape: {embeddings.shape}\n")

    # Create the final edge features matrix with the embeddings, doc_type and citations
    print(
        "Creating the final edge features matrix with the text embeddings, doc_type and citations..."
    )
    doc_type = df["doc_type"].to_numpy().reshape(-1, 1)
    citations = df["citations"].to_numpy().reshape(-1, 1)
    ml_citations = np.concatenate((embeddings, doc_type, citations), axis=1)
    OUT_FOLDER.mkdir(parents=True, exist_ok=True)
    np.save(OUT_FOLDER / "ml_citations.npy", ml_citations)
    print("Done!")
    print(f"Edge features shape: {ml_citations.shape}\n")
    print(f"Edge features saved in: {OUT_FOLDER / 'ml_citations.npy'}\n")


if __name__ == "__main__":
    main()
