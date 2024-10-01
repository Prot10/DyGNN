# Dynamic Graph Neural Network for Link Prediction

This repository uses the library proposed in the paper [Towards Better Dynamic Graph Learning: New Architecture and Unified Library](https://arxiv.org/abs/2303.13047) to train a Dynamic GNN for link prediction on a citations networks to predict collaborations between authors. 

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)

## Dataset

The AMiner Citation Network Dataset is a comprehensive collection of academic papers and their citation relationships. This dataset is widely used for various research tasks in the fields of data mining, network analysis, and scientometrics.

- **Version**: 14
- **Source**: [AMiner](https://www.aminer.cn/citation)
- **Contents**: Academic papers with metadata (title, authors, year, venue, etc.) and citation relationships
- **Size**: Large-scale dataset (exact size may vary)

## Installation

### Clone the Repository

To get started, clone the repository:

```bash
git clone https://github.com/Prot10/DyGNN.git
```

Navigate to the repository directory:

```bash
cd DyGNN
```

### Set Up the Environment

Create a conda environment with Python 3.11 and install the required packages:

```bash
conda create -n dygnn python=3.11
conda activate dygnn
pip install -e .
```

### Downloading the Dataset

You can download and unzip the dataset inside the data folder using one of the following methods:

#### Raw Dataset

To download the raw dataset:

```bash
mkdir -p dygnn/dataset/raw_data && wget -c https://originalfileserver.aminer.cn/misc/dblp_v14.tar.gz -O - | tar -xz -C dygnn/dataset/raw_data
```

#### Preprocessed Data

Alternatively, you can download the preprocessed data:

```bash
mkdir -p dygnn/DyGLib/processed_data && \
gdown --fuzzy https://drive.google.com/uc?id=17BFRcP_wOwRwsCfxbUbIyZobX2ji2K3b -O dygnn/DyGLib/processed_data/citations.tar.gz && \
tar -xvf dygnn/DyGLib/processed_data/citations.tar.gz -C dygnn/DyGLib/processed_data
```

## Usage

You can run the full pipeline to preprocess the data using different parameters from the default one:

```bash
bash run_data_pipeline.sh
```

To have information about the parameters that you can change use the `--help` flag.

