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

This project has three main steps: data preprocessing, model training, and evaluation. You can run all steps or skip some if you have preprocessed data or pre-trained weights.

### 1. Data Preprocessing
- **Script**: `run_data_preprocessing.sh`
- **Description**: Processes raw data and generates features.
  
```bash
bash run_data_preprocessing.sh
```

### 2. Model Training
- **Script**: `run_train_link_prediction.sh`
- **Description**: Trains the model on preprocessed data.

```bash
bash run_train_citations.sh
```

### 3. Model Evaluation
- **Script**: `run_evaluate_link_prediction.sh`
- **Description**: Evaluates the model's performance.

```bash
bash run_evaluate_citations.sh
```

### Customizing Parameters

Each script can be customized by passing arguments. Use the `--help` flag for more information on available options:

```bash 
bash run_data_preprocessing.sh --help
bash run_train_link_prediction.sh --help
bash run_evaluate_link_prediction.sh --help
```

