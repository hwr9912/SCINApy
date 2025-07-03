"""
SCINA: A Semi-Supervised Category Identification and Assignment Tool.

This package provides an automatic cell type detection and assignment algorithm
for single-cell RNA-Seq (scRNA-seq) and Cytof/FACS data. It uses prior knowledge
of signature genes to assign cell type identities.

Main functions:
- SCINA: Core algorithm for cell type assignment.
- plotheat_scina: Visualize results with a heatmap.
- preprocess_signatures: Load and preprocess signature genes from CSV files.
- run_scina_cli: Command-line interface for running SCINA.
- load_sample_data: Load sample data from the data/ directory (CSV format).
"""

__version__ = "0.1.0"

import json
import pandas as pd
import anndata as ad
from os.path import exists
from scipy.sparse import csr_matrix

from .core import SCINA
from .visual import plotheat_scina
from .utils import preprocess_signatures
from .interface import run_scina_cli


def load_sample_data():
    """
    Load sample single-cell data from the data/ directory (CSV format).

    :param genes: Optional list of gene names to filter the data.
    :param file_name: Name of the CSV file in data/ (default: "expmat.csv").
    :return: AnnData object containing the filtered data.
    """
    scdata_path = "data/matrix.csv"
    sigdata_path = "data/signatures.json"
    if not exists(sigdata_path):
        raise FileNotFoundError(f"Sample data file {sigdata_path} not found.")
    if not exists(scdata_path):
        raise FileNotFoundError(f"Sample data file {scdata_path} not found.")
    
    # 读取 CSV，假设第一列是基因名，第一行是细胞名，数据为基因×细胞
    exp = pd.read_csv(scdata_path, index_col=0)
    adata = ad.AnnData(X=csr_matrix(exp.T))  # 转置为细胞×基因
    adata.var_names = exp.index
    adata.obs_names = exp.columns

    # 读取 JSON，每一列第一行是细胞名，其下每一行都是marker基因名
    with open(sigdata_path, "r") as json_file:
        sig = json.load(json_file)

    return adata, sig

__all__ = [
    "SCINA",
    "plotheat_scina",
    "preprocess_signatures",
    "run_scina_cli",
    "load_sample_data",
]