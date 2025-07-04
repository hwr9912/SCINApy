import json
import pandas as pd
import anndata as ad
from os.path import exists, join
from scipy.sparse import csr_matrix


def preprocess_signatures(file_path):
    """
    Convert signatures from a CSV file to a list of signature gene dictionary for SCINA.
    
    :param file_path: Path to the CSV file where each column represents a cell type (header) and 
        contains its signature genes. The file has no index column.
    
    :return: A dictionary mapping each cell type (key) to a list of its signature genes (value).
    """
    # Read CSV file
    csv_signatures = pd.read_csv(file_path, header=0, dtype=str, keep_default_na=False)
    
    # 使用字典推导式，过滤掉空字符串和 NaN
    signature_dict = {
        cell_type: [gene for gene in csv_signatures[cell_type].dropna().tolist() if gene.strip()]
        for cell_type in csv_signatures.columns
    }

    return signature_dict


def load_sample_data():
    """
    Load sample single-cell data from the data/ directory (CSV format).

    :param genes: Optional list of gene names to filter the data.
    :param file_name: Name of the CSV file in data/ (default: "expmat.csv").
    :return: AnnData object containing the filtered data.
    """
    scdata_path = join("data", "matrix.csv")
    sigdata_path = join("data", "signatures.json")
    true_label_path = join("data", "true_label.csv")
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
    
    # 读取 CSV，假设第一列是基因名，第一行是细胞名，数据为基因×细胞
    df = pd.read_csv(true_label_path)
    adata.obs["true_label"] = df['vector'].to_list()

    return adata, sig