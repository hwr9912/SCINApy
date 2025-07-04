import numpy as np
import pandas as pd
from scipy.sparse import issparse




def preprocess_signatures(file_path):
    """
    Convert signatures from a CSV file to a list of signature gene lists for SCINA.
    
    :param file_path: Path to the CSV file containing signature genes.

    :return: List of signature gene lists.
    """
    # Read CSV file
    csv_signatures = pd.read_csv(file_path, header=0, dtype=str, keep_default_na=False)
    
    # Convert to list of lists, removing NA and empty strings
    signatures = [col.tolist() for _, col in csv_signatures.items()]
    signatures = [[gene for gene in sig if gene and pd.notna(gene)] for sig in signatures]
    
    return signatures

# # 读取 CSV 文件并转换为字典，函数有问题   
# def read_markers_to_dict(csv_path):
#     """读取 CSV 文件，将每一列的第一行作为细胞名，其下每一行作为 marker 基因名，转换为字典。"""
#     df = pd.read_csv(csv_path)
#     cell_names = df.columns.tolist()
#     marker_genes = df.index.tolist()
#     result_dict = {cell: [gene for gene in marker_genes if pd.notna(gene) and gene != ""] for cell in cell_names}
#     return result_dict