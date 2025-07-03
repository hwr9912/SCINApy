import numpy as np
import pandas as pd
from scipy.sparse import issparse

def check_inputs(exp, gene_names, signatures, max_iter, convergence_n, convergence_rate, sensitivity_cutoff, rm_overlap, log_file):
    """
    检查 SCINA 算法的输入参数格式、完整性以及值是否在设计范围内。
    这是一个内部函数，由 SCINA 调用，用于验证输入并根据需要清理参数。

    参数:
    exp -- numpy 数组或稀疏矩阵，表示目标数据集的归一化表达矩阵。
           行对应属性（例如基因符号），列对应对象（例如细胞条形码）。
    gene_names -- 基因名称列表，与 exp 的行对应。
    signatures -- 字典，包含多个签名基因列表，每个列表代表一个细胞类型。
    max_iter -- 整数 > 0，默认 100，表示 EM 算法的最大迭代次数。
    convergence_n -- 整数 > 0，默认 10，表示收敛性检查的最近 n 轮迭代。
    convergence_rate -- 浮点数 (0-1)，默认 0.99，表示最近 n 轮中保持稳定的细胞百分比。
    sensitivity_cutoff -- 浮点数 (0-1)，默认 1，表示移除未使用的签名的截止值。
    rm_overlap -- 二元值 (0 或 1)，默认 1 (True)，表示是否移除不同类型细胞基因签名列表之间的共享基因。
    log_file -- 字符串，表示日志文件的名称，默认 'SCINA.log'。

    返回:
    dict -- 包含以下键的字典：
            qual -- 二元值 (0 或 1)，0 表示输入不合格，1 表示输入合格。
            sig -- 处理后的 signatures 字典。
            para -- 列表，包含 [max_iter, convergence_n, convergence_rate, sensitivity_cutoff]。
            msg -- 字符串，描述检查结果或问题。
    """
    # 初始化质量标志和默认参数
    quality = 1  # 初始假设输入合格
    def_max_iter = 1000  # 默认最大迭代次数
    def_conv_n = 10  # 默认收敛性检查轮数
    def_conv_rate = 0.99  # 默认收敛率
    def_dummycut = 0.33  # 默认敏感性截止值
    all_genes = gene_names  # 所有基因名称，从输入的 gene_names 获取

    # 检查表达矩阵中是否存在 NA (NaN)
    if issparse(exp):
        nan_check = np.isnan(exp.toarray()).any()
    else:
        nan_check = np.isnan(exp).any()
    if nan_check:
        with open(log_file, 'a') as f:
            print('NA exists in expression matrix.', file=f)
        quality = 0  # 如果存在 NaN，质量标记为不合格

    # 检查 signatures 是否包含 NA
    if any(isinstance(v, (list, np.ndarray)) and any(pd.isna(g) for g in v) for v in signatures.values()):
        with open(log_file, 'a') as f:
            print('Null cell type signature genes.', file=f)
        quality = 0  # 如果签名中包含 NA，质量标记为不合格
    else:
        # 清理 signatures：保留非 NA 且在 all_genes 中的唯一基因
        cleaned_signatures = {}
        for key, genes in signatures.items():
            valid_genes = [g for g in genes if not pd.isna(g) and g in all_genes]
            cleaned_signatures[key] = list(set(valid_genes))  # 去重
        signatures = cleaned_signatures

        # 如果 rm_overlap 为 1 (True)，移除签名列表之间的共享基因
        if rm_overlap:
            gene_counts = pd.Series([g for v in signatures.values() for g in v]).value_counts()
            unique_genes = gene_counts[gene_counts == 1].index.tolist()  # 只保留出现一次的基因
            signatures = {k: [g for g in v if g in unique_genes] for k, v in signatures.items()}

        # 检查是否有基因表达全为 0 的情况，移除方差为 0 的基因
        if not issparse(exp):
            exp_array = exp
        else:
            exp_array = exp.toarray()
        # 修正：exp_array 是 (n_cells, n_genes)，转置以匹配基因为行
        exp_df = pd.DataFrame(exp_array.T, index=all_genes, columns=range(exp.shape[0]))
        for key in signatures:
            signatures[key] = [g for g in signatures[key] if g in all_genes and exp_df.loc[g].std() > 0]

    # 清理其他参数，处理 NA 或超出范围的情况
    if pd.isna(convergence_n):
        with open(log_file, 'a') as f:
            print('Using convergence_n=default', file=f)
        convergence_n = def_conv_n
    if pd.isna(max_iter):
        with open(log_file, 'a') as f:
            print('Using max_iter=default', file=f)
        max_iter = def_max_iter
    else:
        if max_iter < convergence_n:
            with open(log_file, 'a') as f:
                print('Using max_iter=default due to smaller than convergence_n.', file=f)
            max_iter = convergence_n
    if pd.isna(convergence_rate):
        with open(log_file, 'a') as f:
            print('Using convergence_rate=default.', file=f)
        convergence_rate = def_conv_rate
    if pd.isna(sensitivity_cutoff):
        with open(log_file, 'a') as f:
            print('Using sensitivity_cutoff=default.', file=f)
        sensitivity_cutoff = def_dummycut

    # 返回清理后的参数
    return {
        'qual': quality,
        'sig': signatures,
        'para': [max_iter, convergence_n, convergence_rate, sensitivity_cutoff],
        'msg': "Inputs checked successfully." if quality else "Inputs contain issues."
    }


def density_ratio(e, mu1, mu2, inverse_sigma1, inverse_sigma2):
    """
    计算每个样本点表达向量 e 在两个高斯模型下的概率密度比：
    ratio = pdf_high / pdf_low
    
    参数：
    - e: numpy array，形状 (n_cells, n_genes)
    - mu1, mu2: numpy array，形状 (n_genes,)
    - inverse_sigma1, inverse_sigma2: numpy array，形状 (n_genes, n_genes)，协方差矩阵的逆
    
    返回：
    - ratio: numpy array，形状 (n_cells,)
    """
    print(e.shape, mu1.shape, mu2.shape, inverse_sigma1.shape, inverse_sigma2.shape)
    diff1 = (e - mu1).T  # (n_genes, n_cells)
    diff2 = (e - mu2).T

    # 马氏距离计算
    tmp1 = np.einsum('ij,jk,ik->k', diff1, inverse_sigma1, diff1)  # shape (n_cells,)
    tmp2 = np.einsum('ij,jk,ik->k', diff2, inverse_sigma2, diff2)

    # 计算行列式的log，注意用 np.linalg.slogdet 提高数值稳定性
    sign1, logdet1 = np.linalg.slogdet(np.linalg.inv(inverse_sigma1))  # 协方差矩阵
    sign2, logdet2 = np.linalg.slogdet(np.linalg.inv(inverse_sigma2))

    # 计算指数部分，参照 R 代码表达式
    log_ratio = -0.5 * (tmp1 + logdet1 - tmp2 - logdet2)
    ratio = np.exp(log_ratio)

    # 数值限制
    ratio = np.clip(ratio, 1e-200, 1e200)

    return ratio


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

# 读取 CSV 文件并转换为字典
def read_markers_to_dict(csv_path):
    """读取 CSV 文件，将每一列的第一行作为细胞名，其下每一行作为 marker 基因名，转换为字典。"""
    df = pd.read_csv(csv_path, index_col=0)  # index_col=0 使用第一列作为索引
    cell_names = df.columns.tolist()
    marker_genes = df.index.tolist()
    result_dict = {cell: [gene for gene in marker_genes if pd.notna(gene) and gene != ""] for cell in cell_names}
    return result_dict