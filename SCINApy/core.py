import re
import numpy as np
import pandas as pd
import scipy as scp
import  anndata as ad

"""
SCINA: A Semi-Supervised Category Identification and Assignment Tool.

Description:
An automatic cell type detection and assignment algorithm for single cell RNA-Seq (scRNA-seq) and Cytof/FACS data.
SCINA assigns cell type identities to a pool of cells profiled by scRNA-Seq or Cytof/FACS data using prior knowledge of identifiers,
such as genes or protein symbols, that are highly or lowly expressed in each category.
See Ze Z, Danni L, et al (2018) for more details.

Dependencies:
- numpy: For matrix operations and numerical computations.
- pandas: For handling data frames and expression matrices.
- anndata: For handling single-cell data in AnnData format.
- scipy: For multivariate normal distribution calculations.
- re: For regular expression operations on gene names.
"""
def SCINA(adata, signatures, max_iter=100, convergence_n=10, convergence_rate=0.99,
          sensitivity_cutoff=1, allow_unknown=True, log_file='SCINA.log', inplace=False):
    """
    :param adata: AnnData object containing the expression data, where `.X` is the expression matrix and `.var_names` are gene names.
    :param signatures: Dictionary with string keys (cell types) and values as lists of strings (signature genes, no empty or NaN values).
    :param max_iter: Integer > 0, default 100. Maximum iterations allowed for the EM algorithm.
    :param convergence_n: Integer > 0, default 10. Stops the EM algorithm if cell type assignments remain stable for the last n iterations.
    :param convergence_rate: Float between 0 and 1, default 0.99. Percentage of cells with stable type assignments over the last n iterations.
    :param sensitivity_cutoff: Float between 0 and 1, default 1. Cutoff to remove signatures whose cell types are deemed non-existent in the data.
    :param allow_unknown: Boolean, default True. If True, allows cells to be assigned to an 'unknown' category.
    :param log_file: String, default 'SCINA.log'. Path to the file recording the running status of the SCINA algorithm.
    :param inplace: Boolean, default False. If True, modifies the input adata object in-place and returns None; if False, returns a new AnnData object with results without modifying the input.
    :return: None if inplace=True; a new AnnData object with 'scina_labels' in .obs and 'probabilities' in .obsm if inplace=False.
    """
    # 构建日志文件
    with open(log_file, 'w') as f:
        print('Start running SCINA.', file=f)
    # 构建状态日志文件
    status_file = f'{log_file}.status'
    # 统计所有标记基因
    all_sig = list(set([item for sublist in signatures.values() for item in sublist]))
    # 统计所有特征性低表达标记基因
    invert_sigs = [sig for sig in all_sig if re.match(r'^low_', sig)]
    # 提取表达量矩阵,并转换为稠密矩阵
    expr = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)

    if invert_sigs:
        with open(log_file, 'a') as f:
            print('Converting expression matrix for low_genes.', file=f)
        # 去除特征低表达基因的字符串头"low_"
        invert_sigs = [sig for sig in invert_sigs if re.sub(r'^low_', '', sig) in adata.var_names]
        # 筛选高变的特征低表达基因
        invert_hv_sigs = list(set(invert_sigs) & set(adata.var_names))
        # 对低表达基因的表达量取负
        expr[:, adata.var_names.isin(invert_hv_sigs)] *= -1

        del invert_sigs, invert_hv_sigs

    # 输入检查
    quality = check_inputs(expr, adata.var_names, signatures, max_iter, convergence_n, convergence_rate,
                           sensitivity_cutoff, log_file)

    if not quality['qual']:
        with open(log_file, 'a') as f:
            print('EXITING due to invalid parameters.', file=f)
        with open(status_file, 'w') as f:
            print('0', file=f)
        raise ValueError('SCINA stopped.')
    # 加载检查后的参数
    signatures = quality['sig']
    max_iter, convergence_n, convergence_rate, sensitivity_cutoff = quality['para']
    # expr:pd.Dataframe, n genes * n cells
    expr = expr.loc[:, [gene for sig in signatures.values() for gene in sig]].T
    # labels:np.ndarray n cells * n 收敛前iters
    labels = np.zeros((adata.shape[0], convergence_n))
    unsatisfied = True

    # 三类别概率tao：(n cell_type,)
    if allow_unknown:
        tao = np.full(len(signatures), 1 / (len(signatures) + 1))
    else:
        tao = np.full(len(signatures), 1 / len(signatures))

    theta = []
    for _, markers in signatures.items():
        # expr:pd.Dataframe, n marker_genes * n cells
        expr = adata[:, markers].X.toarray().T
        # mean_high & mean_low: (n marker_genes,)
        mean_high = np.percentile(expr, 70, axis=1)
        mean_low = np.percentile(expr, 30, axis=1)
        variances = np.var(expr, axis=1)
        # 对角矩阵：左上至右下
        sigma = np.diag(variances)

        # theta：list n cell_type
        # theta[i]['mean'] (n celltype, 2)
        # theta[i]['sigma1'] (n celltype, n celltype)
        # theta[i]['sigma2'] (n celltype, n celltype)
        theta.append({
            'mean': np.vstack([mean_high, mean_low]).T,
            'sigma1': sigma.copy(),
            'sigma2': sigma.copy()
        })

    # cran的版本里没有这一段
    # def is_empty(matrix):
    #     return np.all(np.isnan(matrix) | (matrix == 0))
    # theta = [t for t in theta if not is_empty(t['sigma1'])]

    sigma_min = min(min(sig['sigma1'].diagonal().min(), sig['sigma2'].diagonal().min()) for sig in theta) / 100

    while unsatisfied:
        # prob_mat: np.array n marker_genes * n cells
        prob_mat = np.tile(tao[:, np.newaxis], (1, expr.shape[1]))
        iter = 0
        labels_i = 0

        while iter < max_iter:
            iter += 1

            # E step: estimate variables.
            for i in range(len(signatures)):
                # Python 计算 chol2inv 等效操作
                chol_matrix = scp.linalg.cholesky(theta[i]['sigma1'], lower=False)  # Cholesky 分解，上三角
                inv_chol = scp.linalg.inv(chol_matrix)  # 求 Cholesky 矩阵的逆
                # 对 sigma1 进行 Cholesky 分解，得到下三角因子 L, 之后计算逆矩阵
                theta[i]['inverse_sigma1'] = inv_chol @ inv_chol.T
                theta[i]['inverse_sigma2'] = theta[i]['inverse_sigma1']

            for r, (cell_type, markers) in enumerate(signatures.items()):
                prob_mat[r, :] = tao[r] * density_ratio(
                    e = adata[:, markers].X.toarray().T,
                    mu1 = theta[r]['mean'][:, 0],
                    mu2 = theta[r]['mean'][:, 1],
                    inverse_sigma1 = theta[r]['inverse_sigma1'],
                    inverse_sigma2 = theta[r]['inverse_sigma2']
                )

            prob_mat = prob_mat / (1 - sum(tao) + prob_mat.sum(axis=0))

            # M step: update sample distributions.
            tao = prob_mat.mean(axis=1)

            for i, (cell_type, markers) in enumerate(signatures.items()):
                expr = adata[:, markers].X.toarray().T
                mean_high = (expr @ prob_mat[i]) / prob_mat[i].sum()
                mean_low = (expr @ (1 - prob_mat[i])) / (1 - prob_mat[i]).sum()

                keep = mean_high <= mean_low
                if keep.any():
                    mean_high[keep] = expr[keep].mean(axis=1)
                    mean_low[keep] = mean_high[keep]
                theta[i]['mean'] = np.vstack([mean_high, mean_low]).T

                tmp1 = ((expr - mean_high[:, np.newaxis]) ** 2).T
                tmp2 = ((expr - mean_low[:, np.newaxis]) ** 2).T

                prob_row_r = prob_mat[i][:, np.newaxis]
                weighted_sum_of_squares = (tmp1 * prob_row_r + tmp2 * (1 - prob_row_r))
                updated_variances = np.sum(weighted_sum_of_squares, axis=0) / tmp1.shape[0]
                # tmp1.shape[0] 就是 n_cells, 取协方差

                # 使用 np.fill_diagonal 更新整个对角线，并应用最小值限制
                # R: diag(theta[[i]]$sigma1)[diag(theta[[i]]$sigma1)<sigma_min]=sigma_min
                np.fill_diagonal(theta[i]['sigma1'], np.clip(updated_variances, sigma_min, None))
                np.fill_diagonal(theta[i]['sigma2'], np.clip(updated_variances, sigma_min, None))

            labels[:, labels_i] = np.argmax(np.vstack((1 - prob_mat.sum(axis=0), prob_mat)), axis=0) - 1

            # Compare estimations with stop rules.
            if np.mean([len(set(labels[x])) == 1 for x in range(labels.shape[0])]) >= convergence_rate:
                with open(log_file, 'a') as f:
                    print('Job finished successfully.', file=f)
                with open(status_file, 'w') as f:
                    print('1', file=f)
                break

            labels_i = (labels_i + 1) % convergence_n
            if iter == max_iter:
                with open(log_file, 'a') as f:
                    print('Maximum iterations, breaking out.', file=f)

        dummytest = np.array([np.mean(theta[i]['mean'][:, 0] - theta[i]['mean'][:, 1] == 0) for i in range(len(signatures))])
        if np.all(dummytest <= sensitivity_cutoff):
            unsatisfied = False
        else:
            rev = np.where(dummytest > sensitivity_cutoff)[0]
            with open(log_file, 'a') as f:
                print(f'Remove dummy signatures: {rev}', file=f)
            signatures = [mrk for i, (ct, mrk) in enumerate(signatures.items()) if i not in rev]
            tao = tao[np.isin(np.arange(len(tao)), rev, invert=True)]
            tao = tao / (1 - sum(tao) + sum(tao))
            theta = [theta[i] for i in range(len(theta)) if i not in rev]

    prob_mat = pd.DataFrame(prob_mat, index=list(signatures), columns=adata.obs_names)
    labels = pd.DataFrame(labels, index=adata.obs_names)

    cell_labels = ['unknown'] + list(signatures)
    # 加入了unknown，标签要后移+1
    final_labels = [cell_labels[int(l) + 1] for l in labels.iloc[:, -1]]

    if inplace:
        # 修改原 adata
        adata.obs['scina_labels'] = final_labels
        adata.obsm['probabilities'] = prob_mat.T
        with open(log_file, 'a') as f:
            print('Results stored in adata object in-place.', file=f)
        return None
    else:
        # 创建新 adata 对象，不修改原 adata
        new_adata = ad.AnnData(X=adata.X.copy(), obs=adata.obs.copy(), var=adata.var.copy())
        new_adata.obs['scina_labels'] = final_labels
        new_adata.obsm['probabilities'] = prob_mat.T
        return new_adata

def check_inputs(exp, allgenes, signatures, max_iter, convergence_n, convergence_rate,
                 sensitivity_cutoff, log_file='SCINA.log'):
    """
    检查下列参数是否不含非空值
    :param exp:
    :param allgenes:
    :param signatures:
    :param max_iter:
    :param convergence_n:
    :param convergence_rate:
    :param sensitivity_cutoff:
    :param log_file:
    :return:
    """
    quality = 1

    # 检查表达矩阵中是否有空值
    if np.isnan(exp).sum().any():
        with open(log_file, 'a') as f:
            print('NA exists in expression matrix.', file=f)
        quality = 0

    # 检查signature中是否包含空值
    if any(not gene for sig in signatures.values() for gene in sig):
        with open(log_file, 'a') as f:
            print('Null cell type signature genes.', file=f)
        quality = 0
    else:
        # 去除基因集中的空值和数据集中没有的基因
        signatures = {k: [g for g in v if g in allgenes] for k, v in signatures.items()}

        # 去除全0基因
        std_devs = exp.std(axis=0)
        signatures = {k: [g for g in v if std_devs[g] > 0] for k, v in signatures.items()}

        # 删除值为空列表的键
        signatures = {k: v for k, v in signatures.items() if v != []}

    # 检查其他参数
    if pd.isna(convergence_n):
        with open(log_file, 'a') as f:
            print('Using convergence_n=10 by default', file=f)
        convergence_n = 10

    if pd.isna(max_iter):
        with open(log_file, 'a') as f:
            print('Using max_iter=1000 by default', file=f)
        max_iter = 1000
    elif max_iter < convergence_n:
        with open(log_file, 'a') as f:
            print('Using max_iter=convergence_n by default due to smaller than convergence_n.', file=f)
        max_iter = convergence_n

    if pd.isna(convergence_rate):
        with open(log_file, 'a') as f:
            print('Using convergence_rate=0.99 by default.', file=f)
        convergence_rate = 0.99

    if pd.isna(sensitivity_cutoff):
        with open(log_file, 'a') as f:
            print('Using sensitivity_cutoff=0.33 by default.', file=f)
        sensitivity_cutoff = 0.33

    return {
        'qual': quality,
        'sig': signatures,
        'para': [max_iter, convergence_n, convergence_rate, sensitivity_cutoff]
    }


def density_ratio(e, mu1, mu2, inverse_sigma1, inverse_sigma2):
    """
    计算两个多元正态分布的密度比（对数尺度上）ratio = pdf_high / pdf_low

    参数:
    e (np.ndarray): 数据点矩阵，形状 (n_genes, n_cells)。
                    如果只有一个样本，也可以是 (n_genes,)。
    mu1 (np.ndarray): 第一个分布的均值向量，形状 (n_genes,)。
    mu2 (np.ndarray): 第二个分布的均值向量，形状 (n_genes,)。
    inverse_sigma1 (np.ndarray): 第一个分布的逆协方差矩阵，形状 (n_genes, n_genes)。
    inverse_sigma2 (np.ndarray): 第二个分布的逆协方差矩阵，形状 (n_genes, n_genes)。

    返回:
    np.ndarray: 密度比，形状与 e 的列数相同 (n_cells,)。
    """

    # 确保 e 是一个二维数组，即使只有一个样本
    # R 的 colSums 行为通常意味着 e 的每一列是一个样本
    # 在 NumPy 中，数据通常是 (n_cells, n_genes) 或 (n_genes, n_cells)
    # 这里假设 e 的形状是 (n_genes, n_cells)，与R的矩阵列向操作习惯一致
    # 如果 e 是 (n_cells, n_genes)，需要相应调整
    if e.ndim == 1:
        e_reshaped = e[:, np.newaxis] # 转换为 (n_genes, 1)
    else:
        e_reshaped = e

    # 计算 (e - mu)
    # 对于多列（多个样本），这里需要广播或循环
    # R 的 (e-mu1) 会自动处理列广播
    # NumPy 需要更显式的处理
    diff1 = e_reshaped - mu1[:, np.newaxis] # 使得 mu1 可以广播到 e 的所有列
    diff2 = e_reshaped - mu2[:, np.newaxis]

    tmp1 = np.sum(diff1 * (inverse_sigma1 @ diff1), axis=0)
    tmp2 = np.sum(diff2 * (inverse_sigma2 @ diff2), axis=0)

    log_det_inv_sigma1 = np.linalg.slogdet(inverse_sigma1)[1] # 返回 (sign, logdet)
    log_det_inv_sigma2 = np.linalg.slogdet(inverse_sigma2)[1]

    exponent_term = -0.5 * (tmp1 - log_det_inv_sigma1 - tmp2 + log_det_inv_sigma2)
    tmp = np.exp(exponent_term, dtype=np.float64)

    # 边界条件处理
    ratio = np.clip(tmp, 1e-200, 1e200)

    return ratio

# if __name__ == "__main__":
#     import json
#     import pandas as pd
#     import anndata as ad
#     import scanpy as sc
#     # scdata_path = "data/matrix.csv"
#     # sigdata_path = "data/signatures.json"
#     # # 读取 CSV，假设第一列是基因名，第一行是细胞名，数据为基因×细胞
#     # exp = pd.read_csv(scdata_path, index_col=0)
#     # adata = ad.AnnData(X=csr_matrix(exp.T))  # 转置为细胞×基因
#     # adata.var_names = exp.index
#     # adata.obs_names = exp.columns

#     scdata_path = "/Volumes/MacPassport/project/bioinfo/GSE230692/data/GSE276202_annotated.h5ad"
#     sigdata_path = "/Volumes/MacPassport/project/bioinfo/GSE230692/data/small_marker_dict.json"
#     adata = sc.read(scdata_path)

#     # 读取 JSON，每一列第一行是细胞名，其下每一行都是marker基因名
#     with open(sigdata_path, "r") as json_file:
#         sig = json.load(json_file)
    
#     # # 构建 DataFrame（行名是细胞，列名是基因）
#     # expr_df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

#     # # 把 obs 中的 cell_type 添加到表达矩阵中
#     # expr_df[group_key] = adata.obs[group_key].values

#     # # 按组取均值，得到 group × genes 的矩阵
#     # mean_expr = expr_df.groupby(group_key).mean()

#     SCINA(adata=adata, signatures=sig, inplace=True)
#     print(adata.obs)