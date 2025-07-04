import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import seaborn as sns

def plotheat_scina(adata, signatures, figsize=(15, 10), legend_bbox_to_anchor=(1.4, 1), xlabel="Cells", ylabel="Genes", save_path=None):
    """
    绘制 SCINA 结果的热图，包含签名基因和细胞标签。
    
    :param adata: AnnData 对象，包含表达数据 (.X)、细胞标签 (.obs['scina_labels']) 和概率 (.obsm['probabilities'])。
    :param signatures: 包含每个细胞类型的签名基因的字典。
    :param figsize: 元组，指定图形大小 (宽度, 高度)，默认 (10, 10) 英寸。
    :param xlabel: 字符串，设置 x 轴标签，默认 "Cells"。
    :param ylabel: 字符串，设置 y 轴标签，默认 "Genes"。
    :param save_path: 字符串，指定保存路径，若为 None 则显示图形，否则保存为文件，默认 None。
    :return: None，显示或保存热图。
    """
    # 如果 adata.X 是稀疏矩阵，则转换为数组，否则直接使用
    if hasattr(adata.X, 'toarray'):
        exp = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
    else:
        exp = pd.DataFrame(adata.X, index=adata.var_names, columns=adata.obs_names)

    # 提取细胞标签和概率矩阵
    cell_labels = adata.obs['scina_labels'].tolist()
    # prob_mat = adata.obsm['probabilities'].T  # 确保概率矩阵方向正确

    # # 移除不存在的签名基因
    # allgenes = adata.var_names.tolist()

    # 构建侧边颜色条
    n_signatures = len(signatures)  # 签名数量
    col_row = plt.cm.tab10(np.linspace(0, 1, n_signatures))  # 生成签名颜色的渐变色带，使用 tab10 颜色映射
    unique_labels = list(dict.fromkeys(cell_labels))  # 去重细胞标签，保持原始顺序
    if 'unknown' not in unique_labels:
        unique_labels.append('unknown')
    n_labels = len(unique_labels)  # 唯一标签数量
    col_col = plt.cm.Pastel1(np.linspace(0, 1, n_labels))  # 生成细胞标签颜色的渐变色带，使用 Pastel1 颜色映射

    # 构建热图矩阵
    signature_genes = [gene for _, sig in signatures.items() for gene in sig]  # 展平签名基因列表
    sorted_labels = sorted(cell_labels, key=lambda x: unique_labels.index(x))  # 按唯一标签排序的标签列表
    cell_order = [i for i, _ in sorted(enumerate(cell_labels), key=lambda x: unique_labels.index(x[1]))]  # 按标签排序的细胞索引
    exp2plot = exp.T.loc[signature_genes, exp.index[cell_order]]  # 转置并选择签名基因和排序后的细胞数据

    # 侧边颜色
    col_colside = [col_col[unique_labels.index(label)] for label in sorted_labels]  # 为排序后的标签分配颜色
    row_indices = []
    for i, sig in enumerate(signatures):
        row_indices.extend([i] * len(sig))  # 为每个签名基因分配索引
    col_rowside = [col_row[i] for i in row_indices]  # 为签名基因分配颜色

    # 创建图形并设置透明背景
    fig = plt.figure(figsize=figsize)  # 创建图形对象，设置大小为 (10, 10) 英寸
    fig.patch.set_alpha(0)  # 设置图形背景透明度为 0（完全透明）
    ax = plt.gca()  # 获取当前轴对象
    ax.patch.set_alpha(0)  # 设置轴背景透明度为 0（完全透明）

    # 绘制热图
    sns.heatmap(
        exp2plot,  # 热图数据
        cmap=["#FFFFFF", '#FFE4E1', '#FFB6C1', '#FF9999', '#FF6666', '#FF4040'],  # 颜色映射，模仿 R 颜色
        cbar=False,  # 禁用颜色条
        xticklabels=False,  # 禁用 x 轴刻度标签
        yticklabels=True,  # 启用 y 轴刻度标签
    )

    # 去掉 y 轴刻度线
    ax.tick_params(axis='y', length=0)  # 设置 y 轴刻度线长度为 0，隐藏刻度线

    # 设置轴的可见范围
    n_cells = len(cell_order)  # 细胞数量
    n_genes = len(signature_genes)  # 基因数量
    ax.set_xlim(0, n_cells)  # 设置 x 轴范围，从 0 到细胞数
    ax.set_ylim(0, n_genes)  # 设置 y 轴范围，从 0 到基因数
    # 确保顶部颜色条可见，调整 ylim 上限
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 1.5)  # 扩展 y 轴上限，增加 1.5 单位
    ax.set_xlim(ax.get_xlim()[0] - 1.5/n_genes*n_cells, ax.get_xlim()[1])  # 扩展 x 轴左侧，基于比例计算

    # 添加侧边颜色条
    for i, color in enumerate(col_colside):
        plt.gca().add_patch(plt.Rectangle(
            (i, ax.get_ylim()[1] - 1),  # 矩形左下角 x 坐标为细胞索引 i，y 坐标为 y 轴上限减 1
            1,  # 矩形宽度为 1
            1,  # 矩形高度为 1
            color=color  # 填充颜色
        ))
    for i, color in enumerate(col_rowside):
        plt.gca().add_patch(plt.Rectangle(
            (ax.get_xlim()[0], i),  # 矩形左下角 x 坐标为 x 轴下限，y 坐标为基因索引 i
            1/n_genes*n_cells,  # 矩形宽度为比例计算值（1/28*400）
            1,  # 矩形高度为 1
            color=color  # 填充颜色
        ))

    # 设置标签
    plt.xlabel(xlabel)  # 设置 x 轴标签为 "Cells"
    plt.ylabel(ylabel)  # 设置 y 轴标签为 "Genes"

    # 创建图例
    legend_text = [f'Gene identifiers_{name}' for name in signatures.keys()] + unique_labels  # 图例文本，包含基因标识和唯一标签
    legend_cor = list(col_row[:n_signatures]) + [col_col[unique_labels.index(label)] for label in unique_labels]  # 图例颜色
    legend_elements = [Patch(facecolor=color, label=text) for text, color in zip(legend_text, legend_cor)]  # 创建图例元素
    plt.legend(
        handles=legend_elements,  # 图例元素列表
        loc='upper right',  # 图例位置，右上角
        # 图例的移动会改变整个图像的可见范围，而patch不会
        bbox_to_anchor=legend_bbox_to_anchor,  # 图例锚点位置，移到右侧 1.35 倍图表宽度，顶部对齐
        fontsize=8  # 图例字体大小
    )

    plt.tight_layout()  # 注释掉，防止与自定义布局冲突
    if save_path:
        plt.savefig(save_path)  # 保存图形
        plt.close()
    else:
        plt.show()
    
    return None