import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plotheat_scina(exp, results, signatures):
    """
    Plot a heatmap for SCINA results with signature genes and cell labels.
    
    :param exp: Expression matrix with genes as rows and cells as columns (pandas DataFrame).
    :param results: Dictionary containing 'cell_labels' with cell type assignments.
    :param signatures: List of lists containing signature genes for each cell type.

    :return: None, displays the heatmap.
    """
    # Remove non-existent signature genes
    allgenes = exp.index.tolist()
    signatures = [list(set([gene for gene in sig if gene in allgenes and pd.notna(gene)])) for sig in signatures]
    
    # Build side color bars
    n_signatures = len(signatures)
    col_row = plt.cm.tab10(np.linspace(0, 1, n_signatures))  # Equivalent to topo.colors
    unique_labels = list(dict.fromkeys(results['cell_labels']))  # Preserve order
    if 'unknown' not in unique_labels:
        unique_labels.append('unknown')
    n_labels = len(unique_labels)
    col_col = plt.cm.Pastel1(np.linspace(0, 1, n_labels))  # Equivalent to cm.colors
    
    # Build matrices for heatmap
    signature_genes = [gene for sig in signatures for gene in sig]
    sorted_labels = sorted(results['cell_labels'], key=lambda x: unique_labels.index(x))
    cell_order = [i for i, _ in sorted(enumerate(results['cell_labels']), key=lambda x: unique_labels.index(x[1]))]
    exp2plot = exp.loc[signature_genes, exp.columns[cell_order]]
    
    # Side colors
    col_colside = [col_col[unique_labels.index(label)] for label in sorted_labels]
    row_indices = []
    for i, sig in enumerate(signatures):
        row_indices.extend([i] * len(sig))
    col_rowside = [col_row[i] for i in row_indices]
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        exp2plot,
        cmap=["#FFFFFF", '#FFE4E1', '#FFB6C1', '#FF9999', '#FF6666', '#FF4040'],  # Mimics R colors
        cbar=False,  # Equivalent to key=False
        xticklabels=False,  # Equivalent to labCol=NA
        yticklabels=True,
        dendrogram_ratio=0,  # Equivalent to Rowv=FALSE, Colv=FALSE
    )
    
    # Add side color bars
    for i, color in enumerate(col_colside):
        plt.gca().add_patch(plt.Rectangle((i, 0), 1, 0.5, color=color, transform=plt.gca().get_xaxis_transform()))
    for i, color in enumerate(col_rowside):
        plt.gca().add_patch(plt.Rectangle((-0.5, i), 0.5, 1, color=color, transform=plt.gca().get_yaxis_transform()))
    
    # Set labels
    plt.xlabel('Cells')
    plt.ylabel('Genes')
    
    # Create legend
    legend_text = [f'Gene identifiers_{name}' for name in results['cell_labels']] + unique_labels
    legend_cor = list(col_row[:n_signatures]) + [col_col[unique_labels.index(label)] for label in unique_labels]
    legend_elements = [Patch(facecolor=color, label=text) for text, color in zip(legend_text, legend_cor)]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return None