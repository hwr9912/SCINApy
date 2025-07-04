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


from .core import SCINA
from .visual import plotheat_scina
from .utils import preprocess_signatures, load_sample_data


__all__ = [
    "SCINA",
    "preprocess_signatures",
    "load_sample_data",
    "plotheat_scina"
]