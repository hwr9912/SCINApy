import argparse
import pandas as pd
import anndata

from .core import SCINA
from .utils import preprocess_signatures
from .visual import plotheat_scina

def _run_scina_cli():
    """
    Still debugging!!!
    Command-line interface for SCINA algorithm, similar to EM_interface.R.
    Usage: python -m my_package --data <exp_csv> --signatures <signatures_csv> ...
    """
    parser = argparse.ArgumentParser(description="Run SCINA algorithm for cell type assignment.")
    parser.add_argument("--data", required=True, help="Path to expression matrix CSV file")
    parser.add_argument("--signatures", required=True, help="Path to signatures CSV file")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum iterations for EM algorithm")
    parser.add_argument("--convergence_n", type=int, default=10, help="Iterations for convergence check")
    parser.add_argument("--convergence_rate", type=float, default=0.99, help="Convergence rate threshold")
    parser.add_argument("--sensitivity_cutoff", type=float, default=1.0, help="Sensitivity cutoff for signatures")
    parser.add_argument("--allow_unknown", type=int, choices=[0, 1], default=0, help="Allow unknown category (0 or 1)")
    parser.add_argument("--output", required=True, help="Path to output file (e.g., results.pkl)")
    parser.add_argument("--job_id", required=True, help="Job ID for output files")
    
    args = parser.parse_args()

    # 读取数据
    exp = pd.read_csv(args.data, index_col=0)
    adata = anndata.AnnData(X=exp.T)  # 转换为AnnData，细胞为行，基因为列
    adata.var_names = exp.index
    adata.obs_names = exp.columns
    signatures = preprocess_signatures(args.signatures)

    # 运行SCINA
    results = SCINA(
        adata,
        signatures,
        max_iter=args.max_iter,
        convergence_n=args.convergence_n,
        convergence_rate=args.convergence_rate,
        sensitivity_cutoff=args.sensitivity_cutoff,
        allow_unknown=bool(args.allow_unknown),
        log_file=f"{args.job_id}_SCINA.log"
    )

    # 生成热图
    import matplotlib.pyplot as plt
    plotheat_scina(exp, results, list(signatures.values()))
    plt.savefig(f"{args.job_id}_output_plot.jpg", dpi=600)
    plt.close()

    # 保存结果
    import pickle
    with open(f"{args.job_id}_{args.output}", "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    _run_scina_cli()