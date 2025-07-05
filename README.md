# SCINApy

A Semi-Supervised Category Identification and Assignment Tool for single-cell RNA-Seq and Cytof/FACS data.

## Installation

You can install SCINApy from PyPI using pip:

```bash
pip install SCINApy
```

SCINApy requires Python 3.7 or higher and the following dependencies:

- numpy>=2.1.2
- pandas>=2.2.3
- scipy>=1.15.3
- anndata>=0.11.4
- seaborn>=0.13.2
- matplotlib>=3.10.0

## Usage

SCINApy provides tools for cell type assignment and visualization of single-cell data. Key features include:

- SCINA: Core algorithm for semi-supervised cell type identification.
- plotheat_scina: Visualize SCINA results with a heatmap.
- Command-line interface via scinapy for easy execution.(***still working in progress!!!***)

### Command-Line Example

Run the command-line interface with sample data:

```bash
scinapy --data data/matrix.csv --signatures data/signatures.json --output results.pkl --job_id test
```

### Jupyter Notebook Example

An example Jupyter Notebook (example.ipynb) is included to demonstrate the usage of SCINApy. 

## Application

The SCINA algorithm implemented in this package is based on the methodology originally developed by Zhang et al. (2019), where the technical details are comprehensively elaborated ([SCINA: A Semi-Supervised Subtyping Algorithm of Single Cells and Bulk Samples](https://pubmed.ncbi.nlm.nih.gov/31336988/)). This package, SCINApy, is also developed based on the same author's R package [SCINA](https://github.com/jcao89757/SCINA), adapting its functionality to the Python ecosystem for enhanced usability and integration with modern single-cell analysis tools.

## Documentation

- **PyPI Page**: [SCINApy 0.1.1](https://pypi.org/project/SCINApy/0.1.1/)
- **Source Code**: [GitHub Repository](https://github.com/hwr9912/SCINApy)
- **Issues**: [Report Issues](https://github.com/hwr9912/SCINApy/issues)

## Changelog

### [0.1.1] - 2025-07-05

#### Fixed

- Fixed ValueError in SCINA function when no signature genes are found in adata.

### [0.1.0] - 2025-07-04

- Initial release of SCINApy with core SCINA algorithm and visualization tools.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please submit issues or pull requests on the GitHub repository.

## Contact

For support or questions, please open an issue on GitHub or contact the author at [hwr9912@gmail.com](mailto:hwr9912@gmail.com).
