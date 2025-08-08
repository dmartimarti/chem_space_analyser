# Chemical space analysis
Generating multidimensional analyses of chemical compounds with ease

This script is designed to generate a multidimensional analysis of chemical compounds. It is based on the RDKit library and allows for multiple featurization methods and clustering algorithms to explore the chemical space.

The script will read a csv file, generate features for each molecule, perform dimensionality reduction (t-SNE or UMAP), and optionally cluster the results.

## Input Data

The input **must be** a csv file with three columns where the first **must** be named as `smiles` and the second **must** be named `name`. The third one can contain any metadata and will be used for coloring the plots (unless clustering is enabled). The file should have the following format:

`smiles,name,metadata
CC(C)(CN(CC1)CCC1NCc1nc([nH]cc2)c2cc1)O,name_1,active
CN(C)C1CCN(CCCNc2c3[s]c(N(C)C)nc3ncn2)CC1,name_2,active
CCCN1CC(CNC(NC(CN(C)C2)c3c2cccc3)=O)CC1,name_3,inactive
...`

An example input file `example_data.csv` is provided.

## Output Files

The script will create an output folder containing the following files:
- `features.csv`: A CSV file with the calculated features for each compound (e.g., functional groups, fingerprints, or physicochemical properties).
- `tsne_results_perplexity_...csv`: A CSV file with the t-SNE coordinates. If clustering is performed, this file will include a `cluster` column and will be named `..._clustered.csv`.
- `umap_results_...csv`: A CSV file with the UMAP coordinates. If clustering is performed, this file will include a `cluster` column and will be named `..._clustered.csv`.
- `tsne_plot.html`: An interactive 3D HTML plot of the t-SNE results.
- `umap_plot.html`: An interactive 3D HTML plot of the UMAP results (if `--umap` is used).

## Dependencies

You need to install these dependencies. We recommend using a Conda environment.

`pip install rdkit hdbscan
conda install plotly pandas scikit-learn tqdm`

## Usage

Basic usage:

`python chem_spacer.py -i input.csv -o output_folder`

### Optional Parameters

Besides the required input and output, you can also set the following optional parameters:

- `--threads`: The number of threads to use. Default is 1.
- `--featurizer`: The type of molecular features to calculate. Default is `fragments`.
  - `fragments`: Counts of RDKit functional groups.
  - `morgan` or `ecfp`: Morgan/ECFP circular fingerprints (2048 bits, radius 2).
  - `physchem`: A panel of physicochemical properties (MolWt, LogP, TPSA, etc.).
- `--perplexity`: The perplexity of the t-SNE algorithm. Default is 30.
- `--iterations`: The number of iterations of the t-SNE algorithm. Default is 1000.
- `--perplexity_grid`: Bypass the `--perplexity` parameter and run t-SNE with a grid of perplexities ([5, 10, 20, 30, 40, 50]). Default is False.
- `--umap`: Calculate UMAP coordinates in addition to t-SNE. Default is False.
- `--cluster_method`: The clustering algorithm to apply to the dimensionality reduction results. No clustering is performed by default.
  - `kmeans`: K-Means clustering.
  - `dbscan`: DBSCAN clustering.
  - `hdbscan`: HDBSCAN clustering.
- `--n_clusters`: The number of clusters to use for K-Means. Default is 5.


### Advanced Example

An example of a more advanced usage is shown below. This command will:
- Use the `morgan` featurizer.
- Compute both t-SNE and UMAP.
- Apply `hdbscan` clustering to the results.
- Use 8 threads for computation.

`python chem_spacer.py -i example_data.csv -o results --featurizer morgan --umap --cluster_method hdbscan -t 8`

## TODO:
- [x] Add UMAP support
- [x] Add the option to use the Morgan fingerprints
- [x] Add the option to use the ECFP fingerprints
- [x] Support for Drug names
- [ ] Add options to customize fingerprint parameters (e.g., radius, nBits).
- [ ] Add options to customize clustering parameters (e.g., eps for DBSCAN).
