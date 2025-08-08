#!/usr/bin/env python3
# description: script to extract functional groups from a list of molecules and analyse them with t-SNE and UMAP

from rdkit import Chem
import pandas as pd
from rdkit.Chem import Fragments, AllChem, Descriptors
import re
from optparse import OptionParser 
from tqdm import tqdm
from multiprocessing import get_context
import glob
import os
from sklearn.manifold import TSNE
import plotly.express as px
from umap import UMAP
from sklearn.cluster import KMeans, DBSCAN
import hdbscan

# parameters
parser = OptionParser()

parser.add_option("-i", "--inp",
                    dest = "input",
                    help = "input file in form of a csv file with with at least one column named 'smiles', and other colums as metadata to be preserved",
                    metavar = "INPUT")

parser.add_option("-o", "--out",
                    dest = "output",
                    help = "name of the output file that will store the chemical groups in a csv",
                    metavar = "OUTPUT")

parser.add_option("-t", "--threads",
                    default=1,
                    dest = "threads",
                    help = "number of threads to use",
                    metavar = "THREADS")

parser.add_option("-f", "--featurizer",
                    default="fragments",
                    dest = "featurizer",
                    help = "type of featurizer to use. Options are 'fragments', 'morgan', 'ecfp', 'physchem'. Default is 'fragments'",
                    metavar = "FEATURIZER")

parser.add_option("-p", "--perplexity",
                    default=30,
                    dest = "perplexity",
                    help = "perplexity for t-SNE",
                    metavar = "PERPLEXITY")

parser.add_option("-g", "--perplexity_grid",
                  default=False,
                  dest="perplexity_grid",
                  action="store_true",
                  help="use a grid search for perplexity with values 5, 10, 20, 30, 40, 50")

parser.add_option("-n", "--iterations",
                    default=1000,
                    dest = "iterations",
                    help = "number of iterations for t-SNE",
                    metavar = "ITERATIONS")

parser.add_option("-u", "--umap",
                  default=False,
                  dest="umap",
                  action="store_true",
                  help="Calculate UMAP with default parameters")

parser.add_option("-v", "--version",
                    action="store_true",
                    dest="version",
                    help="print the version of the script")

parser.add_option("--cluster_method",
                    dest="cluster_method",
                    default=None,
                    help="Clustering method to use. Options: 'kmeans', 'dbscan', 'hdbscan'. Default: None",
                    metavar="CLUSTER_METHOD")

parser.add_option("--n_clusters",
                    dest="n_clusters",
                    default=5,
                    help="Number of clusters for KMeans. Default: 5",
                    metavar="N_CLUSTERS")

(options, args) = parser.parse_args()

threads = int(options.threads)

# version info from the script
def version():
    """
    Returns the version of the script.
    """
    return '0.0.1'

def create_output_folder(output):
    """
    Create the output folder if it does not exist.
    """
    if not os.path.exists(output):
        os.makedirs(output)

def read_input(input_file):
    """
    Read the input file and return a pandas dataframe.
    """

    # read the input file
    df = pd.read_csv(input_file)

    # check if the input file has the required columns
    if 'smiles' not in df.columns or 'name' not in df.columns:
        raise ValueError("The input file does not have a column named 'smiles' or 'name'.")
    
    # check for duplicates
    if df.duplicated(subset='smiles').any():
        print('The input file has duplicates. Removing them.')

    # remove duplicates and NA values
    df = df.drop_duplicates(subset='smiles')
    df = df.dropna(subset=['smiles'])

    return df


def get_fragments(molecule):
    '''
    Parameters
    ----------
    molecule : RDKit molecule type
        Molecule object from the RDKit library in python.

    Returns
    -------
    Returns a vector of the functional groups present in the molecule.
    '''
    
    vector = []
    
    for func in frag_functions:
        str_to_eval = 'Fragments.'+func + '(molecule)'
        number = eval(str_to_eval)
        
        vector.append(number)
        
    return vector

def get_morgan_fingerprints(molecule, radius=2, nBits=2048):
    """
    Generates Morgan fingerprints for a molecule.
    """
    if molecule is None:
        return [0] * nBits
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=nBits)
    return list(fp)

# List of (name, function) tuples for the descriptors we want
_descriptor_list = [
    (name, func) for name, func in Descriptors.descList
    if name in ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds"]
]
# Keep the names and functions separate for convenience
physchem_descriptor_names = [name for name, func in _descriptor_list]
physchem_descriptor_funcs = [func for name, func in _descriptor_list]


def get_physchem_properties(molecule):
    """
    Calculates a defined set of physicochemical properties for a molecule.
    """
    if molecule is None:
        return [0] * len(physchem_descriptor_names)

    props = []
    for func in physchem_descriptor_funcs:
        try:
            props.append(func(molecule))
        except:
            props.append(0)
    return props

# get the functions within Fragments and store them in a list
frag = dir(Fragments)
r = re.compile("fr_.*")
frag_functions = list(filter(r.match, frag)) 

def calculate_features(input_file, featurizer, threads):
    print(f"\nReading input file: {input_file}\n")
    smiles = read_input(input_file)

    smiles['mols'] = smiles['smiles'].apply(Chem.MolFromSmiles) # convert smiles to mols
    smiles.dropna(subset=['mols'],inplace=True) # remove na in mols
    smiles.reset_index(drop=True, inplace=True) # reset index

    # calculate the features
    print(f"\nCalculating features using '{featurizer}' with {threads} threads.\n")
    p = get_context("fork").Pool(int(threads))

    if featurizer == "fragments":
        results = list(tqdm(p.imap(get_fragments, smiles['mols']), total=smiles.shape[0]))
        feature_names = frag_functions
    elif featurizer in ["morgan", "ecfp"]:
        from functools import partial
        radius = 2
        nBits = 2048
        morgan_func = partial(get_morgan_fingerprints, radius=radius, nBits=nBits)
        results = list(tqdm(p.imap(morgan_func, smiles['mols']), total=smiles.shape[0]))
        feature_names = [f"morgan_{i}" for i in range(nBits)]
    elif featurizer == "physchem":
        results = list(tqdm(p.imap(get_physchem_properties, smiles['mols']), total=smiles.shape[0]))
        feature_names = physchem_descriptor_names
    else:
        p.close()
        raise ValueError(f"Unknown featurizer: {featurizer}")

    p.close()

    # create a dataframe with the results
    features_df = pd.DataFrame(results, columns=feature_names)

    # concatenate the two dataframes
    final_df = pd.concat([smiles, features_df], axis=1)
    # remove the mols column
    final_df.drop(columns=['mols'], inplace=True)

    # save the final dataframe
    final_df.to_csv(f"{options.output}/features.csv", index=False)

    return final_df

def tsne_calc(final_df, perplexity=30, iterations=1000):
    """
    Calculate the t-SNE for the final dataframe.
    
    Parameters
    
    final_df : pandas dataframe that comes from the calculate_features function
    
    Returns
    
    tsne_df : pandas dataframe
    """
    print(f"\nCalculating t-SNE with perplexity {perplexity} and {options.iterations} iterations. This can take a while, go grab a coffee.\n")
    tsne = TSNE(n_components=3, random_state=123,
            perplexity=int(perplexity), 
            max_iter=int(iterations), 
            verbose=1,
            n_jobs=threads)

    # calculate the tSNE from the column fr_Al_COO to the last column
    X = final_df.iloc[:, 3:].values
    X_embedded = tsne.fit_transform(X)

    # create a dataframe with the tSNE results
    tsne_df = pd.DataFrame(X_embedded, columns=['tsne_1', 'tsne_2', 'tsne_3'])
    # add the two original columns from the final_df
    tsne_df = pd.concat([final_df.iloc[:, 0:3], tsne_df], axis=1)

    print("\nt-SNE calculation finished.\n")

    # save the t-SNE dataframe
    tsne_df.to_csv(f"{options.output}/tsne_results_perplexity_{perplexity}.csv", index=False)

    return tsne_df

def plotly_tsne(tsne_df, outfile='tsne_plot.html'):
    """
    Plot the t-SNE with plotly.
    
    Parameters
    
    tsne_df : pandas dataframe that comes from the tsne_calc function
    
    Returns
    
    None
    """
    columns = tsne_df.columns

    color_column = 'cluster' if 'cluster' in tsne_df.columns else columns[2]
    fig = px.scatter_3d(tsne_df, x='tsne_1', y='tsne_2', z='tsne_3', color=color_column, hover_data=['name', 'smiles'])

    fig.update_traces(marker=dict(size=6,opacity=0.8))
    
    # save the plot in the output folder
    fig.write_html(f"{options.output}/{outfile}")

    print(f"\nPlot saved in {options.output}/{outfile}\n")


def umap_calc(final_df, n_neighbors=15, min_dist=0.1, n_components=3):
    """
    Calculate the UMAP for the final dataframe.
    
    Parameters
    
    final_df : pandas dataframe that comes from the calculate_features function
    
    Returns
    
    umap_df : pandas dataframe
    """
    print(f"\nCalculating UMAP. This can take a while, go grab a coffee.\n")
    umap = UMAP(n_components=n_components,
            verbose=1, n_neighbors=n_neighbors, min_dist=min_dist,
            n_jobs=threads)

    # calculate the UMAP from the column fr_Al_COO to the last column
    X = final_df.iloc[:, 3:].values
    X_embedded = umap.fit_transform(X)

    # create a dataframe with the UMAP results
    umap_df = pd.DataFrame(X_embedded, columns=['umap_1', 'umap_2', 'umap_3'])
    # add the two original columns from the final_df
    umap_df = pd.concat([final_df.iloc[:, 0:3], umap_df], axis=1)

    print("\nUMAP calculation finished.\n")

    # save the UMAP dataframe
    umap_df.to_csv(f"{options.output}/umap_results_neigh_{n_neighbors}_mindist_{min_dist}.csv", index=False)

    return umap_df


def plotly_umap(umap_df, outfile='umap_plot.html'):
    """
    Plot the UMAP with plotly.
    
    Parameters
    
    umap_df : pandas dataframe that comes from the umap_calc function
    
    Returns
    
    None
    """
    columns = umap_df.columns

    color_column = 'cluster' if 'cluster' in umap_df.columns else columns[2]
    fig = px.scatter_3d(umap_df, x='umap_1', y='umap_2', z='umap_3', color=color_column, hover_data=['name', 'smiles'])

    fig.update_traces(marker=dict(size=6,opacity=0.9))
    
    # save the plot in the output folder
    fig.write_html(f"{options.output}/{outfile}")

    print(f"\nPlot saved in {options.output}/{outfile}\n")


def perform_clustering(df, method, n_clusters=5):
    """
    Performs clustering on the t-SNE or UMAP coordinates.
    """
    if 'tsne_1' in df.columns:
        coords = df[['tsne_1', 'tsne_2', 'tsne_3']]
    elif 'umap_1' in df.columns:
        coords = df[['umap_1', 'umap_2', 'umap_3']]
    else:
        raise ValueError("Dataframe does not contain t-SNE or UMAP coordinates.")

    print(f"\nPerforming clustering with {method}...\n")

    if method == 'kmeans':
        model = KMeans(n_clusters=int(n_clusters), random_state=123, n_init=10)
    elif method == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=5)
    elif method == 'hdbscan':
        model = hdbscan.HDBSCAN(min_cluster_size=5)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    clusters = model.fit_predict(coords)
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    df_clustered['cluster'] = df_clustered['cluster'].astype(str)

    print(f"Clustering complete. Found {len(df_clustered['cluster'].unique())} clusters.\n")

    return df_clustered


def main():
    # check the version
    if options.version:
        print(version())
        return

    if options.input is None:
        parser.error("option -i/--inp is required")

    if options.output is None:
        parser.error("option -o/--out is required")

    # create the output folder
    create_output_folder(options.output)

    # calculate the features
    final_df = calculate_features(options.input, options.featurizer, threads)


    if options.perplexity_grid:
        print(f"\nCalculating t-SNE with a grid search for perplexity. This will take a while! \n")
        perplexity_values = [5, 10, 20, 30, 40, 50]
        for perplexity in perplexity_values:
            tsne_df = tsne_calc(final_df, perplexity, int(options.iterations))
            if options.cluster_method:
                tsne_df = perform_clustering(tsne_df, options.cluster_method, options.n_clusters)
                tsne_df.to_csv(f"{options.output}/tsne_results_perplexity_{perplexity}_clustered.csv", index=False)
            plotly_tsne(tsne_df, f'tsne_plot_perplexity_{perplexity}.html')
        # if perplexity grid is used, umap is not calculated. So we exit here.
        return
    else:
    # calculate the t-SNE
        tsne_df = tsne_calc(final_df,
                            int(options.perplexity),
                            int(options.iterations))
        if options.cluster_method:
            tsne_df = perform_clustering(tsne_df, options.cluster_method, options.n_clusters)
            tsne_df.to_csv(f"{options.output}/tsne_results_perplexity_{options.perplexity}_clustered.csv", index=False)

        # plot the t-SNE
        plotly_tsne(tsne_df)

    if options.umap:
        # calculate the UMAP
        umap_df = umap_calc(final_df)

        if options.cluster_method:
            umap_df = perform_clustering(umap_df, options.cluster_method, options.n_clusters)
            umap_df.to_csv(f"{options.output}/umap_results_clustered.csv", index=False)

        # plot the UMAP
        plotly_umap(umap_df)

if __name__ == '__main__':
    main()