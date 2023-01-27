import cudf
import cuml
import torch
import cupy as cp
import numpy as np
import scanpy as sc
import muon as mu
import pandas as pd
import gc
from numba import cuda
import matplotlib.pyplot as plt

from warnings import filterwarnings
from cuml import UMAP, PCA, HDBSCAN, KMeans, DBSCAN

from sklearn.model_selection import train_test_split
from scipy.stats import mode as scimode
from utils import *
from tqdm import tqdm

filterwarnings('ignore')

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

DATASET_PATH='../../sc_training.h5ad'

def generate_proportions(adata, genes, pca_comps=100, ko_delta=100, plot=False, verbose=True, save_plot=False):
    
    # Setup basic variables
    ordering = ['progenitor','effector','terminal exhausted','cycling','other']
    classes = [0,1,2,3,4]
    classdict = {s:c for (s,c) in zip(ordering, classes)}
    
    # Arrays for accessing categories
    genecolors = [colordict[s] for s in adata.obs['state']]
    classes = [classdict[s] for s in adata.obs['state']]
    
    if verbose:
        print("Loading adata into cudf...")
    # Cuda for speedup
    cudata = cudf.DataFrame(adata.to_df())
    
    if verbose:
        print("Performing training setup for gene knockout simulations...")
    # Copy all unperturbed cells out of adata
    unpert_ind = adata.obs['condition'] == 'Unperturbed'
    adata_unpert = adata[unpert_ind]
    
    train_ind = [x != 'Unperturbed' for x in adata.obs['condition']]
    
    # PCA fit only train data
    if verbose:
        print("Fitting PCA on train data...")
    pca_train = PCA(n_components = pca_comps)
    pca_train.fit(cudata[train_ind])
    if verbose:
        print("Done!")
    
    # Project train data into lower dimension
    train_x = pca_train.transform(cudata[train_ind])

    # Known labels for train data
    train_y = cudf.DataFrame([classdict[s] for s in adata.obs['state'][train_ind]])

    reducer = UMAP(n_neighbors=30, min_dist=0.1)
    reducer.fit(train_x,train_y)
    
    # Train embeddings
    t_embed = reducer.transform(train_x)
    
    # Cluster on Train
    thresh = 0.3

    hdbscan_t = HDBSCAN(min_samples=10,
        min_cluster_size=100,)

    clusters_t = hdbscan_t.fit_predict(t_embed)

    clustered = [p > thresh for p in hdbscan_t.probabilities_.to_numpy()]

    t_dn = t_embed[clustered]
    t_labels = hdbscan_t.labels_[clustered].to_numpy()
    
    # Set deleted labels to -1
    t_proba = hdbscan_t.probabilities_.to_numpy()
    t_deleted = np.where(t_proba <= thresh, True , False)
    t_labels_padded = hdbscan_t.labels_.copy()
    t_labels_padded[t_deleted] = -1

    if plot:
        plt.scatter(t_dn.to_numpy()[:,0], 
                    t_dn.to_numpy()[:,1],
                    c=t_labels,
                    s=0.13,
                    cmap='Spectral')
        plt.title("Umap train embeddings")
        plt.savefig("Umap-train.png")
        plt.pause(0.05)
        plt.clf()
    if save_plot:
        plt.scatter(t_dn.to_numpy()[:,0], 
                    t_dn.to_numpy()[:,1],
                    c=t_labels,
                    s=0.13,
                    cmap='Spectral')
        plt.title("Umap train embeddings")
        plt.savefig("Umap-train.png")
        plt.clf()
    proportions_dict = {}
    if verbose:
        print("Starting gene knockout simulations...")
    # Remove genes
    for gene_removed in genes:
        if verbose:
            print("Performing knockout on "+gene_removed+"...")
        # Get index of the gene and zero it out of our adata copy
        ind = (adata.var_names == gene_removed).nonzero()[0]

        adata_pert_copy = adata_unpert.to_df().to_numpy(copy=True)
        adata_pert_copy[:,ind] -= ko_delta
        
        # Convert to cudf and embed it in pca dims
        pert_cudata = cudf.DataFrame(adata_pert_copy)
        output = pca_train.transform(pert_cudata)
        
        # perturbed embeddings
        p_embed = reducer.transform(output)
        
        # Cluster on Perturbed
        thresh = 0.0

        hdbscan_p = HDBSCAN(min_samples=10,
            min_cluster_size=10,)

        clusters_p = hdbscan_p.fit_predict(p_embed)

        clustered = [p > thresh for p in hdbscan_p.probabilities_.to_numpy()]

        p_dn = p_embed[clustered]
        p_labels = hdbscan_p.labels_[clustered].to_numpy()
        
        if plot:
            plt.scatter(p_dn.to_numpy()[:,0], 
                        p_dn.to_numpy()[:,1],
                        c=p_labels,
                        s=0.13,
                        cmap='Spectral');
            plt.title("Umap "+str(ko_delta)+"-perturbed "+gene_removed+" embeddings")
            plt.savefig(gene_removed+"-"+str(ko_delta)+"-delta.png")
            plt.pause(0.05)
            plt.clf()
        if save_plot:
            plt.scatter(p_dn.to_numpy()[:,0], 
                        p_dn.to_numpy()[:,1],
                        c=p_labels,
                        s=0.13,
                        cmap='Spectral');
            plt.title("Umap "+str(ko_delta)+"-perturbed "+gene_removed+" embeddings")
            plt.savefig(gene_removed+"-"+str(ko_delta)+"-delta.png")
            plt.clf()
        # Convert cluster categories back to original categories
        p_to_t_map = catgy_cluster_map(p_dn.to_numpy(),p_labels,t_dn.to_numpy(), t_labels)
        t_to_orig_map = get_cluster_mapping_to_orig(adata, classdict, train_ind, t_labels_padded.to_pandas())
        
        # Compose above maps
        p_to_orig_map = {}
        for k in p_to_t_map:
            p_to_orig_map[k] = t_to_orig_map[p_to_t_map[k]]
        
        data_subset = pd.Series(p_labels)
        mapping = p_to_orig_map

        # The target range is 0 to 4, as in the given categories
        counts = np.zeros(5)
        categories = data_subset.unique()
        for i,cat in enumerate(categories):
            counts[mapping[cat]] += (data_subset == cat).sum()
        n = len(data_subset)
        proportions = counts / n
        
        proportions_dict[gene_removed] = proportions
    if verbose:
        print("Done!")
    return proportions_dict

if __name__ == '__main__':

    adata = sc.read_h5ad(DATASET_PATH)
    
    colors = ['blue','red','green','yellow','pink']
    ordering = ['progenitor','effector','terminal exhausted','cycling','other']
    classes = [0,1,2,3,4]

    colordict = {s:c for (s,c) in zip(ordering, colors)}
    classdict = {s:c for (s,c) in zip(ordering, classes)}

    genecolors = [colordict[s] for s in adata.obs['state']]
    classes = [classdict[s] for s in adata.obs['state']]

    valid_genes = [ 'Aqr', 'Bach2', 'Bhlhe40']
    test_genes = [ 'Ets1', 'Fosb', 'Mafk', 'Stat3']
    
    ko_deltas=[100]
    
    ko_deltas_results = {}
    for kod in tqdm(ko_deltas):
        while True:
            try:
                ko_deltas_results[kod] = generate_proportions(adata=adata, 
                                                              genes=valid_genes + test_genes, 
                                                              ko_delta=kod, 
                                                              plot=False, 
                                                              save_plot=False, 
                                                              verbose=True)
                break
            except AssertionError:
                print("Train clustering failed, retrying...")
 
    val_output = pd.DataFrame(columns=['gene','a_i','b_i','c_i','d_i','e_i'])

    proportions = ko_deltas_results[100]

    for g in valid_genes:
        g_props = proportions[g].tolist()
        data_g = [g] + g_props
        row = pd.DataFrame(columns=['gene','a_i','b_i','c_i','d_i','e_i'], data=[data_g])
        val_output = pd.concat([val_output,row])

    val_output.to_csv("validation_output.csv", index=False)

    test_output = pd.DataFrame(columns=['gene','a_i','b_i','c_i','d_i','e_i'])

    for g in test_genes:
        g_props = proportions[g].tolist()
        data_g = [g] + g_props
        row = pd.DataFrame(columns=['gene','a_i','b_i','c_i','d_i','e_i'], data=[data_g])
        test_output = pd.concat([test_output,row])

    test_output.to_csv("test_output.csv", index=False)