from cuml import UMAP, PCA, HDBSCAN, KMeans, DBSCAN
import torch
import cupy as cp
from sklearn.model_selection import train_test_split
from scipy.stats import mode as scimode

def catgy_cluster_map(from_coords, from_cat, to_coords, to_cat):
    # So that we can define a mapping
    assert(len(np.unique(from_cat)) >= len(np.unique(to_cat)))
    
    dists = c1toc2(from_coords, from_cat, to_coords, to_cat)
    dists = sorted(dists, key=lambda x: x[0][0])
    
    mapper = {c1:c2 for (c1,c2),d in dists}
    return mapper

def get_centroid(posns):
    return np.sum(posns, axis=0)/len(posns)

def c1toc2(coords1, clust1, coords2, clust2):
    """
    coords1: nx2-dim numpy array of coordinates in embedded cluster
    clust1: n-dim numpy array of clusters
    
    coords2: mx2-dim numpy array of coordinates in embedded cluster
    clust2: m-dim numpy array of clusters
    """
    cents1 = {}
    for c in np.unique(clust1):
        c_coords = coords1[clust1 == c]
        cents1[c] = get_centroid(c_coords)
        
    cents2 = {}
    for c in np.unique(clust2):
        c_coords = coords2[clust2 == c]
        cents2[c] = get_centroid(c_coords)
    pairs = []
    for c1 in cents1:
        mindist = 10000
        pair = (0,0)
        for c2 in cents2:
            dist = np.sum(np.square(cents1[c1] - cents2[c2]))
            if dist < mindist:
                mindist = dist
                pair = (c1,c2)
        pairs.append((pair,mindist))
    return pairs



def cluster_match(adata, classdict, cluster_filter):
    '''
    cluster_filter: array of true/false that indicates which rows of adata we are concerned with
    
    returns: adata class given by classdict mapping
    '''
    cluster_data = [classdict[c] for c in adata.obs['state'][cluster_filter]]
    cluster_mode = mode(cluster_data).mode[0]
    return cluster_mode

def xentropy(p,q):
    return -np.sum(p*np.log(q))

def EMD(p, q):
    return np.abs(p-q).sum()

def proportions_clusters(data_subset, mapping, ordering=[0,1,2,3,4], return_n = False):
    counts = np.zeros(len(ordering))
    categories = data_subset.unique()
    for i,cat in enumerate(categories):
        counts[mapping[cat]] += (data_subset == cat).sum()
    n = len(data_subset)
    if return_n:
        return counts / n, n
    else:
        return counts / n

def get_orig_cluster(adata, classdict, subset, cluster):
    '''
    cluster selects the elements of the subset selected
    subset selects elements of adata
    '''
    indices_cluster = adata.obs['state'][subset].index[cluster]
    states_cluster = adata.obs['state'][indices_cluster]
    classes_cluster = [classdict[s] for s in states_cluster]
    mode = scimode(classes_cluster).mode[0]
    return mode

def get_cluster_mapping_to_orig(adata, classdict, subset, labels):
    """
    labels categorizes subset,
    subset selects part of adata
    
    range of mapping will always be {0,1,2,3,4}
    """
    mapping = {}
    categories = labels.unique()
    categories = categories[categories != -1]
    for c in categories:
        cluster = [c == cat for cat in labels.to_numpy()]
        mapping[c] = get_orig_cluster(adata, classdict, subset, cluster)
    return mapping