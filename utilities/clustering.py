
import faiss
import torch
import sklearn
import numpy as np
from einops import rearrange

from utilities.slic import slic
from pymf.pymf.cnmf import CNMF

def cluster_features(features, max_num_clusters=10, elbow=False, elbow_threshold=0.975,layer=None, verbose=False,
                     sample_interval=10, n_segments=10, slic_compactness=0.01, full_dataset=False, spacing=[1,1,1]):
    '''
    :param features (B, C, T, H, W) tensor.
    :cost (K, B, T, H, W) tensor
    '''
    centroids = []
    if full_dataset:
        assert elbow in ['yellowbrick', 'dino', 'dino_og', 'cnmf']

    (B, C, T, H, W) = features.shape

    if elbow == 'kmeans':
        features = rearrange(features, 'B C T H W -> (B T H W) C')
        features = np.array(features)
        normalized_all_descriptors = features.astype(np.float32)
        sampled_descriptors = features[::sample_interval, :]
        normalized_all_sampled_descriptors = sampled_descriptors.astype(np.float32)
        sum_of_squared_dists = []
        n_cluster_range = list(range(1, 15))
        for num_clusters in n_cluster_range:
            algorithm = faiss.Kmeans(d=normalized_all_sampled_descriptors.shape[1], k=num_clusters, niter=300, nredo=10)
            algorithm.train(normalized_all_sampled_descriptors.astype(np.float32))
            squared_distances, labels = algorithm.index.search(normalized_all_descriptors.astype(np.float32), 1)
            objective = squared_distances.sum()
            sum_of_squared_dists.append(objective / normalized_all_descriptors.shape[0])
            if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow_threshold * sum_of_squared_dists[-2]):
                break
        cost = sklearn.metrics.euclidean_distances(normalized_all_descriptors, algorithm.centroids)
        centroids = algorithm.centroids
    elif elbow == 'slic':
        # rearrange features to be in the correct 3D format for slic
        features = np.ascontiguousarray(np.array(rearrange(features, 'B C T H W -> (B T) H W C')))
        all_labels = []
        for idx in range(len(n_segments)):
            n_clusters = int(n_segments[idx])
            compactness = float(slic_compactness[idx])
            labels = slic(features, n_segments=n_clusters, compactness=compactness, start_label=0, spacing=[1,1,1])
            all_labels.append(torch.tensor(labels))
        return all_labels, n_clusters, None

    cost = torch.from_numpy(cost)
    cost = rearrange(cost, '(B T H W) K -> B K T H W', B=B, T=T, H=H, W=W, K=num_clusters)
    return cost, num_clusters, centroids



def cluster_dataset(features, max_num_clusters=10, elbow=False, elbow_threshold=0.975, layer=None, verbose=False):
    '''
    :param features (N, C) tensor.
    :return
    '''

    (N, C) = features.shape
    if N < max_num_clusters:
        max_num_clusters = N

    normalized_all_descriptors = np.array(features).astype(np.float32)
    sum_of_squared_dists = []
    n_cluster_range = list(range(1, max_num_clusters))
    for num_clusters in n_cluster_range:
        algorithm = faiss.Kmeans(d=normalized_all_descriptors.shape[1], k=num_clusters, niter=300, nredo=10,min_points_per_centroid=1,max_points_per_centroid=10000000)
        algorithm.train(normalized_all_descriptors.astype(np.float32))
        squared_distances, labels = algorithm.index.search(normalized_all_descriptors.astype(np.float32), 1)
        objective = squared_distances.sum()
        sum_of_squared_dists.append(objective / normalized_all_descriptors.shape[0])
        if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow_threshold * sum_of_squared_dists[-2]):
            break

    # now use the discovered elbow for CNMF
    features =  np.ascontiguousarray(np.array(features)).astype(np.float32)
    mdl = CNMF(features.T, num_bases=num_clusters)
    mdl.factorize(niter=10000)
    centers = mdl.W.T
    weights = mdl.G
    asg = np.argmax(weights, 1)
    cost = np.min(weights, -1)
    return asg, cost, centers, (mdl.W, mdl.G, mdl.H)

