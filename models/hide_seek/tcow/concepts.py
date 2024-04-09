from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer
import numpy as np
import faiss
from einops import rearrange
import torch


def cluster_features(features, max_num_clusters=2, elbow=False):
    '''
    :param features (B, C, T, H, W) tensor.
    :cost (K, B, T, H, W) tensor
    '''

    (B, C, T, H, W) = features.shape

    # rearrange to N, C
    features = rearrange(features, 'B C T H W -> (B T H W) C')
    if elbow:
        clustering_alg = KMeans(n_init='auto')
        elbow_alg = kelbow_visualizer(clustering_alg, np.array(features.cpu()), k=(2, max_num_clusters), metric='silhouette',show=False, timings=False)
        # elbow_alg = kelbow_visualizer(clustering_alg, np.array(cluster_feature.cpu()), k=(2, 10), metric='distortion',show=False, timings=False)
        # elbow_alg.fit(y[0].cpu())
        num_clusters = elbow_alg.elbow_value_
        if num_clusters is None:
            num_clusters = max_num_clusters+1
        # clust_out =  KMeans(n_clusters=num_clusters, random_state=0).fit_predict(y[0,1:].cpu())
    else:
        num_clusters = max_num_clusters
    cost =  KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit_transform(features.cpu())
    cost = torch.from_numpy(cost)
    # cost = rearrange(cost, '(B H W T) K -> (K B) T H W', B=B, T=T, H=H, W=W, K=num_clusters)
    cost = rearrange(cost, '(B T H W) K -> (K B) T H W', B=B, T=T, H=H, W=W, K=num_clusters)

    return cost