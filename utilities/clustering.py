
from sklearn.cluster import KMeans
# from yellowbrick.cluster.elbow import kelbow_visualizer
import numpy as np
import faiss
from einops import rearrange
import torch
import sklearn
import matplotlib.pyplot as plt
# from skimage.segmentation import slic
# from fast_slic import Slic as fast_slic
# from cuda_slic.slic import slic as cuda_slic
from utilities.slic import slic
from pymf.pymf.cnmf import CNMF
from pymf.pymf.chnmf import CHNMF
from pymf.pymf.snmf import SNMF
# from smooth_cnmf.scnmf import *
import time
# import torchvideo

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
    # rearrange to N, C
    if elbow == 'yellowbrick':
        # rearrange to N, C
        features = rearrange(features, 'B C T H W -> (B T H W) C')
        clustering_alg = KMeans(n_init='auto')
        elbow_alg = kelbow_visualizer(clustering_alg, np.array(features.cpu()), k=(2, max_num_clusters), metric='silhouette',show=False, timings=False)
        # elbow_alg = kelbow_visualizer(clustering_alg, np.array(cluster_feature.cpu()), k=(2, 10), metric='distortion',show=False, timings=False)
        # elbow_alg.fit(y[0].cpu())
        num_clusters = elbow_alg.elbow_value_
        if num_clusters is None:
            num_clusters = max_num_clusters+1
        # clust_out =  KMeans(n_clusters=num_clusters, random_state=0).fit_predict(y[0,1:].cpu())
        alg =  KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
        cost = alg.fit_transform(features)
    elif elbow == 'dino':
        features = rearrange(features, 'B C T H W -> (B T H W) C')

        features = np.ascontiguousarray(np.array(features))
        # faiss.normalize_L2(features)

        # sample features by sample_interval
        features_sampled = np.ascontiguousarray(features[::sample_interval, :])
        # faiss.normalize_L2(features_sampled)


        sil_scores = []
        costs = []
        all_centroids = []
        # if applying three stages, use elbow to determine number of clusters in second stage, otherwise use the specified
        # number of parts.
        n_cluster_range = list(range(2, max_num_clusters))
        for idx, num_clusters in enumerate(n_cluster_range):
            concept_algorithm = faiss.Kmeans(d=features_sampled.shape[1], k=num_clusters, niter=300, nredo=10)
            concept_algorithm.train(features_sampled.astype(np.float32))
            cost, labels = concept_algorithm.index.search(features_sampled.astype(np.float32), 1)

            # OG dino
            # objective = squared_distances.sum()
            # sum_of_squared_dists.append(objective / features.shape[0])
            # if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow_threshold * sum_of_squared_dists[-2]):
            sillhouette_score = sklearn.metrics.silhouette_score(features_sampled, labels[:,0])
            curr_cost = sklearn.metrics.euclidean_distances(features, concept_algorithm.centroids)

            sil_scores.append(sillhouette_score)
            costs.append(curr_cost)
            all_centroids.append(concept_algorithm.centroids)
            if sillhouette_score > elbow_threshold:
                print('Threshold: {} > {}   Elbow at {}'.format(sillhouette_score, elbow_threshold, num_clusters))
                break

        # choose cost with highest sillhouette score
        cost = costs[np.argmax(sil_scores)]
        num_clusters = n_cluster_range[np.argmax(sil_scores)]
        centroids = all_centroids[np.argmax(sil_scores)]

        if verbose:
            if layer is not None:
                print('Layer {} Elbow at {}'.format(layer, num_clusters))
            else:
                print('Elbow at {}'.format(num_clusters))

    elif elbow == 'dino_og':
        features = rearrange(features, 'B C T H W -> (B T H W) C')
        features = np.array(features)
        normalized_all_descriptors = features.astype(np.float32)
        # normalize across feature channels

        # sampled_descriptors_list = [x[:, :, ::sample_interval, :] for x in descriptors_list]
        sampled_descriptors = features[::sample_interval, :]
        # all_sampled_descriptors = np.ascontiguousarray(np.concatenate(sampled_descriptors_list, axis=2)[0, 0])
        normalized_all_sampled_descriptors = sampled_descriptors.astype(np.float32)

        # new normalization
        # normalized_all_descriptors = (normalized_all_descriptors.T / np.linalg.norm(normalized_all_descriptors, axis=1)).T
        # normalized_all_sampled_descriptors = (normalized_all_sampled_descriptors.T / np.linalg.norm(normalized_all_sampled_descriptors, axis=1)).T

        # old normalization
        # faiss.normalize_L2(normalized_all_descriptors)  # in-place operation
        # faiss.normalize_L2(normalized_all_sampled_descriptors)  # in-place operation


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


        # print('Intra Elbow for layer {} at {}'.format(layer, num_clusters))
        # gives the same output as the curr_cost argmax
        # num_descriptors_per_image = [[T * H * W] for x in range(B)]
        # labels_per_image = np.split(labels, np.cumsum(num_descriptors_per_image))
        cost = sklearn.metrics.euclidean_distances(normalized_all_descriptors, algorithm.centroids)
        centroids = algorithm.centroids

    elif elbow == 'multikmeans':
        features = rearrange(features, 'B C T H W -> (B T H W) C')
        features =  np.ascontiguousarray(np.array(features))
        normalized_all_descriptors = features.astype(np.float32)
        # normalize across feature channels

        # sampled_descriptors_list = [x[:, :, ::sample_interval, :] for x in descriptors_list]
        mdl = CNMF(features, num_bases=2)
        mdl.factorize(niter=100)
        sampled_descriptors = features[::sample_interval, :]
        # all_sampled_descriptors = np.ascontiguousarray(np.concatenate(sampled_descriptors_list, axis=2)[0, 0])
        normalized_all_sampled_descriptors = sampled_descriptors.astype(np.float32)

        all_costs = []
        all_centroids = []
        for idx in range(len(n_segments)):
            num_clusters = int(n_segments[idx])

            algorithm = faiss.Kmeans(d=normalized_all_sampled_descriptors.shape[1], k=num_clusters, niter=300, nredo=10)
            algorithm.train(normalized_all_sampled_descriptors.astype(np.float32))
            # squared_distances, labels = algorithm.index.search(normalized_all_descriptors.astype(np.float32), 1)
            cost = sklearn.metrics.euclidean_distances(normalized_all_descriptors, algorithm.centroids)
            centroids = algorithm.centroids
            cost = torch.from_numpy(cost)
            cost = rearrange(cost, '(B T H W) K -> B K T H W', B=B, T=T, H=H, W=W, K=num_clusters)
            all_costs.append(cost)
            all_centroids.append(torch.tensor(centroids))

        return all_costs, n_segments, all_centroids

    elif elbow == 'slic':
        # rearrange features to be in the correct 3D format for slic
        features = np.ascontiguousarray(np.array(rearrange(features, 'B C T H W -> (B T) H W C')))
        # sampled_descriptors = features[::sample_interval, :]

        all_labels = []
        for idx in range(len(n_segments)):
            n_clusters = int(n_segments[idx])
            compactness = float(slic_compactness[idx])

            # start_time = time.time()
            # labels = cuda_slic(features, n_segments=n_segments, compactness=compactness, min_size_factor=0.5, max_size_factor=3)
            # print('CUDA_SLIC total time in minutes: {:.2f}'.format((time.time() - start_time) / 60))

            # start_time = time.time()
            # for i in range(5,15):
            #     labels = slic(features, n_segments=i, compactness=compactness, start_label=0,spacing=[1, 1, 1])
            #     print('Number of actual/unique labels: {}/{}'.format(np.unique(labels).shape[0], i))

            labels = slic(features, n_segments=n_clusters, compactness=compactness, start_label=0, spacing=[1,1,1])
            # print('SKIMAGE_SLIC total time in minutes: {:.2f}'.format((time.time() - start_time) / 60))
            # spacing -> a higher spacing value puts more weight
            all_labels.append(torch.tensor(labels))

            # for v in np.unique(labels):
            #     # create a mask to access one region at the time
            #     mask = np.ones(features.shape[:3])
            #     mask[labels == v] = 0
            #     feature_centroid = torch.matmul(torch.tensor(mask).reshape(-1).float(), torch.tensor(features).reshape(-1,64).float()) / mask.sum()
                # [12.737149  34.15757   27.006994   6.0785937  6.3228564  6.1999993, 5.810527   7.564957   6.229608   4.290546   5.6015296
                #  5.6308684, 4.3334565  5.3655014  7.9903765  6.698946   6.7373285  5.600165, 6.8132277  5.587966   7.381221   7.0578036
                #  5.8571987
                # normal slic
                # [[12.737149   34.15757    27.006994    6.0785937   6.3228564   6.1999993, 5.810527    7.564957    6.229608    4.290546
                #   5.6015296   5.6308684, 4.3334565   5.3655014   7.9903765   6.698946    6.7373285   5.600165, 6.8132277   5.587966
                #   7.381221    7.0578036   5.8571987   6.3492374, 6.1231346   6.176895    5.4583497   5.0926137   3.952702    6.5277147,
                #   4.530099    7.1805024   5.7432795   6.015345    5.6561136   7.040934, 6.5437126   6.254782    7.12064     2.9690883
                #   5.1957564   5.627487, 7.853937    5.957773    5.8674035   5.26683     6.140152    6.2338405, 6.0875483   5.369197
                #   4.865013    5.0432734   1.2618723   4.309877, 5.8991985   5.871149    6.3022428   6.982421    5.907023    5.6986346,
                #   8.328649    5.9272933   6.280222    4.88701     5.850719    5.759514, 6.213825],
                #  [19.012297   26.405676   97.44625     6.3754563   6.282541    6.468054, 5.8583455   7.539127    6.3865066   4.029367
                #   5.517306    5.902568, 4.2849555   4.8613577   8....
        # start_time = time.time()
        # labels = slic(features, n_segments=max_num_clusters, compactness=compactness, start_label=0, slic_zero=slic_zero)
        # print('SKIMAGE_SLIC total time in minutes: {:.2f}'.format((time.time() - start_time) / 60))

        # cuda slic
        # start_time = time.time()
        # labels = cuda_slic(features, n_segments=max_num_clusters, compactness=compactness, min_size_factor=0.5, max_size_factor=3, enforce_connectivity=False)
        # print('CUDA_SLIC total time in minutes: {:.2f}'.format((time.time() - start_time) / 60))

        # fast_slic
        # algorithm = fast_slic(num_components=max_num_clusters, compactness=compactness)
        # start_time = time.time()
        # labels = algorithm.iterate(features)
        # print('fast_slic total time in minutes: {:.2f}'.format((time.time() - start_time) / 60))

        return all_labels, n_clusters, None
    elif elbow == 'cnmf':
        features = rearrange(features, 'B C T H W -> (B T H W) C')
        features =  np.ascontiguousarray(np.array(features))
        normalized_all_descriptors = features.astype(np.float32)
        # sampled_descriptors_list = [x[:, :, ::sample_interval, :] for x in descriptors_list]

        # normalize across feature channels
        all_cost = []
        all_centroids = []
        for idx in range(len(n_segments)):
            n_clusters = int(n_segments[idx])
            mdl = CNMF(normalized_all_descriptors.T, num_bases=n_clusters)
            mdl.factorize(niter=100)
            cost = mdl.H
            cost = torch.from_numpy(cost)
            cost = rearrange(cost, '(B T H W) K -> B K T H W', B=B, T=T, H=H, W=W, K=n_clusters)
            centroids = mdl.W.T
            if full_dataset:
                return cost, n_clusters, centroids

            all_cost.append(cost)
            all_centroids.append(centroids)
            return all_cost, n_segments, all_centroids



    elif elbow == 'faiss_k':
        num_clusters = max_num_clusters
        features = np.array(features)
        # if applying three stages, use elbow to determine number of clusters in second stage, otherwise use the specified
        # number of parts.
        concept_algorithm = faiss.Kmeans(d=features.shape[1], k=max_num_clusters, niter=300, nredo=10)
        concept_algorithm.train(features.astype(np.float32))
        _, labels = concept_algorithm.index.search(features.astype(np.float32), 1)

        # OG dino
        # objective = squared_distances.sum()
        # sum_of_squared_dists.append(objective / features.shape[0])
        # if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow_threshold * sum_of_squared_dists[-2]):
        cost = sklearn.metrics.euclidean_distances(features, concept_algorithm.centroids)
    elif elbow == 'crop':
        num_rows = [3]
        num_cols = [4]
        num_depths = [1]

        # num_rows = [3]
        # num_cols = [3, 4]
        # num_depths = [2, 3]
        # crop features into 9 spatial crops and 3 temporal crops
        # b c t h w
        all_labels = []
        for curr_num_row in num_rows:
            for curr_num_col in num_cols:
                for curr_num_depth in num_depths:
                    curr_crop = torch.zeros_like(features[:,0]).type(torch.uint8)
                    row_size = features.shape[3] // curr_num_row
                    col_size = features.shape[4] // curr_num_col
                    depth_size = features.shape[2] // curr_num_depth
                    label_count = 0
                    for row_idx in range(curr_num_row):
                        for col_idx in range(curr_num_col):
                            for depth_idx in range(curr_num_depth):
                                curr_crop[:, depth_idx * depth_size: (depth_idx + 1) * depth_size,
                                row_idx * row_size: (row_idx + 1) * row_size,
                                col_idx * col_size: (col_idx + 1) * col_size] = label_count
                                label_count += 1
                    all_labels.append(curr_crop)
                    # get crop
        return all_labels, 0, None

    elif elbow == 'random':
        # rearrange features to be in the correct 3D format for slic
        # features = np.ascontiguousarray(np.array(rearrange(features, 'B C T H W -> (B T) H W C')))
        # flatten features into 1D array
        features = rearrange(features, 'B C T H W -> (B T H W) C')
        # randomly permute features
        features = features[torch.randperm(features.shape[0])]
        # reshape features back into 3D array
        features = np.ascontiguousarray(np.array(rearrange(features, '(B T H W) C -> (B T) H W C', B=B, T=T, H=H, W=W)))


        all_labels = []
        for idx in range(len(n_segments)):
            n_clusters = int(n_segments[idx])
            compactness = float(slic_compactness[idx])
            labels = slic(features, n_segments=n_clusters, compactness=compactness, start_label=0, spacing=[1,1,1])
            all_labels.append(torch.tensor(labels))

        return all_labels, n_clusters, None

    else:
        num_clusters = max_num_clusters
        alg =  KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
        cost = alg.fit_transform(features)
    cost = torch.from_numpy(cost)
    cost = rearrange(cost, '(B T H W) K -> B K T H W', B=B, T=T, H=H, W=W, K=num_clusters)

    return cost, num_clusters, centroids



def cluster_dataset(features, max_num_clusters=10, elbow=False, elbow_threshold=0.975, layer=None, normalize=False, verbose=False):
    '''
    :param features (N, C) tensor.
    :return
    '''

    (N, C) = features.shape

    if N < max_num_clusters:
        max_num_clusters = N

    if elbow == 'yellowbrick':
        clustering_alg = KMeans(n_init='auto')
        elbow_alg = kelbow_visualizer(clustering_alg, features, k=(2, max_num_clusters), metric='silhouette',show=False, timings=False)
        num_clusters = elbow_alg.elbow_value_
        if num_clusters is None:
            num_clusters = max_num_clusters+1
        alg =  KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
        cost = alg.fit_transform(features)
    elif elbow == 'dino':
        # dino feature clustering method
        features = np.array(features)
        # faiss.normalize_L2(features)
        sil_scores = []
        costs = []
        center_list = []
        # if applying three stages, use elbow to determine number of clusters in second stage, otherwise use the specified
        # number of parts.
        n_cluster_range = list(range(2, max_num_clusters))
        for idx, num_clusters in enumerate(n_cluster_range):
            concept_algorithm = faiss.Kmeans(d=features.shape[1], k=num_clusters, niter=300, nredo=10)
            concept_algorithm.train(features.astype(np.float32))
            cost, labels = concept_algorithm.index.search(features.astype(np.float32), 1)

            # OG dino
            # objective = squared_distances.sum()
            # sum_of_squared_dists.append(objective / features.shape[0])
            # if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow_threshold * sum_of_squared_dists[-2]):
            sillhouette_score = sklearn.metrics.silhouette_score(features, labels[:, 0])
            curr_cost = sklearn.metrics.euclidean_distances(features, concept_algorithm.centroids)

            sil_scores.append(sillhouette_score)
            costs.append(curr_cost)
            center_list.append(concept_algorithm.centroids)
            # if sillhouette_score > elbow_threshold:
            #     print('Threshold: {} > {}   Elbow at {}'.format(sillhouette_score, elbow_threshold, num_clusters))
            #     break

        # choose cost with highest sillhouette score
        # cost = costs[np.argmax(sil_scores)]
        num_clusters = n_cluster_range[np.argmax(sil_scores)]
        centers = center_list[np.argmax(sil_scores)]
        if verbose:
            if layer is not None:
                print('Layer {} Elbow at {}'.format(layer, num_clusters))
            else:
                print('Elbow at {}'.format(num_clusters))

    elif elbow == 'dino_og':
        features = np.array(features)
        normalized_all_descriptors = features.astype(np.float32)
        # normalize across feature channels

        # sampled_descriptors_list = [x[:, :, ::sample_interval, :] for x in descriptors_list]
        # sampled_descriptors = features[::sample_interval, :]
        # all_sampled_descriptors = np.ascontiguousarray(np.concatenate(sampled_descriptors_list, axis=2)[0, 0])
        # normalized_all_sampled_descriptors = sampled_descriptors.astype(np.float32)

        # new normalization
        # normalized_all_descriptors = (normalized_all_descriptors.T / np.linalg.norm(normalized_all_descriptors, axis=1)).T
        # normalized_all_sampled_descriptors = (normalized_all_sampled_descriptors.T / np.linalg.norm(normalized_all_sampled_descriptors, axis=1)).T

        # old normalization
        if normalize:
            faiss.normalize_L2(features)  # in-place operation
            faiss.normalize_L2(normalized_all_descriptors)  # in-place operation


        sum_of_squared_dists = []
        n_cluster_range = list(range(1, max_num_clusters))
        for num_clusters in n_cluster_range:
            algorithm = faiss.Kmeans(d=normalized_all_descriptors.shape[1], k=num_clusters, niter=300, nredo=10)
            try:
                algorithm.train(normalized_all_descriptors.astype(np.float32))
            except:
                print('Faiss error, skipping')
                continue
            squared_distances, labels = algorithm.index.search(normalized_all_descriptors.astype(np.float32), 1)
            objective = squared_distances.sum()
            sum_of_squared_dists.append(objective / normalized_all_descriptors.shape[0])
            if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow_threshold * sum_of_squared_dists[-2]):
                break


        # gives the same output as the curr_cost argmax
        # num_descriptors_per_image = [[T * H * W] for x in range(B)]
        # labels_per_image = np.split(labels, np.cumsum(num_descriptors_per_image))
        # cost = sklearn.metrics.euclidean_distances(normalized_all_descriptors, algorithm.centroids)
        centers = algorithm.centroids


    elif elbow == 'kmeans':
        num_clusters = max_num_clusters
        alg = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
        alg.fit(features)
        centers = alg.cluster_centers_

    elif elbow == 'cnmf':
        features =  np.ascontiguousarray(np.array(features)).astype(np.float32)
        if normalize:
            faiss.normalize_L2(features)
        # sampled_descriptors_list = [x[:, :, ::sample_interval, :] for x in descriptors_list]

        num_clusters = max_num_clusters

        mdl = CNMF(features.T, num_bases=num_clusters)

        mdl.factorize(niter=10000)

        '''
        mdl.W = np.dot(mdl.data, mdl.G)
        '''

        centers = mdl.W.T
        # weights = mdl.H.T
        weights = mdl.G
        asg = np.argmax(weights, 1)
        cost = np.min(weights, -1)
        return asg, cost, centers, (mdl.W, mdl.G, mdl.H)


    elif elbow == 'cnmf_elbow':
        normalized_all_descriptors = np.array(features).astype(np.float32)
        # normalize across feature channels

        # sampled_descriptors_list = [x[:, :, ::sample_interval, :] for x in descriptors_list]
        # sampled_descriptors = features[::sample_interval, :]
        # all_sampled_descriptors = np.ascontiguousarray(np.concatenate(sampled_descriptors_list, axis=2)[0, 0])
        # normalized_all_sampled_descriptors = sampled_descriptors.astype(np.float32)

        # new normalization
        # normalized_all_descriptors = (normalized_all_descriptors.T / np.linalg.norm(normalized_all_descriptors, axis=1)).T
        # normalized_all_sampled_descriptors = (normalized_all_sampled_descriptors.T / np.linalg.norm(normalized_all_sampled_descriptors, axis=1)).T

        # old normalization
        if normalize:
            faiss.normalize_L2(normalized_all_descriptors)
        # faiss.normalize_L2(normalized_all_descriptors)  # in-place operation
        # faiss.normalize_L2(normalized_all_sampled_descriptors)  # in-place operation


        sum_of_squared_dists = []
        n_cluster_range = list(range(1, max_num_clusters))
        for num_clusters in n_cluster_range:
            algorithm = faiss.Kmeans(d=normalized_all_descriptors.shape[1], k=num_clusters, niter=300, nredo=10)
            algorithm.train(normalized_all_descriptors.astype(np.float32))
            squared_distances, labels = algorithm.index.search(normalized_all_descriptors.astype(np.float32), 1)
            objective = squared_distances.sum()
            sum_of_squared_dists.append(objective / normalized_all_descriptors.shape[0])
            if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow_threshold * sum_of_squared_dists[-2]):
                break



        # now use elbow for CNMF
        features =  np.ascontiguousarray(np.array(features)).astype(np.float32)
        if normalize:
            faiss.normalize_L2(features)
        # faiss.normalize_L2(features)  # in-place operation

        # smoothConvexNMF(features, 3, beta=0.01, max_iter=1000)
        mdl = CNMF(features.T, num_bases=num_clusters)
        # mdl = CHNMF(features, num_bases=num_clusters)
        mdl.factorize(niter=10000)


        '''
        mdl.W = np.dot(mdl.data, mdl.G)
        '''

        centers = mdl.W.T
        # weights = mdl.H.T
        weights = mdl.G
        asg = np.argmax(weights, 1)
        cost = np.min(weights, -1)
        return asg, cost, centers, (mdl.W, mdl.G, mdl.H)


    else:
        raise NotImplementedError

    d = np.linalg.norm(np.expand_dims(np.array(features), 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
    asg = np.argmin(d, 1)
    cost = np.min(d, -1)
    return asg, cost, centers, None