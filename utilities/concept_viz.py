import os
import sys
import time
import random

import argparse
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
sys.path.append('/home/matthewkowal/Research/video_concept_discovery')
from utilities.utils import load_model, save_vcd, prepare_directories


def plot_num_clusters_perlayer(vcd, args):

    layers = list(vcd.dic.keys())
    num_concepts = []
    for layer in vcd.dic.keys():
        concepts = len(vcd.dic[layer]['concepts'])
        num_concepts.append(concepts)

    # plot line plot of num_concepts vs. layers
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    plt.plot(layers, num_concepts, label='Num Concepts')
    plt.title('Number of Inter-Concepts vs. layers')
    plt.xlabel('Layers')
    # set y min and max
    plt.ylim(0, 9)

    # set x ticks as labels
    plt.xticks(layers)
    plt.ylabel('Concepts')
    # make legend two columns
    plt.legend(loc='upper right', prop={'size': 8}, ncol=2)
    plt.savefig(os.path.join('results', args.exp_name, 'figures', 'InterNumCon.png'))
    # clear figure
    plt.clf()

    # plt.show()
    return

def plot_intra_clusters_perlayer(vcd, args):

    layers = list(vcd.dic.keys())
    num_concepts = []
    for layer in vcd.dic.keys():
        concepts = vcd.metrics[layer]['num_intra_clusters']
        num_concepts.append(concepts)

    # plot line plot of num_concepts vs. layers
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    plt.plot(layers, num_concepts, label='Num Concepts')
    plt.title('Number of Intra-Concepts vs. layers')
    plt.xlabel('Layers')
    # set y min and max
    plt.ylim(0, 9)
    # set x ticks as labels
    plt.xticks(layers)
    plt.ylabel('Concepts')
    plt.legend(loc='upper right', prop={'size': 8}, ncol=2)
    plt.savefig(os.path.join('results', args.exp_name,'figures',  'IntraNumCon.png'))
    # clear figure
    plt.clf()

    # plt.show()



def plot_temporal_support_perlayer(vcd, args):

    layers = list(vcd.dic.keys())
    num_concepts = []
    for layer in vcd.dic.keys():
        concepts = vcd.metrics[layer]['num_intra_clusters']
        num_concepts.append(concepts)

    # plot line plot of num_concepts vs. layers
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    plt.plot(layers, num_concepts, label='Num Concepts')
    plt.title('Number of Intra-Concepts vs. layers')
    plt.xlabel('Layers')
    # set y min and max
    plt.ylim(0, 9)
    # set x ticks as labels
    plt.xticks(layers)
    plt.ylabel('Concepts')
    plt.legend(loc='upper right', prop={'size': 8})
    plt.savefig(os.path.join('results', args.exp_name, 'figures', 'TemporalSupport.png'))
    # clear figure
    plt.clf()

    # plt.show()
    return

def compute_metrics(vcd):
    # print('Metrics:')
    metric_dict = {}
    for layer in vcd.args.cluster_layer:
        metric_dict[layer] = {}
        # average number of clusters per video
        metric_dict[layer]['num_intra_clusters'] = sum(vcd.num_intra_clusters[layer])/len(vcd.num_intra_clusters[layer])
        metric_dict[layer]['num_intra_clusters_var'] = float(np.var(vcd.num_intra_clusters[layer]))
        metric_dict[layer]['silhouette'] = str(vcd.dic[layer]['silhouette'])
        metric_dict[layer]['num_concepts'] = len(vcd.dic[layer]['concepts'])
        metric_dict[layer]['concepts'] = {}
        # print('')
        # print('Layer: {}'.format(layer))
        # print('Num Intra Clusters: {}'.format(metric_dict[layer]['num_intra_clusters']))
        # print('Sillhouette: {}'.format(vcd.dic[layer]['silhouette']))
        for concept in vcd.dic[layer]['concepts']:
            # print('Concept: {} - Spread: {}'.format(concept, vcd.dic[layer][concept]['spread']))
            metric_dict[layer]['concepts'][concept] = {}
            B, T, H, W = vcd.dic[layer][concept]['video_mask'].shape
            metric_dict[layer]['concepts'][concept]['temporal_support'] = \
                str(((vcd.dic[layer][concept]['video_mask'].sum((2,3))>0).sum(1)/T).mean().item())
            metric_dict[layer]['concepts'][concept]['spatial_support'] = \
                str(((vcd.dic[layer][concept]['video_mask'].sum((2,3)))/(H * W)).mean().item())
            metric_dict[layer]['concepts'][concept]['spread'] = str(vcd.dic[layer][concept]['spread'])

    return metric_dict

def main(args):

    args = prepare_directories(args)
    results_path = os.path.join('results', args.exp_name, 'figures')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # load vcd with pickle
    vcd_path = os.path.join('results', args.exp_name, 'vcd.pkl')
    print('Loading VCD from {}'.format(vcd_path))
    vcd = pickle.load(open(vcd_path, 'rb'))
    args.model = vcd.args.model
    args.cluster_layer = vcd.args.cluster_layer
    args.cluster_subject = vcd.args.cluster_subject
    args.concept_clustering = vcd.args.concept_clustering
    args.cluster_memory = vcd.args.cluster_memory


    # get metrics
    print('Computing metrics')
    if args.use_og_metrics:
        metric_dict = compute_metrics(vcd)
        vcd.metrics = metric_dict

    # save as json file
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as fp:
        json.dump(metric_dict, fp)

    # intra elbow plot per layer
    if args.plot_intra_clusters_perlayer:
        print('Plotting intra clusters per layer')
        plot_intra_clusters_perlayer(vcd, args)

    # inter elbow plot per layer
    if args.plot_num_clusters_perlayer:
        print('Plotting number of clusters per layer')
        plot_num_clusters_perlayer(vcd, args)

    # temporal support plot per layer
    if args.plot_temporal_support_perlayer:
        print('Plotting temporal support per layer')
        plot_temporal_support_perlayer(vcd, args)

    # save concept videos
    if args.save_concept_videos:
        print('Saving concept videos')
        vcd.save_concepts()





    # nxn concept visualization





    # save vcd
    # save_vcd(vcd)





def vcd_args():

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--exp_name', default='times_occ_keys_v1', type=str,help='experiment name (used for saving)')
    # parser.add_argument('--exp_name', default='test', type=str,help='experiment name (used for saving)')
    parser.add_argument('--results_name', default='FeatPerturb', type=str,help='figure name (used for saving)')
    parser.add_argument('--fig_name', default='FeatPerturb', type=str,help='figure name (used for saving)')

    # data
    parser.add_argument('--dataset', default='kubric', type=str,help='dataset to use')
    parser.add_argument('--kubric_path', default='/home/matthewkowal/data/kubcon_v10', type=str,help='kubric path')

    # visualizations use_og_metrics
    parser.add_argument('--use_og_metrics', action='store_false', help='Flag to plot elbow.')
    parser.add_argument('--plot_num_clusters_perlayer', action='store_true', help='Flag to plot elbow.')
    parser.add_argument('--plot_intra_clusters_perlayer', action='store_true', help='Flag to plot elbow.')
    parser.add_argument('--plot_temporal_support_perlayer', action='store_true', help='Flag to plot elbow.')
    parser.add_argument('--save_concept_videos', action='store_true', help='Flag to save concept videos.')

    # computation
    parser.add_argument('--max_num_workers', default=16, type=int,help='Maximum number of workers for clustering')

    # reproducibility
    parser.add_argument('--seed', default=0, type=int,help='seed')

    args = parser.parse_args(sys.argv[1:])

    # random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return args

if __name__ == '__main__':
    start_time = time.time()
    vcd_args = vcd_args()
    main(vcd_args)
    print('Total time in minutes: {:.2f}'.format((time.time()-start_time)/60))