import os
import sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cdlib import NodeClustering, evaluation

import matplotlib as mpl
from matplotlib.patches import Arc

sys.path.append('../..')
from my_module import *

def create_fairness_metric_figure(save_fig=True):
    """
    example figure showing the least squares approximation, which is used to
      calculate the fairness metric
    """
    fig, ax = plt.subplots()

    fontsize_ticks = 20
    fontsize_label = 22
    fontsize_annot = fontsize_label + 4

    x = np.array(range(11))/10
    # y = [0.2, 0.4, 0.15, 0.3, 0.55, 0.45, 0.65, 0.8, 0.7, 0.9, 0.8]
    # y = [0.1, 0.2, 0.15, 0.2, 0.6, 0.7, 0.6, 0.9, 0.96, 0.9, 0.88]
    y = [0.3]*11
    y.reverse()

    a, b = np.polyfit(x, y, 1)

    x_slope = np.array([min(x), max(x)])
    y_slope = a * x_slope + b

    x0 = (x_slope[0], y_slope[0])
    x1 = (x_slope[1], y_slope[0])
    y0 = (x_slope[1], y_slope[0])
    y1 = (x_slope[1], y_slope[1])

    # ax.annotate('', xy=x0, xytext=x1, arrowprops={
    #     'arrowstyle': '<|-|>, head_width=.45, head_length=.8', 'edgecolor':colors[2],
    #     'facecolor':colors[2], 'linewidth':2, 'shrinkA':0, 'shrinkB':0})
    # ax.annotate('$\\Delta_x$', xy=((x1[0]-x0[0])/2+x0[0], x1[1]-0.07), ha='center', va='center',
    #     fontsize=fontsize_annot)

    # ax.annotate('', xy=y1, xytext=y0, arrowprops={
    #     'arrowstyle': '<|-|>, head_width=.45, head_length=.8', 'edgecolor':colors[2],
    #     'facecolor':colors[2], 'linewidth':2, 'shrinkA':0, 'shrinkB':0})
    # ax.annotate('$\\Delta_y$', xy=(y0[0]+0.052, (y1[1]-y0[1])/2+y0[1]), ha='center', va='center',
    #     fontsize=fontsize_annot)

    # center = x0
    # theta2 = np.arctan(a)*180/np.pi
    # ax.add_patch(Arc((center), width=0.3, height=0.3, angle=0, theta1=0, theta2=theta2))

    # ax.annotate('$\\theta$', xy=(0.18, 0.24), ha='center', va='center',
    #     fontsize=fontsize_annot)


    ax.plot(x, y, 'o', color=colors[0], markersize=marker_sizes[0]-2)
    ax.plot(x_slope, y_slope, color=colors[1], linestyle=linestyles[0], linewidth=2, 
        label='Linear least squares approximation')

    # ax.set_title('Showing Score Against Community Size')
    ax.set_xlabel('Normalized community size', fontsize=fontsize_label)
    ax.set_ylabel('Detection Score', fontsize=fontsize_label)
    ax.set_ylim(0,1.05)
    ax.set_xlim(-0.08,1.1)
    # ax.set_xticks(x)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)

    ax.legend(loc='upper left', fontsize=17)

    print('delta_y:', (y1[1]-y0[1]))
    print('fairness:', calc_fairness_metric(x, y))

    figure_str = 'presentation/angle_equal2.png'
    save_or_show_fig(figure_str, True)


def create_per_community_figure(save_fig=True):
    """
    example figure showing the scores per community
    """
    fig, ax = plt.subplots()

    fontsize_ticks = 20
    fontsize_label = 22
    fontsize_annot = fontsize_label + 4

    x = np.array(range(11))/10
    y = [0.2, 0.4, 0.15, 0.3, 0.55, 0.45, 0.65, 0.8, 0.7, 0.9, 0.8]
    # y = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    # ax.set_xticks([0, 30, 60, 90, 120])

    ax.plot(x, y, 'o', color=colors[0], markersize=marker_sizes[0]-2)

    # ax.set_title('Showing Score Against Community Size')
    ax.set_xlabel('Normalized community size', fontsize=fontsize_label)
    ax.set_ylabel('Detection Score', fontsize=fontsize_label)
    ax.set_ylim(0,1.05)
    ax.set_xlim(-0.08,1.1)
    
    # ax.set_xticks(x)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)

    figure_str = 'presentation/normalized.png'
    save_or_show_fig(figure_str, True)


def create_correlation_matrix():
    for attr in ['size', 'dens', 'cond']:
        with open(f'../../results/synthetic/LFR/res_{attr}_SG_0.pkl', 'rb') as handle:
            result_dict = pickle.load(handle)

        fm_metrics = [key for key in result_dict[list(result_dict.keys())[0]] if 'fm' in key]
        
        fm_dict = {fm_metric: [] for fm_metric in fm_metrics}

        for key in result_dict.keys():
            for fm_metric in result_dict[key]:
                if fm_metric in all_acc_measures:
                    continue
                fm_dict[fm_metric].append(result_dict[key][fm_metric])

        df = pd.DataFrame.from_dict(fm_dict)
        corr = df.corr(method='pearson')

        sns.heatmap(corr, cmap="magma", annot=True)
        plt.show()

        # fig, ax = plt.subplots()
        # plt.scatter(df['fm_mFCCN'], df['fm_AvgF1'])
        # plt.show()

        # plt.matshow(corr)
        # cb = plt.colorbar()
        # cb.ax.tick_params(labelsize=14)

        # plt.xticks(range(len(fm_metrics)), fm_metrics, rotation=45, ha='left')
        # plt.yticks(range(len(fm_metrics)), fm_metrics)
        # plt.show()



# def create_correlation_matrix(attr, res_path, file_names):
#     result_dict_list = combine_results(file_names, res_path)

#     fm_metrics = [metric for metric in result_dict_list[0][list(result_dict_list[0].keys())[0]] if 'fm' in metric]
#     fm_dict = {fm_metric: [] for fm_metric in fm_metrics}

#     for result_dict in result_dict_list:
#         for cdm in result_dict.keys():
#             for fm_metric in fm_metrics:
#                 fm_dict[fm_metric].append(result_dict[cdm][fm_metric])

#     df = pd.DataFrame.from_dict(fm_dict)
#     corr = df.corr(method='pearson')

#     sns.heatmap(corr, cmap="magma", annot=True)

#     fig_str = f'figures/thesis_text/corr_matrix_{attr[:4]}'
#     save_or_show_fig(fig_str, save_figure)

#     # fig, ax = plt.subplots()
#     # plt.scatter(df['fm_mFCCN'], df['fm_AvgF1'])
#     # plt.show()

#     # plt.matshow(corr)
#     # cb = plt.colorbar()
#     # cb.ax.tick_params(labelsize=14)

#     # plt.xticks(range(len(fm_metrics)), fm_metrics, rotation=45, ha='left')
#     # plt.yticks(range(len(fm_metrics)), fm_metrics)
#     # plt.show()


def create_different_mu_fig():
    result_path = '../../results/synthetic/LFR_0_to_1/'

    files = os.listdir(result_path)
    
    files = sorted([file for file in files if 'size' in file])
    files.append(files.pop(0))

    mus = [f[f.find('mu')+2:] for f in files]
    mus = [float(mu[:mu.find('_')])/10 for mu in mus]

    leiden_nmi = []
    louvain_nmi = []
    node2vec_nmi = []
    walktrap_nmi = []

    for file in files:
        with open(f'{result_path}{file}', 'rb') as handle:
            result_dict = pickle.load(handle)

        leiden_nmi.append(result_dict['Leiden']['nmi'])
        louvain_nmi.append(result_dict['Louvain']['nmi'])
        node2vec_nmi.append(result_dict['Node2Vec']['nmi'])
        walktrap_nmi.append(result_dict['Walktrap']['nmi'])

    nmi_scores = [leiden_nmi, louvain_nmi, node2vec_nmi, walktrap_nmi]

    fig, ax = plt.subplots()

    fontsize_ticks = 20
    fontsize_label = 22

    for i in range(4):
        ax.plot(mus, nmi_scores[i], marker=markers[i], markersize=marker_sizes[i]-2, 
            color=colors[i])

    ax.set_ylabel('NMI', fontsize=fontsize_label)
    ax.set_xlabel('Mixing Parameter $\\mu$', fontsize=fontsize_label)
    ax.tick_params(axis='both', labelsize=fontsize_ticks)
    plt.xlim(0, 1)

    plt.legend(labels=['Leiden', 'Louvain', 'Node2Vec', 'Walktrap'], fontsize=16)
    save_or_show_fig('fig/mixing_param_example', save_fig=False)


def property_correlation_matrix():

    network_dirs = ['LFR_large_246', 'ABCD', 'HICH_BA/multiple_maj', 'HICH_BA/multiple_maj']
    # network_dirs = ['HICH_BA/multiple_maj', 'HICH_BA/multiple_min']

    for network_type in network_dirs:
        data_path = '../../data/synthetic/'+network_type+'/'

        files = os.listdir(data_path)

        all_net_names = sorted(list(set([file[:-10] for file in files 
            if file[-10:] == '_nodes.csv'])), reverse=False)

        if network_type == 'LFR_large_246':
            mp = 'mu'
        elif network_type == 'ABCD':
            mp = 'xi'
        else:
            network_type = 'HICH_BA_'+network_type[-12:]
            print(network_type)

        for mp_value in [2,4,6]:
            if 'HICH' in network_type:
                net_names = all_net_names
                print(net_names)
            else:
                net_names = [net_name for net_name in all_net_names if mp+str(mp_value) in net_name]
            
            property_dict = {'Size': [], 'Density': [], 'Conductance': []}

            for net_name in net_names:
                G, node_clustering_dict = get_network_communities(data_path, net_name)
                ground_truth = node_clustering_dict['ground_truth']

                sizes = [len(com) for com in ground_truth.communities]
                densities = evaluation.internal_edge_density(G, ground_truth, summary=False)
                conductances = evaluation.conductance(G, ground_truth, summary=False)

                property_dict['Size'] += sizes
                property_dict['Density'] += densities
                property_dict['Conductance'] += conductances

                # break

            df = pd.DataFrame.from_dict(property_dict)
            corr = df.corr(method='pearson')

            sns.set(font_scale=1.8)
            sns.heatmap(corr, cmap='RdBu', annot=True, square=True, linewidth=1, annot_kws={'size': 28},
                vmin=-1, vmax=1, cbar=False)

            # fig, ax = plt.subplots()
            # plt.scatter(df['Size'], df['Conductance'])
            # plt.show()


            # plt.show()
            
            

            if 'HICH' in network_type:
                save_or_show_fig('fig/'+'corr_'+network_type, True)
                break
            else:
                save_or_show_fig('fig/'+'corr_'+network_type+'_'+mp+str(mp_value), True)

        # exit()

def create_color_bar():
    fig, ax = plt.subplots(figsize=(1, 4))
    vmin, vmax =  -1, 1
    cmap = mpl.colormaps['RdBu']
    norm = mpl.colors.Normalize(-1, 1)
    ax.tick_params(labelsize=18)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), ax)
    # ax.set_yticks([-1, -0.66, -0.33, 0, 0.33, 0.66, 1])


    plt.tight_layout()
    save_or_show_fig('fig/color_bar', True)


def get_network_values(G, gt_com):
    n = len(G.nodes)
    edges = len(G.edges)
    avg_deg = np.average([G.degree[n] for n in G.nodes])
    highest_deg = max([G.degree[n] for n in G.nodes])
    lowest_deg = min([G.degree[n] for n in G.nodes])
    num_coms = len(gt_com)
    largest_com = max([len(c) for c in gt_com])
    smallest_com = min([len(c) for c in gt_com])

    values = {
        'n': n, 
        'edges': edges,
        'avg_deg': avg_deg,
        'highest_deg': highest_deg, 
        'lowest_deg': lowest_deg, 
        'num_coms': num_coms,
        'largest_com': largest_com,
        'smallest_com': smallest_com
    }
    return values

def get_synthetic_network_info():
    network_dirs = ['LFR_large_246', 'ABCD', 'HICH_BA/multiple_maj', 'HICH_BA/multiple_maj']
    # network_dirs = ['HICH_BA/multiple_maj', 'HICH_BA/multiple_min']

    for network_type in network_dirs:
        print(network_type)

        data_path = '../../data/synthetic/'+network_type+'/'
        files = os.listdir(data_path)

        net_names = sorted(list(set([file[:-10] for file in files 
            if file[-10:] == '_nodes.csv'])), reverse=False)

        columns = ['n', 'edges', 'avg_deg', 'highest_deg', 'lowest_deg',
            'num_coms', 'largest_com', 'smallest_com']
        all_values = {col: [] for col in columns}

        for net_name in net_names:
            G, node_clustering_dict = get_network_communities(data_path, net_name)
            ground_truth = node_clustering_dict['ground_truth'].communities

            values = get_network_values(G, ground_truth)
            for col, value in values.items():
                all_values[col].append(value)

        print_dict(all_values)
        avg_values = {col: np.mean(all_values[col]) for col in all_values.keys()}
        print(avg_values)
    # exit()




if __name__ == '__main__':
    create_fairness_metric_figure()
    # create_per_community_figure()

    # property_correlation_matrix()
    # get_synthetic_network_info()
    # create_color_bar()

  

    # create_correlation_matrix()
    # create_different_mu_fig()
