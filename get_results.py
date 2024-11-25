import os
import json
import pickle
import pandas as pd
import networkx as nx
from my_module import *
from cdlib import NodeClustering, evaluation


def get_gt_info(G, ground_truth):
    """
    Provides dictionary of ground-truth community values for size, density, conductance

    :param G: NetworkX graph
    :param ground_truth: NodeClustering object with ground-truth communities
    :return info_dict: dict containing lists of community attribute values
    """
    sizes = [len(com) for com in ground_truth.communities]
    densities = evaluation.internal_edge_density(G, ground_truth, summary=False)
    conductances = evaluation.conductance(G, ground_truth, summary=False)

    info_dict = {
        'size': sizes,
        'density': densities,
        'conductance': conductances
    }
    
    return info_dict

def result_fig(G, ground_truth, cdm, mapping, mapped_metrics):
    print('cdm name', cdm.method_name)
    print('num gt coms', len(ground_truth.communities))
    print('num pred coms', len(cdm.communities))

    # exit()
    metric = 'mF1'
    for metric in ['mFCCN', 'mF1', 'mFCCE']:
        x = [len(gt_com) for gt_com in ground_truth.communities]
        # x = evaluation.internal_edge_density(G, ground_truth, summary=False) # density
        # x = evaluation.conductance(G, ground_truth, summary=False) # conductance

        metric_score = [metrics[metric] for metrics in mapped_metrics]

        fig, ax = plt.subplots()

        a, b = np.polyfit(x, metric_score, 1)

        ax.scatter(x, metric_score, color=colors[0])

        x_fm = [min(x), max(x)]
        y_fm = [a*x_fm[0]+b, a*x_fm[1]+b]
        ax.plot(x_fm, y_fm, '--', color=colors[1], 
            label='FM='+str(round(calc_fairness_metric(x,metric_score), 3)))

        plt.title(f'{cdm.method_name}')
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Community size', fontsize=fontsize_label)
        ax.set_ylabel(metric[1:], fontsize=fontsize_label)        
        ax.tick_params(axis='both', labelsize=fontsize_ticks)
        ax.legend(fontsize=fontsize_ticks)

        save_or_show_fig(f'{cdm.method_name}_{metric}_test.png', True)
    exit()




def get_metric_scores(G, ground_truth, cdm):
    """
    Returns dictionary with metric scores for several mapped/global metrics

    :param G: NetworkX graph
    :param ground_truth: NodeClustering object containing the ground-truth communities
    :param cdm: NodeClustering object of the CDM's community assignments
    :return metric_results: dictionary of the metric scores
    """
    pred_coms = cdm.communities

    # mapping is sorted by gt coms: 0, ..., |C|-1
    mapping = iterative_mapping(ground_truth, cdm)

    mapped_metrics = []
    for gt, pred in mapping:
        if pred != None:
            pred_com = cdm.communities[pred]
        else:
            pred_com = None
        scores = calc_mapped_scores(G, ground_truth.communities[gt], pred_com)
        mapped_metrics.append(scores)

    # result_fig(G, ground_truth, cdm, mapping, mapped_metrics)

    global_metrics = [calc_global_scores(G, gt_com, pred_coms) 
        for gt_com in ground_truth.communities]

    metric_scores = {
        'mFCCNs': [metrics['mFCCN'] for metrics in mapped_metrics],
        'mF1s': [metrics['mF1'] for metrics in mapped_metrics],
        'mFCCEs': [metrics['mFCCE'] for metrics in mapped_metrics],
        'mFCCEps': [metrics['mFCCE+'] for metrics in mapped_metrics],

        'AvgF1s': [metrics['AvgF1'] for metrics in global_metrics],
        'SWF1s': [metrics['SWF1'] for metrics in global_metrics],
        'SWFCCEs': [metrics['SWFCCE'] for metrics in global_metrics],
        'SWFCCEps': [metrics['SWFCCE+'] for metrics in global_metrics]
    }
    return metric_scores


def get_fm_results(metric_scores, x, ground_truth, cdm):
    """
    Returns dictionary with fairness metric scores for several mapped/global metrics

    :param metric_scores: Dictionary containing metric scores
    :param x: attribute values of communities
    :return fm_results: dictionary of the fairness metric scores
    """
    fm_scores = {
        'mFCCN': calc_fairness_metric(x, metric_scores['mFCCNs']),
        'mF1': calc_fairness_metric(x, metric_scores['mF1s']),
        'mFCCE': calc_fairness_metric(x, metric_scores['mFCCEs']),
        'mFCCE+': calc_fairness_metric(x, metric_scores['mFCCEps']),

        'AvgF1': calc_fairness_metric(x, metric_scores['AvgF1s']),
        'SWF1': calc_fairness_metric(x, metric_scores['SWF1s']),
        'SWFCCE': calc_fairness_metric(x, metric_scores['SWFCCEs']),
        'SWFCCE+': calc_fairness_metric(x, metric_scores['SWFCCEps']),

        'nmi': ground_truth.normalized_mutual_information(cdm).score,
        'ari': ground_truth.adjusted_rand_index(cdm).score,
        'vi': ground_truth.variation_of_information(cdm).score,
        'f1': cdm.f1(ground_truth).score,       # old: ground_truth.f1(cdm).score,
        'nf1': cdm.nf1(ground_truth).score,     # old: ground_truth.nf1(cdm).score,
        'rmi': ground_truth.rmi(cdm, norm_type='normalized').score
    }
    return fm_scores

def result_hub(data_path, net_name, res_path):
    """
    Collects results and puts them in result_dict to be stored
    Results are gathered for all communities and stored once per network

    :param data_path: path to where network data is stored
    :param net_name: name of network
    :param res_path: path to where results are going to be stored
    """
    print(f'\nNetwork: {net_name}')

    G, node_clustering_dict = get_network_communities(data_path, net_name)
    
    ground_truth = node_clustering_dict['ground_truth']
    gt_coms = node_clustering_dict['ground_truth'].communities
    gt_info = get_gt_info(G, ground_truth)

    # print_network_info(G, net_name, gt_coms)    

    attributes = ['size', 'density', 'conductance']

    all_result_dict = {
        'size': {},
        'density': {},
        'conductance': {},
    }

    for cdm_name in node_clustering_dict:
        if cdm_name == 'ground_truth':
            continue
        cdm = node_clustering_dict[cdm_name]
        # cdm = node_clustering_dict['Louvain']

        if not cdm:
            # invalid method
            fm_scores = {
                'mFCCN': None,
                'mF1': None,
                'mFCCE': None,
                'mFCCE+': None,
                'AvgF1': None,
                'SWF1': None,
                'SWFCCE': None,
                'SWFCCE+': None,

                'nmi': None,
                'ari': None,
                'vi': None,
                'f1': None,
                'nf1': None,
                'rmi': None,
            }
            for attr in attributes:
                all_result_dict[attr][cdm_name] = fm_scores
        else:
            metric_scores = get_metric_scores(G, ground_truth, cdm)

            for attr in attributes:
                x = np.array(gt_info[attr])
                fm_scores = get_fm_results(metric_scores, x, ground_truth, cdm)
                all_result_dict[attr][cdm_name] = fm_scores

    show = False
    store = True
    for attr in attributes:
        filename = f'{res_path}/res_{attr[:4]}_{net_name}.pkl'
        if show:
            print(attr)
            print(json.dumps(all_result_dict[attr], indent=4))
        if store:
            print(f'saving {filename}')
            with open(filename, 'wb') as handle:
                pickle.dump(all_result_dict[attr], handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print(f'NOT saving {filename}')


if __name__ == '__main__':
    p = Parser()
    dir_name = p.directory
    net_name = p.network
    dir_path_input = path_prefix(dir_name, 'data_applied_methods')
    res_path = f'results/{remove_prefix(dir_name, "data_applied_methods/")}'

    print('start get_results.py')
    print(f'dir_path_input = {dir_path_input}, res_path = {res_path}, net_name = {net_name}')
    
    if not os.path.exists(dir_path_input):
        print('Directory does not exist:', dir_path_input)
        print('Did you forget "data_applied_methods"?')
        exit()

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    if net_name == None:
        # get all network names in dir_path_input        
        files = os.listdir(dir_path_input)
        net_names = sorted(list(set([file[:-10] for file in files 
            if file[-10:] == '_nodes.csv'])), reverse=False)

        # net_names = [name for name in net_names if 'xi2' in name]
        # net_names = net_names[3:]

        print('Networks run:', net_names)
        for net_name in net_names:
            result_hub(dir_path_input, net_name, res_path)
    else:
        # use only the provided net_name (using -n ...)
        result_hub(dir_path_input, net_name, res_path)