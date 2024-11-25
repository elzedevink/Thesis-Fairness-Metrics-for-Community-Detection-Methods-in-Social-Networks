import os
import sys
import json
import random
import numpy as np
import networkx as nx
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


from cdlib import NodeClustering, evaluation

# from create_figures import *
sys.path.append('../..')
from my_module import *
# from experiment import *
fontsize_ticks = 20 # 24
fontsize_label = 20 # 26

def mapped_assignments(G, dict_coms, step=50):
    n = len(dict_coms['ground_truth'].communities[0])
    assignment = list(range(n))
    num_wrong = 0

    dict_coms[num_wrong] = deepcopy(assignment)
    while len(assignment) >= step:
        kill_idx = random.sample(list(range(len(assignment))), step)

        for idx in sorted(kill_idx, reverse=True):
            del assignment[idx]

        num_wrong = n - len(assignment)
        
        dict_coms[num_wrong] = deepcopy(assignment)

    return dict_coms

def min_maj_assignment(G, dict_coms, step=5):
    gt_com0 = set(dict_coms['ground_truth'].communities[0])
    gt_com1 = set(dict_coms['ground_truth'].communities[1])
    
    already_chosen0 = set([])
    already_chosen1 = set([])
    
    assignment = [list(gt_com0), list(gt_com1)]
    dict_coms[str(0)] = NodeClustering(assignment, graph=G, 
        method_name=str(0))

    for i in range(int(len(gt_com1)/step)):
        left0 = gt_com0-already_chosen0
        left1 = gt_com1-already_chosen1

        sample0 = random.sample(left0, k=step)
        sample1 = random.sample(left1, k=step)

        already_chosen0.update(sample0)
        already_chosen1.update(sample1)

        left0 = gt_com0-already_chosen0
        left1 = gt_com1-already_chosen1
        
        left0.update(already_chosen1)
        left1.update(already_chosen0)

        pred_com0 = list(left0)
        pred_com1 = list(left1)

        assignment = [pred_com0, pred_com1]
        dict_coms[str((i+1)*step)] = NodeClustering(assignment, 
            graph=G, method_name=str((i+1)*step))
    
    return dict_coms


def global_evenly_assignments(G, dict_coms):
    n = len(dict_coms['ground_truth'].communities[0])
    
    assignment = [list(range(n))]
    dict_coms[str(len(assignment))] = NodeClustering(deepcopy(assignment), 
        graph=G, method_name=str(len(assignment)))

    while len(assignment[0]) > 32:
        new_assignment = []
        num_split = int(len(assignment[0])/2)

        for comm in assignment:
            split0 = random.sample(comm, num_split)
            split1 = list(set(comm) - set(split0))

            new_assignment.append(split0)
            new_assignment.append(split1)
        assignment = new_assignment

        dict_coms[str(len(assignment))] = NodeClustering(deepcopy(assignment), 
            graph=G, method_name=str(len(assignment)))

    return dict_coms


def global_constant_assignments(G, dict_coms):
    n = len(dict_coms['ground_truth'].communities[0])
    
    assignment = [list(range(n))]
    dict_coms[str([n])] = NodeClustering(deepcopy(assignment), 
        graph=G, method_name=str([n]))

    cut_off = 100

    for i in range(9):
        # get ids 
        cut = random.sample(list(range(len(assignment[0]))), cut_off)
        add = [assignment[0][c] for c in cut]

        for idx in sorted(cut, reverse=True):
            del assignment[0][idx]
        assignment.append(add)

        name = str([len(comm) for comm in assignment])

        dict_coms[name] = NodeClustering(deepcopy(assignment), 
            graph=G, method_name=name)

    return dict_coms

def fairness_case_assignments():
    gt_coms = [list(range(0,100)), list(range(100,300)), list(range(300,600)), 
        list(range(600,1000))]
    G = nx.Graph()
    G.add_nodes_from(list(range(1000)))

    step = 50
    
    assignments = deepcopy(gt_coms)
    wrong = 0
    name = str(wrong)+' wrong'

    dict_coms = {}
    dict_coms['ground_truth'] = gt_coms
    dict_coms[name] = deepcopy(assignments)

    for _ in range(int(100/step)):

        for com in range(len(assignments)):
            # print('before', assignments[com])
            for i in reversed(range(step)):
                del assignments[com][i]
            # print('after', assignments[com], '\n')
            
        wrong += step
        name = str(wrong)+' wrong'
        dict_coms[name] = deepcopy(assignments)
    
    return G, dict_coms

def mapped_metric_figure():
    print('mapped metric figure')
    G, dict_coms = get_network_communities(dir_path='method_behavior', 
        net_name='metric_fig_graph')
    repetitions = 20
    
    mF1s = []
    mFCCNs = []
    mFCCE_total = []
    mFCCEp_total = []
    for i in range(repetitions):
        # multiple iterations for fcce, fcce+
        dict_coms = mapped_assignments(G, dict_coms, step=64)
        gt_com = dict_coms['ground_truth'].communities[0]

        mFCCEs = []
        mFCCEps = []
        for key in dict_coms.keys():
            if key == 'ground_truth':
                continue
            pred_com = dict_coms[key]
            mapped_scores = calc_mapped_scores(G, gt_com, pred_com)
            
            if i == 0:
                mF1s.append(mapped_scores['mF1'])
                mFCCNs.append(mapped_scores['mFCCN'])

            mFCCEs.append(mapped_scores['mFCCE'])
            mFCCEps.append(mapped_scores['mFCCE+'])

        mFCCE_total.append(mFCCEs)
        mFCCEp_total.append(mFCCEps)

    length = len(mFCCE_total[0])
    highest_mFCCE = [0] * length
    lowest_mFCCE = [0] * length
    avg_mFCCE = [0] * length

    highest_mFCCEp = [0] * length
    lowest_mFCCEp = [0] * length
    avg_mFCCEp = [0] * length

    for i in range(length):
        highest_mFCCE[i] = max([run[i] for run in mFCCE_total])
        lowest_mFCCE[i] = min([run[i] for run in mFCCE_total])
        avg_mFCCE[i] = np.mean([run[i] for run in mFCCE_total])

        highest_mFCCEp[i] = max([run[i] for run in mFCCEp_total])
        lowest_mFCCEp[i] = min([run[i] for run in mFCCEp_total])
        avg_mFCCEp[i] = np.mean([run[i] for run in mFCCEp_total])
    
    fig, ax = plt.subplots()

    x = list(dict_coms.keys())[1:]
    alpha = 1

    metrics = [mFCCNs, mF1s, avg_mFCCE, avg_mFCCEp]
    names = ['FCCN', 'F1', 'FCCE', 'FCCE+']
    for i in range(len(metrics)):
        ax.plot(x, metrics[i], color=colors[i], marker=markers[i], 
            markersize=marker_sizes[i], linestyle='-', alpha=alpha, label=names[i])
    ax.fill_between(x, highest_mFCCE, lowest_mFCCE, color=colors[2], alpha=0.2)
    ax.fill_between(x, highest_mFCCEp, lowest_mFCCEp, color=colors[3], alpha=0.2)

    ax.set_xlabel('Number of wrongly predicted nodes', fontsize=fontsize_label)
    ax.set_ylabel('CPM Score', fontsize=fontsize_label)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xticks([0, 200, 400, 600, 800, 1000], fontsize=fontsize_ticks)

    ax.set_ylim(0, 1.05)

    ax.legend(loc='upper right', fontsize=14)

    figure_str = 'fig/behavior_mapped_metrics.png'
    save_or_show_fig(figure_str, True)

def reform_scores_dict(scores_dicts, repetitions):
    scores_dict = {0: [], 1: []}
    std_dict = {0: [], 1: []}

    num_x_points = len(scores_dicts[0][0])

    for gt in [0, 1]:
        for x_point in range(num_x_points):
            FCCEs = [scores_dicts[i][gt][x_point]['mFCCE'] for i in range(repetitions)]
            Avg_FCCEs = np.mean(FCCEs)
            FCCEps = [scores_dicts[i][gt][x_point]['mFCCE+'] for i in range(repetitions)]
            Avg_FCCEps = np.mean(FCCEps)

            scores = {
                'mFCCN': scores_dicts[0][gt][x_point]['mFCCN'], 
                'mF1': scores_dicts[0][gt][x_point]['mF1'], 
                'mFCCE': Avg_FCCEs, 
                'mFCCE+': Avg_FCCEps  
            }
            scores_dict[gt].append(scores)

            stdFCCE = np.std(FCCEs)
            stdFCCEp = np.std(FCCEps)
            std_dict[gt].append({'mFCCE': stdFCCE, 'mFCCE+': stdFCCEp})

    return scores_dict, std_dict

def fairness_metric_min_maj(scores_dict, std_dict, x):
    x_size = [70, 40]

    metrics = ['mFCCN', 'mF1', 'mFCCE']#, 'mFCCE+']

    # y_dict = [{0: 0, 1: 0} for _ in range(len(x))]

    num_x_points = len(scores_dict[0])

    fig, ax = plt.subplots()

    for m_idx in range(len(metrics)):
        FMs = [calc_fairness_metric(x_size, 
            [scores_dict[0][i][metrics[m_idx]], scores_dict[1][i][metrics[m_idx]]]) 
            for i in range(num_x_points)]

        ax.plot(x, FMs, marker=markers[m_idx], markersize=marker_sizes[m_idx]-2, 
                color=colors[m_idx], label=f'$\\Phi^{{{metrics[m_idx][1:]}}}_{{size}}$')    


    high = [calc_fairness_metric(x_size, 
            [scores_dict[0][i][metrics[m_idx]]+std_dict[0][i][metrics[m_idx]], 
            scores_dict[1][i][metrics[m_idx]]+std_dict[1][i][metrics[m_idx]]]) 
            for i in range(num_x_points)]
    low = [calc_fairness_metric(x_size, 
            [scores_dict[0][i][metrics[m_idx]]-std_dict[0][i][metrics[m_idx]], 
            scores_dict[1][i][metrics[m_idx]]-std_dict[1][i][metrics[m_idx]]]) 
            for i in range(num_x_points)]
    ax.fill_between(x, high, low, color=colors[2], alpha=0.2)


    ax.plot([-1, 2], [0,0], '--', color='k', linewidth=1)
    ax.set_xlabel('Fraction of minority nodes\nswapped', fontsize=fontsize_label)
    ax.set_ylabel('$\\Phi^{{F^*}}_{{size}}$', fontsize=fontsize_label)
    ax.set_ylim(-.45, .45)
    ax.set_xlim(-.05, 1.05)
    plt.xticks([0, 0.25, 0.5, 0.75, 1], fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    plt.legend(fontsize=fontsize_ticks)

    figure_str = 'fig/behavior_fairness_min_maj.png'
    save_or_show_fig(figure_str, True)

    # exit()


def mapped_metric_figure_min_maj():
    print('mapped metric figure - minority/majority')
    G, dict_coms = get_network_communities(dir_path='method_behavior', 
        net_name='method_behavior_110')

    # print_network_info(G, 'net_name')
    # exit()

    repetitions = 20
    alpha = 1
    
    scores_dicts = []
    for i in range(repetitions):
        # multiple iterations for fcce, fcce+
        dict_coms = min_maj_assignment(G, dict_coms, step=2) #here

        gt_com = dict_coms['ground_truth']

        scores_dict = {0: [], 1: []}

        for prediction in list(dict_coms.keys())[1:]: # skip gt
            mapping = iterative_mapping(gt_com, dict_coms[prediction])

            for gt, pred in mapping:
                pred_com = dict_coms[prediction].communities[pred]
                scores = calc_mapped_scores(G, gt_com.communities[gt], pred_com)

                scores_dict[gt].append(scores)

        scores_dicts.append(scores_dict)

    scores_dict, std_dict = reform_scores_dict(scores_dicts, repetitions)

    x = [float(switch)/40 for switch in list(dict_coms.keys())[1:]]

    fairness_metric_min_maj(scores_dict, std_dict, x)

    fig, ax = plt.subplots()

    metrics = ['mFCCN', 'mF1', 'mFCCE']#, 'mFCCE+']

    # labeling purposes
    for m_idx in range(len(metrics)):
        plt.plot([0], [1], label=metrics[m_idx][1:], marker=markers[m_idx], 
            markersize=marker_sizes[m_idx]-2, markeredgecolor='k', 
            markerfacecolor='w', color='k', linestyle='-')
    patch = mpatches.Patch(color=colors[0], label='Majority')
    patch2 = mpatches.Patch(color=colors[1], label='Minority')

    num_x_points = len(scores_dict[0])

    for gt in scores_dict.keys():
        for gt_idx in range(len(metrics)):
            values = [scores_dict[gt][score_idx][metrics[gt_idx]] for score_idx in range(num_x_points)]

            ax.plot(x, values, marker=markers[gt_idx], markersize=marker_sizes[gt_idx]-2, 
                color=colors[gt], alpha=alpha)#, markeredgecolor='k', )

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([patch, patch2])

    plt.legend(handles=handles, fontsize=16)

    ax.set_xlabel('Fraction of minority nodes\nswapped', fontsize=fontsize_label)
    ax.set_ylabel('CPM Score', fontsize=fontsize_label)
    ax.set_ylim(0, 1.05)
    plt.xticks([0, 0.25, 0.5, 0.75, 1], fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    hatches = ['', ''] #['x', '+']
    fill_metrics = ['mFCCE']#, 'mFCCE+']
    for gt in scores_dict.keys():
        for m_idx in range(len(fill_metrics)):
            high = [scores_dict[gt][i][fill_metrics[m_idx]] + std_dict[gt][i][fill_metrics[m_idx]] 
                for i in range(num_x_points)]
            low = [scores_dict[gt][i][fill_metrics[m_idx]] - std_dict[gt][i][fill_metrics[m_idx]] 
                for i in range(num_x_points)]
            ax.fill_between(x, high, low, color=colors[gt], alpha=0.2,
                hatch=hatches[m_idx])
            # break

    figure_str = 'fig/behavior_mapped_min_maj.png'
    save_or_show_fig(figure_str, True)

    return


def global_metric_figure(variant):
    print('global metric figure')
    G, dict_coms = get_network_communities(dir_path='method_behavior', 
        net_name='metric_fig_graph')
    repetitions = 20

    AF1s = []
    SWF1s = []
    SWFCCE_total = []
    SWFCCEp_total = []

    for i in range(repetitions):

        if variant == 'constant':
            dict_coms = global_constant_assignments(G, dict_coms)
        elif variant == 'evenly':
            dict_coms = global_evenly_assignments(G, dict_coms)
        else: 
            print('Error: give proper global metric figure variant')
            exit()

        gt_com = dict_coms['ground_truth'].communities[0]

        SWFCCEs = []
        SWFCCEps = []
        for split in list(dict_coms.keys())[1:]:
            pred_coms = dict_coms[split].communities

            overlaps = [len(set(gt_com) & set(pred_com)) for pred_com in pred_coms]
            pred_lens = [len(pred_com) for pred_com in pred_coms]
            gt_size = len(gt_com)

            if i == 0:
                AF1s.append(calc_AF1(overlaps, gt_size, pred_lens))
                SWF1s.append(calc_SWF1(overlaps, gt_size, pred_lens))
            
            SWFCCEs.append(calc_SWFCCE(G, gt_com, pred_coms, overlaps, gt_size, 'FCCE'))
            SWFCCEps.append(calc_SWFCCE(G, gt_com, pred_coms, overlaps, gt_size, 'FCCE+'))

        SWFCCE_total.append(SWFCCEs)
        SWFCCEp_total.append(SWFCCEps)

    length = len(SWFCCE_total[0])
    highest_SWFCCE = [0] * length
    lowest_SWFCCE = [0] * length
    avg_SWFCCE = [0] * length

    highest_SWFCCEp = [0] * length
    lowest_SWFCCEp = [0] * length
    avg_SWFCCEp = [0] * length

    for i in range(length):
        highest_SWFCCE[i] = max([run[i] for run in SWFCCE_total])
        lowest_SWFCCE[i] = min([run[i] for run in SWFCCE_total])
        avg_SWFCCE[i] = np.mean([run[i] for run in SWFCCE_total])

        highest_SWFCCEp[i] = max([run[i] for run in SWFCCEp_total])
        lowest_SWFCCEp[i] = min([run[i] for run in SWFCCEp_total])
        avg_SWFCCEp[i] = np.mean([run[i] for run in SWFCCEp_total])

    fig, ax = plt.subplots()
    x = range(len(dict_coms.keys())-1)
    if variant == 'constant':
        labels = ['[[1024]]', '[[924], [100]]', '[[824], 2x[100]]', '[[724], 3x[100]]', 
            '[[624], 4x[100]]', '[[524], 5x[100]]', '[[424], 6x[100]]', '[[324], 7x[100]]', 
            '[[224], 8x[100]]', '[[124], 9x[100]]']
        plt.xticks(x, labels=labels, fontsize=fontsize_ticks, rotation=90, ha='center')
    else:
        labels = ['[[1024]]', '[2x[512]]', '[4x[256]]', '[8x[128]]', '[16x[64]]', 
            '[32x[32]]']
        plt.xticks(x, labels=labels, fontsize=fontsize_ticks-3)
    
    if variant == 'evenly':
        ax.plot(x, SWF1s, color=colors[1], marker=markers[1], linestyle='-', alpha=1, 
            markersize=marker_sizes[1]+2, label='SWF1')
    else:
        ax.plot(x, SWF1s, color=colors[1], marker=markers[1], linestyle='-', alpha=1, 
            markersize=marker_sizes[1], label='SWF1')
    ax.plot(x, avg_SWFCCE, color=colors[2], marker=markers[2], linestyle='-', alpha=1, 
        markersize=marker_sizes[2], label='SWFCCE')
    ax.plot(x, avg_SWFCCEp, color=colors[3], marker=markers[3], linestyle='-', alpha=1, 
        markersize=marker_sizes[3], label='SWFCCE+')
    ax.plot(x, AF1s, color=colors[0], marker=markers[0], linestyle='-', alpha=1, 
        markersize=marker_sizes[0], label='AvgF1')
    ax.fill_between(x, highest_SWFCCE, lowest_SWFCCE, color=colors[2], alpha=0.2)
    ax.fill_between(x, highest_SWFCCEp, lowest_SWFCCEp, color=colors[3], alpha=0.2)

    plt.yticks(fontsize=fontsize_ticks)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('CPM Score', fontsize=fontsize_label)
    ax.set_xlabel('Global prediction shape', fontsize=fontsize_label)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3,0,1,2]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
        loc='upper right', fontsize=fontsize_ticks-2)

    figure_str = f'fig/behavior_global_{variant}_metrics.png'
    save_or_show_fig(figure_str, True)

def fairness_figure_simple():
    print('fairness score simple')
    save = True

    x = [0.1,.9]
    y0 = [0,1]
    y1 = [.25, .75]
    y2 = [.5, .5]
    y3 = [.75, .25]
    y4 = [1, 0]

    fig, ax = plt.subplots()

    plt.xticks(x, labels=['$c_{0}$', '$c_{1}$'], fontsize=fontsize_ticks)
    ax.set_ylim(-.05, 1.05)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Communities', fontsize=fontsize_label)
    ax.set_ylabel('Score', fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    alpha = 0.8

    ax.scatter(x, y0, color=colors[0], marker=markers[0], s=marker_sizes[0]**2, label='0-100')
    ax.plot(x, y0, color=colors[0], linestyle=linestyles[0], linewidth=2, alpha=alpha, label=' ')
    ax.scatter(x, y1, color=colors[1], marker=markers[1], s=marker_sizes[1]**2, label='25-75')
    ax.plot(x, y1, color=colors[1], linestyle=linestyles[1], linewidth=2, alpha=alpha, label=' ')
    ax.scatter(x, y2, color=colors[2], marker=markers[2], s=marker_sizes[2]**2, label='50-50')
    ax.plot(x, y2, color=colors[2], linestyle=linestyles[2], linewidth=2, alpha=alpha, label=' ')
    ax.scatter(x, y3, color=colors[3], marker=markers[3], s=marker_sizes[3]**2, label='75-25')
    ax.plot(x, y3, color=colors[3], linestyle=linestyles[3], linewidth=2, alpha=alpha, label=' ')
    ax.scatter(x, y4, color=colors[4], marker=markers[4], s=marker_sizes[4]**2, label='100-0')
    ax.plot(x, y4, color=colors[4], linestyle=linestyles[4], linewidth=2, alpha=alpha, label=' ')
    
    ax.legend(loc='lower left', fontsize=13, bbox_to_anchor=(0, 1.05, 1, 0.2), mode='expand', 
            ncol=5, borderaxespad=0)
    
    figure_str = f'fig/behavior_simple_com_scores.png'
    save_or_show_fig(figure_str, save)

    fig, ax = plt.subplots()

    ax.scatter(0, calc_fairness_delta_y(1), color=colors[0], marker=markers[0],s=marker_sizes[0]**2, label='0-100')
    ax.scatter(1, calc_fairness_delta_y(.5), color=colors[1], marker=markers[1],s=marker_sizes[1]**2, label='25-75')
    ax.scatter(2, calc_fairness_delta_y(0), color=colors[2], marker=markers[2],s=marker_sizes[2]**2, label='50-50')
    ax.scatter(3, calc_fairness_delta_y(-.5), color=colors[3], marker=markers[3],s=marker_sizes[3]**2, label='75-25')
    ax.scatter(4, calc_fairness_delta_y(-1), color=colors[4], marker=markers[4],s=marker_sizes[4]**2, label='100-0')

    # ax.set_xlim(0, 1)
    
    ax.set_ylabel('$\Phi$', fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    ax2 = ax.twinx()
    ax2.set_yticks([])
    
    ax.spines['bottom'].set_position('center')
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(-.5, 4.5)
    

    ax.legend(loc='lower left', fontsize=13, bbox_to_anchor=(0, 1.05, 1, 0.2), mode='expand', 
            ncol=5, borderaxespad=0)

    ax.xaxis.set_label_coords(.5,-.08)

    ax.set_xlabel('\% found for $c_{0}, c_{1}$', fontsize=fontsize_label)
    ax.set_xticks(range(5), labels=['0-100', '25-75', '50-50', '75-25', '100-0'])
    ax.tick_params(axis='x', which='major', labelsize=fontsize_ticks-2)
    
    figure_str = f'fig/behavior_simple_fm.png'
    save_or_show_fig(figure_str, save)

def fairness_figure_case():
    print('fairness case figure')
    save = True
    
    G, dict_coms = fairness_case_assignments()

    gt_coms = dict_coms['ground_truth']

    x = [100,200,300,400]

    fig, ax = plt.subplots()

    f1_fms = []
    fccn_fms = []

    for idx, key in enumerate(list(dict_coms.keys())[1:]):
        print(key)

        pred_coms = dict_coms[key]

        mF1s = []
        mFCCNs = []

        # from smallest to largest: 100-200-300-400
        for i in range(len(pred_coms)):
            gt_com = gt_coms[i]
            pred_com = pred_coms[i]

            mapped_scores = calc_mapped_scores(G, gt_com, pred_com)
            mF1s.append(mapped_scores['mF1'])
            mFCCNs.append(mapped_scores['mFCCN'])

        f1_a, f1_b = np.polyfit(x, mF1s, 1)
        f1_delta_y = f1_a*(max(x) - min(x))
        f1_fm = (2 * np.arctan(f1_delta_y)/np.pi)
        f1_fms.append(f1_fm)

        fccn_a, fccn_b = np.polyfit(x, mFCCNs, 1)
        fccn_delta_y = fccn_a*(max(x) - min(x))
        fccn_fm = (2 * np.arctan(fccn_delta_y)/np.pi)
        fccn_fms.append(fccn_fm)

        alpha = 0.6
        ax.scatter(x, mF1s, color=colors[idx], marker=markers[0], s=(marker_sizes[0]-2)**2, label='F1 '+key)
        ax.plot(x, f1_a*np.array(x)+f1_b, color=colors[idx], linestyle=linestyles[0], 
            linewidth=2, alpha=alpha, label='F1 slope')
        ax.scatter(x, mFCCNs, color=colors[idx], marker=markers[1], s=(marker_sizes[1]-2)**2, label='FCCN '+key)
        ax.plot(x, fccn_a*np.array(x)+fccn_b, color=colors[idx], linestyle=linestyles[1], 
            linewidth=2, alpha=alpha, label='FCCN slope')

    plt.xticks(x, labels=[100,200,300,400], fontsize=fontsize_ticks)
    ax.set_ylim(-.05, 1.05)
    ax.set_xlabel('Community Size', fontsize=fontsize_label)
    ax.set_ylabel('CPM Score', fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    ax.legend(loc='lower left', fontsize=12.5, bbox_to_anchor=(0, 1.05, 1, 0.2), mode='expand', 
            ncol=3, borderaxespad=0)
    
    figure_str = f'fig/behavior_fm_case_scores.png'
    save_or_show_fig(figure_str, save)

    x = [0, 50, 100]
    fig, ax = plt.subplots()

    for i in range(len(f1_fms)):
        ax.scatter(x[i], f1_fms[i], color=colors[i], marker=markers[0], s=(marker_sizes[0]-2)**2,
            label=f'F1 {x[i]} wrong')
        ax.scatter(x[i], fccn_fms[i], color=colors[i], marker=markers[1], s=(marker_sizes[1]-2)**2,
            label=f'FCCN {x[i]} wrong')

    plt.xticks(x, labels=[0, 50, 100], fontsize=fontsize_ticks)
    ax.set_ylim(-.05, 1.05)
    ax.set_xlabel('Number of nodes not correctly classified', fontsize=fontsize_label)
    ax.set_ylabel('$\Phi$', fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    ax.legend(loc='lower left', fontsize=12.5, bbox_to_anchor=(0, 1.05, 1, 0.5), mode='expand', 
            ncol=3, borderaxespad=0)

    figure_str = f'fig/behavior_fm_case_fm.png'
    save_or_show_fig(figure_str, save)


def create_method_behavior_figs():

    make_1024_network = False
    if make_1024_network:
        # generate graph required for mapped metric figure and global metric figures 
        n = 1024
        metrics_G = original_hichba(n=n, r=[1], h=0.9, p_PA=0.7, p_N=1/10, p_T=0.3)
        store_network(metrics_G, 'metric_fig_graph', 'method_behavior')

    make_100_network = False
    if make_100_network:
        n = 110
        c = 1
        while True:
            print('try', c)
            metrics_G = original_hichba(n=n, r=[0.7, 0.4], h=0.9, p_PA=0.7, p_N=1/10, p_T=0.3)

            number_c0 = len([n for n in metrics_G.nodes() if metrics_G.nodes[n]['ground_truth'] == 0])

            if number_c0 != 70:
                print(number_c0, 'nodes in c0\n')
                c += 1
            else:
                break

        store_network(metrics_G, 'method_behavior_100', 'method_behavior')


    # mapped_metric_figure() # mapped metrics
    # global_metric_figure('evenly') # global metrics split evenly
    # global_metric_figure('constant') # global metric split constant
    # fairness_figure_simple() # fairness metric simple illustration
    fairness_figure_case() # fairness metric regression line -> FM figure

    # mapped_metric_figure_min_maj()

if __name__ == '__main__':
    
    create_method_behavior_figs()