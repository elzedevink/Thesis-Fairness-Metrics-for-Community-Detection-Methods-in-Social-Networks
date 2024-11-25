import os
import pickle
import numpy as np
import seaborn as sns
from my_module import *
import matplotlib.pyplot as plt
import matplotlib as mpl


attributes = ['size', 'density', 'conductance']


# the method strings have to be the same as the cdm name found in result_dict
optimization_cdms = {
    'name': 'optimization',
    'methods': ['CNM', 'Combo', 'CPM', 'Leiden', 'Louvain', 'Paris', 'RB-C', 
        'RB-ER', 'Significance'],
    'marker_idx': 0
}
spectral_cdms = {
    'name': 'spectral',
    'methods': ['Eigenvector', 'RSC-K', 'RSC-SSE', 'RSC-V', 'Spectral'],
    'marker_idx': 1
}
representational_cdms = {
    'name': 'representational',
    'methods': ['Deepwalk', 'DER', 'Fairwalk', 'Node2Vec'],
    'marker_idx': 2
}
dynamics_cdms = {
    'name': 'dynamics',
    'methods': ['Infomap', 'Spinglass', 'Walktrap'],
    'marker_idx': 3
}
propagation_cdms = {
    'name': 'propagation',
    'methods': ['Fluid', 'Label Propagation'],
    'marker_idx': 4
}
miscellaneous_cdms = {
    'name': 'miscellaneous',
    'methods': ['EM','SBM', 'SBM - Nested'],
    'marker_idx': 5
}

cdm_groups = [optimization_cdms, spectral_cdms, representational_cdms, 
    dynamics_cdms, propagation_cdms, miscellaneous_cdms]

# no longer used in analysis
removed_cdms = ['CPM', 'DER', 'AGDL', 'Belief', 'Ricci', 'GA-Net']

# return CDM group name
def get_cdm_group(cdm_name):
    for group in cdm_groups:
        if cdm_name in group['methods']:
            return group
    print('no group found')
    return None

"""
produces the legend figure, set the number of columns of the CDMs with ncol
called from within fm_accuracy_figure()
"""
def create_legend_fig(ax):
    for ncol in [2,3,6,8]:
        fig_legend, ax_legend = plt.subplots(figsize=(3,0))
        legend = ax_legend.legend(*ax.get_legend_handles_labels(), loc='center', ncol=ncol, 
            borderaxespad=0)
        ax_legend.axis('off')

        save_or_show_fig(f'figures/thesis_text/fig/legend_ncol{ncol}', True)


def fm_accuracy_figure(results, metric, attr, acc_measure, error_bar=None, mp=None):
    """
    Creates fairness/performance figures
    :param results: dict containing the fairness/performance results
    :param metric: fairness metric Phi variant
    :param attr: community property bias is measured against
    :param acc_measure: performance/accuracy measure of entire prediction partition
    :param error_bar: if given, create error bars using std deviations given in dict 
        error_bar
    :param mp: mixing parameter value for certain synthetic networks
    """

    # regular marker sizes
    marker_sizes = [13, 13, 13, 15, 13, 13]

    # # change marker sizes for separate legend figure
    if create_separate_legend:
        marker_sizes = [12, 12, 12, 15, 9, 12]

    # attribute is full attribute string
    # e.g. attr = 'cond', attribute = 'conductance'
    for attribute in attributes:
        if attr in attribute:
            break
    # remove prefix 'm' if present
    if metric[0] == 'm':
        metric_str = metric[1:]
    else:
        metric_str = metric

    cdms = list(results.keys())
    x = [results[cdm][metric] for cdm in cdms]
    y = [results[cdm][acc_measure] for cdm in cdms]

    fig, ax = plt.subplots()

    # change y-axis, vi is not 0-1
    if acc_measure == 'vi':
        ax.set_ylim(0, 3)
        ax.set_yticks([0.6, 1.2, 1.8, 2.4, 3])
    else:
        ax.set_ylim(0, 1.05)    
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

    # set y-label
    if acc_measure == 'f1':
        ax.set_ylabel('PF1', fontsize=fontsize_label)
    else:
        ax.set_ylabel(acc_measure.upper(), fontsize=fontsize_label)
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1) # vertical line
    ax.set_xlabel(f'$\\Phi^{{{metric_str}}}_{{{attribute}}}$', fontsize=fontsize_label+4)
    ax.set_xlim(-.7, .7)
    ax.set_xticks([-0.6, -0.3, 0, 0.3, 0.6])    
    ax.tick_params(axis='both', labelsize=fontsize_ticks)
    
    # plot values
    for i in range(len(x)):
        color_idx = i%10
        group = get_cdm_group(list(cdms)[i]) # get CDM category
        marker = markers[group['marker_idx']] # set marker by group
        markersize = marker_sizes[group['marker_idx']] # set markersize by group
        
        # plot values with error bars
        if error_bar:
            x_err = [error_bar[cdm][metric] for cdm in cdms]
            y_err = [error_bar[cdm][acc_measure] for cdm in cdms]
            marker, _, bars = ax.errorbar(x[i], y[i], xerr=x_err[i], yerr=y_err[i], lw=2, 
                marker=marker, markersize=markersize, color=colors[color_idx], 
                alpha=1, label=list(cdms)[i])
                        
            [bar.set_alpha(0.5) for bar in bars]

        # plot values without error bars
        else: 
            ax.scatter(x[i], y[i], marker=marker, s=markersize**2,
                color=colors[color_idx], label=list(cdms)[i])

    if individual_legend:
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.05, 1, 0.2), mode='expand', 
            ncol=4, borderaxespad=0)

    # creates legend figure with different number of columns
    # only needs to be created once
    if create_separate_legend:
        create_legend_fig(ax)
        print('legend figures created - set create_separate_legend to False')
        exit()
    
    if error_bar:
        fig_pre = f'{fig_path}/{acc_measure}/'
    else:
        fig_pre = f'{fig_path}/{acc_measure}/{net_name}_'

    if mp:
        fig_str = f'{fig_pre}{attr[:4]}_{mixing_param}{mp}_{metric}.png'
    else:
        fig_str = f'{fig_pre}{attr[:4]}_{metric}.png'

    save_or_show_fig(fig_str, save_figure)


# check if methods with results are included above
def check_imported_cdms(cdms):
    imported_cdms = []
    not_imported_cdms = []
    for cdm_group in cdm_groups:
        imported_cdms.extend(cdm_group['methods'])
    for cdm in cdms:
        if cdm not in imported_cdms:
            not_imported_cdms.append(cdm)
    if not_imported_cdms != []:
        print('imported cdms:\t\t', imported_cdms)
        print('not imported cdms:\t', not_imported_cdms)


# filters out CDMs that have encountered an error when assigning communities
def error_method_filter(result_dict, attr):
    cdms = result_dict.keys()

    to_delete = []
    for cdm in cdms:
        fm_mF1 = result_dict[cdm]['mF1']
        if fm_mF1 == None:
            to_delete.append(cdm)

    if to_delete:
        print(f'Deleted {len(to_delete)} from {attr} results')
        print(to_delete)

    for cdm in to_delete:
        del result_dict[cdm]

    return result_dict

# create figures for a single network
def use_results_per_network(res_path, net_name, fig_path):
    print(f'\nNetwork: {net_name}')
    
    for attr in attributes:
        with open(f'{res_path}/res_{attr[:4]}_{net_name}.pkl', 'rb') as handle:
            result_dict = pickle.load(handle)

        result_dict = error_method_filter(result_dict, attr)

        if result_dict == {}:
            print(f'No results for {net_name}, regarding {attr}')
            continue

        cdms = result_dict.keys()
        metrics = [metric for metric in result_dict[list(cdms)[0]] 
            if metric not in all_acc_measures]
        
        # check if I need to add methods above and whether it is correctly added
        check_imported_cdms(cdms)

        for metric in metrics:
            if acc_measure == None:
                for acc_metric in all_acc_measures:
                    fm_accuracy_figure(result_dict, metric, attr, acc_metric)
            else:
                fm_accuracy_figure(result_dict, metric, attr, acc_measure)


def combine_results(file_names, res_path):
    result_dict_list = []
    for file_name in file_names:
        with open(res_path+file_name, 'rb') as handle:
            result_dict = pickle.load(handle)
        result_dict_list.append(result_dict)

    return result_dict_list


# get average and std values from results on multiple networks
def get_avg_std_results(result_dict_list, cdms, metrics):
    avg_dict = {}
    std_dict = {}
    for cdm in cdms:
        if cdm in removed_cdms:
            continue
        
        cdm_metric_score = {metric:[] for metric in metrics}

        for i in range(len(result_dict_list)):
            if not result_dict_list[i][cdm][metrics[1]]:
                continue
            for metric in metrics:
                cdm_metric_score[metric].append(result_dict_list[i][cdm][metric])

        cdm_metric_avg = {metric: np.mean(cdm_metric_score[metric]) for metric in metrics}
        cdm_metric_std = {metric: np.std(cdm_metric_score[metric]) for metric in metrics}

        avg_dict[cdm] = cdm_metric_avg
        std_dict[cdm] = cdm_metric_std

    return avg_dict, std_dict


# call figures for multiple network results
def create_combined_fig(attr, file_names, res_path, acc_metric, mp=None):
    result_dict_list = combine_results(file_names, res_path)

    cdms = result_dict_list[0].keys()

    metrics = [metric for metric in result_dict_list[0][list(cdms)[0]] 
            if metric not in all_acc_measures]

    metrics.append(acc_metric)

    # check if methods need to be added above and whether it is correctly added
    check_imported_cdms(cdms)
            
    avg_dict, std_dict = get_avg_std_results(result_dict_list, cdms, metrics)

    for fm_metric in metrics[:-1]: # don't include acc_metric
        fm_accuracy_figure(avg_dict, fm_metric, attr, acc_metric, error_bar=std_dict,
            mp=mp)


# create figures for results on multiple networks
def use_results_combined_networks(res_path, fig_path, acc_measure):

    for attr in attributes:
        print(attr)

        files = os.listdir(res_path)
        file_names_total = sorted([file for file in files if attr[:4] in file])

        # separate by mixing parameter
        if separate_by_mp:
            mps = list(set([int(f[f.find(mixing_param)+2]) for f in file_names_total]))

            for mp in mps:
                file_names = [file_name for file_name in file_names_total 
                    if mixing_param+str(mp) in file_name]

                if acc_measure == None:
                    for acc_metric in all_acc_measures:
                        create_combined_fig(attr[:4], file_names, res_path, 
                            acc_metric=acc_metric, mp=mp)
                else:
                    create_combined_fig(attr[:4], file_names, res_path, 
                        acc_metric=acc_measure, mp=mp)
        else:
            if acc_measure == None:
                for acc_metric in all_acc_measures:
                    create_combined_fig(attr[:4], file_names_total, res_path, 
                        acc_metric=acc_metric)
            else:
                create_combined_fig(attr[:4], file_names_total, res_path, 
                    acc_metric=acc_measure)
    return



if __name__ == '__main__':
    # settings
    create_separate_legend = True

    # include individual legend
    individual_legend = False 
    
    # True: save figures, False: show figures
    save_figure = False
    
    # create separate figures for different mu or combine results
    separate_by_mp = True 

    # set acc_measure or set as None to use all acc_measures from 
    #   ['nmi', 'ari', 'vi', 'f1', 'nf1', 'rmi']
    acc_measure = None 

    p = Parser()
    dir_name = p.directory 
    res_path = path_prefix(dir_name, 'results')
    fig_path = path_prefix(remove_prefix(res_path, "results/"), 'figures', create=True)
    net_name = p.network

    print('start create_figures.py')
    print(f'res_path = {res_path}, fig_path = {fig_path}')

    # synthetic data: combine results across multiple networks
    if 'synthetic' in res_path:
        mixing_param = ''
        if 'LFR' in res_path:
            mixing_param = 'mu'
        elif 'ABCD' in res_path:
            mixing_param = 'xi'

        use_results_combined_networks(res_path, fig_path, acc_measure)

    # run through all real-world networks
    elif net_name == None:
        # get all network names in res_path        
        files = os.listdir(res_path)
        
        net_names = sorted(list(set([file[9:-4] for file in files 
            if file[-4:] == '.pkl'])))

        print('Networks run:', net_names)
        for net_name in net_names:
            use_results_per_network(res_path, net_name, fig_path)

    # run through specific real-world network
    else:
        use_results_per_network(res_path, net_name, fig_path)
