import os
import sys
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('../..')
from my_module import *

attributes = ['size', 'density', 'conductance']
acc_measure = 'nmi' # from ['nmi', 'ari', 'vi']
skip_cdm = ['CPM', 'DER', 'AGDL', 'Belief', 'Ricci', 'GA-Net']


def get_header_str(cdms, max_per_table):

    num_metrics = 9
    
    header_str = '    \\begin{adjustbox}{width=\\textheight}\n'
    header_str += '        \\begin{tabular}{|r|l|'
    header_str += 'r|'*(num_metrics*len(cdms))+'}\n'
    header_str += '        \\hline &'
    
    for cdm in cdms:
        header_str += f' & \\multicolumn{{{num_metrics}}}{{c|}}{{{cdm}}}'

    header_str += '\\\\ \\hline \n'

    metric_str = '& NMI & mFCCN & mF1 & mFCCE & mFCCE+ & AvgF1 & SWF1 & SWFCCE & SWFCCE+ '
    header_str += '        Attr & Network ' + metric_str*len(cdms)
    header_str += '\\\\ \\hline \n'
    
    return header_str

def get_split_result_str(net_name, split):
    if net_name == 'dolphins': # to-do, change order of real-world
        split_result_str = f'& {net_name.capitalize()}'
    else:
        split_result_str = f'        & {net_name.capitalize()}'
    sp = split.copy()
    
    for cdm in sp.keys():
        rounded = {}
        for key in sp[cdm]:
            if sp[cdm][key] != None:
                if sp[cdm][key] == 0:
                    rounded[key] = '0'
                else:
                    rounded[key] = '{:.2f}'.format(round(sp[cdm][key], 2))
            else:
                # no value for error methods
                rounded[key] = '-'
        
        split_result_str += f' & {rounded["nmi"]} & {rounded["fm_mFCCN"]} & {rounded["fm_mF1"]}'
        split_result_str += f' & {rounded["fm_mFCCE"]} & {rounded["fm_mFCCE+"]}'
        split_result_str += f' & {rounded["fm_AvgF1"]} & {rounded["fm_SWF1"]}'
        split_result_str += f' & {rounded["fm_SWFCCE"]} & {rounded["fm_SWFCCE+"]}'
    num_cdms = len(sp.keys())
    num = 2 + num_cdms*9 # num_metrics
    if net_name == 'yeast':
        split_result_str += f'\\\\ \\hline\n'
    else:
        split_result_str += f'\\\\ \\cline{{2-{num}}}\n'
    
    return split_result_str

def get_end_str(attr, split):
    caption = 'caption here'
    end_str = '    \\end{tabular}\n'
    end_str += '    \\end{adjustbox}\n'
    end_str += f'    \\caption{{{caption}}}\n'
    return end_str

def get_split_cdms(res_path, net_name, max_per_table):
    attr = 'size'
    # only looking at cdm names and how to split them in table sizes
    with open(f'{res_path}/res_{attr[:4]}_{net_name}.pkl', 'rb') as handle:
        result_dict = pickle.load(handle)

    res_copy = result_dict.copy()
    
    list_of_split_cdms = []

    while res_copy:
        split = []
        keys_left = list(res_copy.keys())
        keys_split = keys_left[:max_per_table]
        list_of_split_cdms.append(keys_split)

        for key in keys_split:
            del res_copy[key]

    return list_of_split_cdms
    

def get_split_results(split_cdms, attr, net_name):
    with open(f'{res_path}/res_{attr[:4]}_{net_name}.pkl', 'rb') as handle:
        result_dict = pickle.load(handle)
    # print_dict(result_dict)
    # print('\n\n')

    return {key: result_dict[key] for key in split_cdms}


def print_tables(res_path, net_name):
    max_per_table = 3

    if net_name == None:
        files = os.listdir(res_path)
        net_names = sorted(list(set([file[9:-4] for file in files 
            if file[-4:] == '.pkl'])))
    else:
        net_names = [net_name]

    # splits all results in columns for latex table
    list_cdm_splits = get_split_cdms(res_path, net_names[0], max_per_table)

    print(f'% {attr} table')

        # table_strings is a list of latex table code for each split
    table_strings = [[get_header_str(cdms, max_per_table)] for cdms in list_cdm_splits]
    for attr in attributes:
        for i in range(len(table_strings)):
            table_strings[i][-1] += f'        \\hline\\multirow{{7}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{attr.capitalize()}}}}}'
        for net_name in net_names:
            # split here is a list of cdms
            for i, split in enumerate(list_cdm_splits):
                split_results = get_split_results(split, attr, net_name)
                # add results by split in corresponding table for each network
                table_strings[i].append(get_split_result_str(net_name, split_results))

    # add end of table
    for i, split in enumerate(list_cdm_splits):
        table_strings[i].append(get_end_str(attr, split))

    for i in range(len(table_strings)):
        print(''.join(table_strings[i]))

#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------


def combine_results(file_names, res_path):
    result_dict_list = []
    for file_name in file_names:
        with open(res_path+'/'+file_name, 'rb') as handle:
            result_dict = pickle.load(handle)
        result_dict_list.append(result_dict)

    return result_dict_list

def get_avg_std_results(result_dict_list, cdms, metrics):
    avg_dict = {}
    std_dict = {}
    
    for cdm in cdms:        
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


def best_per_method(score_dict, n=5):
    df = pd.DataFrame.from_dict(score_dict, orient='index')

    metrics = score_dict[list(score_dict.keys())[0]].keys()

    best_dict = {}
    for metric in metrics:
        if metric in all_acc_measures: # highest is best
            if metric == 'vi':
                best_dict[metric] = df[metric].nsmallest(n).values
            else:    
                best_dict[metric] = df[metric].nlargest(n).values
        else:
            best_dict[metric] = df[metric].abs().nsmallest(n).values

    return best_dict

def get_header(metrics):
    string = '\\hline CDMs & '
    length = len(metrics)
    for idx in range(length):
        if 'm' == metrics[idx][0]:
            metric_str = metrics[idx][1:]
        else:
            metric_str = metrics[idx].upper()

        if idx == length - 1:
            string += metric_str + ' \\\\ \\hline \\hline \n'
        else:
            string += metric_str + ' & '
    return string

def get_table_print(dir_name, attr, file_names, mu=None):
    result_dict_list = combine_results(file_names, dir_name)

    cdms = result_dict_list[0].keys()
    cdms = [cdm for cdm in cdms if cdm not in skip_cdm]
    
    # just mapped metrics
    # metrics = [metric for metric in result_dict_list[0][list(cdms)[0]] if metric not in all_acc_measures][:4]
    
    metrics = [metric for metric in result_dict_list[0][list(cdms)[0]] if metric not in all_acc_measures]
    metrics.insert(0, acc_measure)

    avg_dict, std_dict = get_avg_std_results(result_dict_list, cdms, metrics)
    best_values_per_metric = best_per_method(avg_dict, n=5)

    total_str = ''
    total_str += get_header(metrics)
    
    for cdm in cdms:
        row_str = cdm
        for metric in metrics:
            value = avg_dict[cdm][metric]
            r_value = round(value, 3)
            
            if metric == acc_measure:
                func = 'grr'
            else:
                func = 'gr'

            if abs(value) in best_values_per_metric[metric]:
                row_str += ' & \\'+func+'{{{:.3f}}} \\textBF{{{:.3f}}}'.format(r_value, r_value)
            else:
                row_str += ' & \\'+func+'{{{:.3f}}} {:.3f}'.format(r_value, r_value)
        row_str += ' \\\\ \\hline \n' 
        total_str += row_str

    return total_str


def synth_results_table(dir_name):
    for attr in attributes:
        print('\n', attr)

        files = os.listdir(dir_name)
        file_names_total = sorted([file for file in files if attr[:4] in file])

        if separate_by_mu:
            mus = list(set([int(f[f.find('xi')+2]) for f in file_names_total]))

            for mu in mus:
                file_names = [file_name for file_name in file_names_total 
                    if 'xi'+str(mu) in file_name]

                print(f'table for mu={mu} with attribute {attr}\n')
                print(get_table_print(dir_name, attr[:4], file_names, mu), '\n\n')

        else:
            print(f'table with all mus with attribute {attr}')
            print(get_table_print(dir_name, attr[:4], file_names_total))

        # exit()


def get_perf_print(dir_name, attr, file_names, mu=None):
    result_dict_list = combine_results(file_names, dir_name)

    cdms = result_dict_list[0].keys()
    cdms = [cdm for cdm in cdms if cdm not in skip_cdm]
    
    metrics = ['nmi', 'rmi', 'vi', 'ari', 'f1', 'nf1']

    avg_dict, std_dict = get_avg_std_results(result_dict_list, cdms, metrics)
    best_values_per_metric = best_per_method(avg_dict, n=5)

    total_str = ''
    total_str += get_header(metrics)
    
    for cdm in cdms:
        row_str = cdm
        for metric in metrics:
            value = avg_dict[cdm][metric]
            r_value = round(value, 3)
            
            if metric != 'vi':
                func = 'grr'
            else:
                func = 'vi'

            if abs(value) in best_values_per_metric[metric]:
                row_str += ' & \\'+func+'{{{:.3f}}} \\textBF{{{:.3f}}}'.format(r_value, r_value)
            else:
                row_str += ' & \\'+func+'{{{:.3f}}} {:.3f}'.format(r_value, r_value)
        row_str += ' \\\\ \\hline \n' 
        total_str += row_str

    return total_str


def synth_performance_table(dir_name):
    attr = 'size' # does not matter, perf measures are same across fairness measures

    files = os.listdir(dir_name)
    file_names_total = sorted([file for file in files if attr[:4] in file])

    if separate_by_mu:
        mus = list(set([int(f[f.find('xi')+2]) for f in file_names_total]))

        for mu in mus:
            file_names = [file_name for file_name in file_names_total 
                if 'xi'+str(mu) in file_name]

            print(f'table for xi={mu} with attribute {attr}\n')
            print(get_perf_print(dir_name, attr[:4], file_names, mu), '\n\n')

    else:
        print(f'table with all mus with attribute {attr}')
        print(get_perf_print(dir_name, attr[:4], file_names_total))

    

def synth_difference_size_partition(dir_name):
    files = os.listdir(dir_name)
    file_names_total = sorted([file for file in files if 'nodes' in file])

    mus = list(set([int(f[f.find('xi')+2]) for f in file_names_total]))

    true_num = {2:[], 4:[], 6:[]}
    diff_num = {2:{}, 4:{}, 6:{}}

    for mu in mus:
        file_names = [file_name[:-10] for file_name in file_names_total 
            if 'xi'+str(mu) in file_name]

        for file_name in file_names:
            _, node_clustering_dict = get_network_communities(dir_name, file_name)
            gt_num = len(node_clustering_dict['ground_truth'].communities)
            
            true_num[mu].append(gt_num)

            cdms = list(node_clustering_dict.keys())[1:]
            cdms = [cdm for cdm in cdms if cdm not in skip_cdm]

            for cdm in cdms:
                if node_clustering_dict[cdm]:
                    cdm_num = len(node_clustering_dict[cdm].communities)
                    # print(cdm_num)
                    diff = cdm_num - gt_num
                    
                    if cdm not in diff_num[mu].keys():
                        diff_num[mu][cdm] = [diff]
                    else:
                        diff_num[mu][cdm].append(diff)

    for cdm in cdms:
        diff_2 = round(np.mean(diff_num[2][cdm]), 1)
        diff_4 = round(np.mean(diff_num[4][cdm]), 1)
        diff_6 = round(np.mean(diff_num[2][cdm]), 1)
        if diff_2 >= 0:
            diff_2 = '+'+str(diff_2)
        else:
            diff_2 = str(diff_2)

        if diff_4 >= 0:
            diff_4 = '+'+str(diff_4)
        else:
            diff_4 = str(diff_4)

        if diff_6 >= 0:
            diff_6 = '+'+str(diff_6)
        else:
            diff_6 = str(diff_6)

        print(cdm, ', '+diff_2+', '+diff_4+', '+diff_6)



def synth_difference_size_partition_HICH(dir_name):
    files = os.listdir(dir_name)
    file_names_total = sorted([file[:-10] for file in files if 'nodes' in file])

    true_num = []
    diff_num = {}

    for file_name in file_names_total:
        _, node_clustering_dict = get_network_communities(dir_name, file_name)
        gt_num = len(node_clustering_dict['ground_truth'].communities)
        
        true_num.append(gt_num)

        cdms = list(node_clustering_dict.keys())[1:]
        cdms = [cdm for cdm in cdms if cdm not in skip_cdm]

        for cdm in cdms:
            if node_clustering_dict[cdm]:
                cdm_num = len(node_clustering_dict[cdm].communities)
                
                diff = cdm_num - gt_num
                
                if cdm not in diff_num.keys():
                    diff_num[cdm] = [diff]
                else:
                    diff_num[cdm].append(diff)

    for cdm in cdms:
        diff = round(np.mean(diff_num[cdm]), 1)
        if diff >= 0:
            diff = '+'+str(diff)
        else:
            diff = str(diff)

        print(cdm, ', '+diff)


def real_results_table(dir_name, net_name):
    print('to do')
    exit()


if __name__ == '__main__':
    
    p = Parser()
    dir_name = '../../'+p.directory
    net_name = p.network

    # settings on what to show/save
    separate_by_mu = True

    # print('start tables.py')

    if 'synthetic' in dir_name:
        synth_results_table(dir_name)
        # synth_performance_table(dir_name)

        exit()
        if 'HICH' in dir_name:
            synth_difference_size_partition_HICH(dir_name)
        else:
            synth_difference_size_partition(dir_name)
        exit()
    
    if net_name == None:
        # get all network names in res_path        
        files = os.listdir(res_path)
        
        net_names = sorted(list(set([file[9:-4] for file in files 
            if file[-4:] == '.pkl'])))

        print('Networks run:', net_names)
        for net_name in net_names:
            real_results_table(dir_name, net_name)
    else:
        # use only the provided net_name (using -n ...)
        real_results_table(dir_name, net_name)
