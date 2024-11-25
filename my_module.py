import os
import csv
import sys
import json
import random
import pathlib
import warnings
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from cdlib import NodeClustering

import gensim
from sklearn.cluster import KMeans
from fairwalk import FairWalk
from node2vec import Node2Vec
    
plt.rcParams.update({
    "text.usetex": True,
})

warnings.filterwarnings('error', category=np.RankWarning)

colors = sns.color_palette('bright')
markers = ['o', 's', '^', '*', 'D', '>']
marker_sizes = [13, 13, 13, 15, 13, 13]

linestyles = ['--', ':', '-.', (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1))]

fontsize_ticks = 24
fontsize_label = 26

# acc_measure = 'nmi'
all_acc_measures = ['nmi', 'ari', 'vi', 'f1', 'nf1', 'rmi']

def requires_directory():
    python_file = sys.argv[0]
    if python_file == 'auto_generate_networks.py':
        return False
    return True

# for parsing directory and seed information when calling this code
class Parser:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Description for my parser')
        parser.add_argument('-d', '--directory', help='Give directory name', 
            required=requires_directory(), type=str)
        parser.add_argument('-n', '--network', help='Give network name', required=False, 
            default=None)
        
        p = parser.parse_args()
        self.directory = p.directory
        self.network = p.network

# returns for a given graph and algorithm the community assignments
# returns the communities as a list of lists of nodes
def get_assignment(df, alg):
    df = df[alg]

    if df.eq(-1).all():
        # invalid method
        return None

    assignment = []
    for com in range(df.max()+1):
        ids = df.index[df == com]
        assignment.append(list(ids))

    return assignment

def get_node_clustering_dict(G, df):

    algs = df.columns

    node_clustering_dict = {}
    for alg in algs:
        assignment = get_assignment(df, alg)
        if not assignment:
            node_clustering_dict[alg] = None
            continue

        if alg == 'ground_truth':
            node_clustering_dict[alg] = NodeClustering(assignment, graph=G, method_name=alg)
        elif '{' in alg:
            i = alg.index('{')

            # method_name = alg[:i-1] # removes parameter from method_name
            method_name = alg
            param = eval(alg[i:])

            # node_clustering_dict keys are method_name (no parameter info)
            node_clustering_dict[method_name] = NodeClustering(assignment, graph=G, 
                method_name=method_name, method_parameters=param)
        else:
            # print('Warning: weird community prediction.')
            # print(alg)
            node_clustering_dict[alg] = NodeClustering(assignment, graph=G, method_name=alg)
    
    return node_clustering_dict


# returns networkx network from given directory, network name
def get_network_communities(dir_path, net_name, request_node_clustering=True):
    graph_path = dir_path+'/'+net_name
    try:
        H = nx.read_adjlist(graph_path+'_edges.csv', nodetype=int, delimiter=',')
        G = nx.Graph()
        G.add_nodes_from(sorted(H.nodes(data=True)))
        G.add_edges_from(H.edges(data=True))
    except FileNotFoundError:
        print(f'Network "{net_name}" does not exist.')
        exit()

    df = pd.read_csv(graph_path+'_nodes.csv', index_col=0)

    if request_node_clustering:
        return G, get_node_clustering_dict(G, df)
    else:
        comm_assignment_dict = df.to_dict('index')

    return G, comm_assignment_dict

def to_NodeClustering(G, comm_assignment_dict):
    com_dict = {}

    for n in comm_assignment_dict.keys():
        com = comm_assignment_dict[n]['ground_truth']
        if com not in com_dict.keys():
            com_dict[com] = [n]
        else:
            com_dict[com].append(n)
    
    communities = [com_dict[i] for i in range(len(com_dict.keys()))]
    
    return NodeClustering(communities, graph=G, method_name='ground_truth')

def save_or_show_fig(figure_str, save_fig):
    if save_fig:
        print(f'save fig: {figure_str}')
        plt.savefig(figure_str, bbox_inches='tight', dpi=300)
    else:
        print(f'show fig: {figure_str}')
        plt.show()
    plt.close()


def print_dict(some_dict):
    print(json.dumps(some_dict, indent=4))
    

def jaccard_sim(a, b):
    a = set(a)
    b = set(b)

    return len(a.intersection(b)) / len(a.union(b))

def greedy_sim_mapping(source, target):
    mapping = {}
    source_com = source.communities
    target_com = target.communities

    used_targets = []

    for com_n in range(len(source_com)):
        community = source_com[com_n]

        best = -1
        best_score = 0
        for t in range(len(target_com)):
            target = target_com[t]
            
            JS = jaccard_sim(community, target)
            if JS > best_score:
                best_score = round(JS, 2)
                best = t
                num_correct = len(set(community).intersection(set(target)))

        if best not in used_targets:
            used_targets.append(best)

            mapping[com_n] = {'target': best, 'JS': best_score, 
                'source_size': len(community), 'target_size': len(target_com[best]), 
                'correct': num_correct}
        
        # multiple communities choose same target
        # set worse scoring community as missclassified 
        # missclassified: ('target': -1, 'JS': 0, 'correct':0)
        else:
            other_key = [key for key, value in mapping.items() if value['target'] == best][0]
            other_score = mapping[other_key]['JS']

            if best_score > other_score:
                mapping[com_n] = {'target': best, 'JS': best_score, 
                    'source_size': len(community), 'target_size': len(target_com[best]), 
                    'correct': num_correct}
                
                mapping[other_key]['target'] = -1
                mapping[other_key]['JS'] = 0
                mapping[other_key]['correct'] = 0
            else:
                mapping[com_n] = {'target': -1, 'JS': 0, 
                    'source_size': len(community), 'target_size': 0, 'correct': 0}

    return mapping


def find_highest_set_zero(matrix):
    max_val = 0
    max_coord_options = []

    # Find the highest value in the matrix. If tied, add to options
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] > max_val:
                max_val = matrix[i][j]
                max_coord_options = [(i, j)]
            elif matrix[i][j] == max_val:
                max_coord_options.append((i, j))

    if max_coord_options == []:
        return -1, None

    # random choice if tie
    max_coord = random.choice(max_coord_options)

    # Set the corresponding row and column to -1
    row, col = max_coord
    for i in range(len(matrix)):
        matrix[i][col] = -1
    for j in range(len(matrix[row])):
        matrix[row][j] = -1
    
    return max_val, max_coord


def iterative_mapping(source, target):
    mapping = []

    source_coms = source.communities
    target_coms = target.communities

    sim_matrix = []

    for source_com in source_coms:
        JS = [jaccard_sim(source_com, target_com) for target_com in target_coms]
        sim_matrix.append(JS)

    for _ in range(len(source_coms)):
        max_value, max_coord = find_highest_set_zero(sim_matrix)

        # more coms in source than in target, map to None, invalid
        if max_coord == None:
            to_map = set(range(len(source_coms)))
            already_mapped = set([source for source, target in mapping])
            still_to_map = list(to_map.difference(already_mapped))

            for s in still_to_map:
                mapping.append((s,None))
            break

        mapping.append(max_coord)

    # sorts mapping by source
    return sorted(mapping)


def calc_mapped_scores(G, gt_com, pred_com):
    """
        Given the ground-truth community (list) and predicted community (list),
        return the mapped scores in a dictionary
    """
    if pred_com == None:
        mapping_scores = {
            'mF1': 0,
            'mFCCN': 0,
            'mFCCE': 0,
            'mFCCE+': 0
        }
        return mapping_scores

    set_gt_com = set(gt_com)
    set_pred_com = set(pred_com)

    TP = len(set_gt_com & set_pred_com)
    FN = len(set_gt_com) - TP
    FP = len(set_pred_com) - TP

    FCCN = TP / (TP + FN)
    F1 = (2 * TP) / (2 * TP + FP + FN)

    # calculate FCCE: Fraction of Correctly Classified Edges
    gt_SG = G.subgraph(gt_com)
    gt_edges = gt_SG.edges()

    edge_overlap = 0
    half_overlap = 0
    for edge_source, edge_target in gt_edges:
        if edge_source in pred_com and edge_target in pred_com:
            edge_overlap += 1
        elif edge_source in pred_com or edge_target in pred_com:
            half_overlap += 1

    if len(gt_edges) == 0:
        FCCE = 1
        FCCEP = 1
    else:
        FCCE = edge_overlap/len(gt_edges)
        FCCEP = (half_overlap + 2*edge_overlap)/(2*len(gt_edges))

    mapping_scores = {
        'mFCCN': FCCN,
        'mF1': F1,
        'mFCCE': FCCE,
        'mFCCE+': FCCEP
    }
    return mapping_scores


def calc_fairness_delta_y(delta_y):    
    return (2 * np.arctan(delta_y))/np.pi

def calc_fairness_metric(x, y):
    """
    Calculate fairness metric for given attribute values and metric scores
    :param x: attribute values
    :param y: metric scores
    :return fm: fairness metric
    """

    try:
        if np.max(x) == np.min(x):
            return None
            
        x_norm = (x-np.min(x)) / (np.max(x) - np.min(x))
        a, b = np.polyfit(x_norm, y, 1)

        fm = (2 * np.arctan(a)/np.pi)

    except:
        fm = None

    return fm


# Sum of weighted precision, three versions, see notes
def calc_SWP(overlaps, gt_size, pred_lens, version):
    
    WPs = [
        ((overlaps[i]/gt_size) * (overlaps[i]/pred_lens[i]))
        for i in range(len(pred_lens))
    ]

    WPs = [i for i in WPs if i != 0]

    if version == 'v0':
        return sum(WPs)
    elif version == 'v1':
        return sum(WPs)/len(WPs)
    elif version == 'v2':
        return sum(WPs)/(np.log2(len(WPs)+1))
    print('Give Sum of Weighted Overlap version')
    return -1

# Sum of weighted f1
def calc_SWF1(overlaps, gt_size, pred_lens, version='v0'):

    TPs = overlaps
    FPs = [pred_lens[i] - TPs[i] for i in range(len(TPs))]
    FNs = [gt_size - TPs[i] for i in range(len(TPs))]
    
    WF1s = [
        ((TPs[i]/gt_size) * (2 * TPs[i]) / (2 * TPs[i] + FPs[i] + FNs[i]))
        for i in range(len(pred_lens))
    ]

    WF1s = [i for i in WF1s if i != 0]

    if version == 'v0':
        return sum(WF1s)
    elif version == 'v1':
        return sum(WF1s)/len(WF1s)
    elif version == 'v2':
        return sum(WF1s)/(np.log2(len(WF1s)+1))
    print('Give Sum of Weighted F1 version')
    return -1

# Average of F1
def calc_AF1(overlaps, gt_size, pred_lens):
    TPs = overlaps
    FPs = [pred_lens[i] - TPs[i] for i in range(len(TPs))]
    FNs = [gt_size - TPs[i] for i in range(len(TPs))]
    
    F1s = [
        (2 * TPs[i]) / (2 * TPs[i] + FPs[i] + FNs[i])
        for i in range(len(pred_lens))
    ]

    # F1s where overlaps (TP) is not 0
    F1s = [i for i in F1s if i != 0]

    return sum(F1s)/len(F1s)

def calc_SWFCCE(G, gt_com, pred_coms, overlaps, gt_size, version):
    gt_SG = G.subgraph(gt_com)
    gt_edges = gt_SG.edges()
    if len(gt_edges) == 0:
        return 1

    # print(len(gt_edges))
    full_overlaps = [0]*len(pred_coms)
    half_overlaps = [0]*len(pred_coms)
    for pred_idx in range(len(pred_coms)):
        pred_nodes = pred_coms[pred_idx]

        for edge_source, edge_target in gt_edges:
            if edge_source in pred_nodes and edge_target in pred_nodes:
                full_overlaps[pred_idx] += 1
            elif edge_source in pred_nodes or edge_target in pred_nodes:
                half_overlaps[pred_idx] += 1

    if version == 'FCCE':
        WFCCEs = [0]*len(pred_coms)
        for pred_idx in range(len(pred_coms)):
            # weighted fcce
            weight = overlaps[pred_idx]/gt_size
            WFCCEs[pred_idx] = weight * (full_overlaps[pred_idx]/len(gt_edges))
        return sum(WFCCEs)

    elif version == 'FCCE+':
        # weighted FCCE+ aka FCCEP
        WFCCEPs = [0]*len(pred_coms)
        for pred_idx in range(len(pred_coms)):
            # weighted fcce
            weight = overlaps[pred_idx]/gt_size
            WFCCEPs[pred_idx] = weight * \
                ((half_overlaps[pred_idx]+2*full_overlaps[pred_idx])/(2*len(gt_edges)))
        return sum(WFCCEPs)
    else:
        print('give SWFCCE version')

def calc_global_scores(G, gt_com, pred_coms):
    """
    Calculates global scores

    :param G: NetworkX graph
    :param gt_com: ground-truth communities
    :param pred_coms: list of predicted communities
    :return global_scores: dictionary with SFW1 and SWFCCE
    """
    overlaps = [len(set(gt_com) & set(pred_com)) for pred_com in pred_coms]
    pred_lens = [len(pred_com) for pred_com in pred_coms]
    gt_size = len(gt_com)

    AvgF1 = calc_AF1(overlaps, gt_size, pred_lens)
    SWF1 = calc_SWF1(overlaps, gt_size, pred_lens, 'v0')
    SWFCCE = calc_SWFCCE(G, gt_com, pred_coms, overlaps, gt_size, 'FCCE')
    SWFCCEp = calc_SWFCCE(G, gt_com, pred_coms, overlaps, gt_size, 'FCCE+')
    
    global_scores = {
        'AvgF1': AvgF1,
        'SWF1': SWF1,
        'SWFCCE': SWFCCE,
        'SWFCCE+': SWFCCEp
    }
    return global_scores

def print_network_info(G, net_name, gt_coms=None):
    # print network data
    print(net_name)
    print('num nodes', len(G.nodes))
    print('num edges', len(G.edges))
    print('avg degree', np.average([G.degree[n] for n in G.nodes]))
    print('highest degree', max([G.degree[n] for n in G.nodes]))
    print('lowest degree', min([G.degree[n] for n in G.nodes]))
    if gt_coms:
        print('num communities', len(gt_coms))
        print('largest community', max([len(c) for c in gt_coms]))
        print('smallest community', min([len(c) for c in gt_coms]))
    return


# The graph is stored in two csv files:
# - 'name'_edges.csv: adjlist of nodes
# - 'name'_nodes.csv: list of nodes and their communities
def store_network(G, name, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    file_str = path + '/' + name

    nx.write_adjlist(G, file_str+'_edges.csv', delimiter=',')
    community = nx.get_node_attributes(G, 'ground_truth')
    
    n = len(G.nodes)
    num_com = max(list(community.values()))+1

    com_csv = [['id', 'ground_truth']]
    for i in range(n):
        com_csv.append([i, community[i]])
    
    with open(file_str + '_nodes.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(com_csv)

    print(f'Stored {file_str} with {n} nodes and {num_com} communities')


def show_degree_dist(G):

    degrees = sorted([G.degree[n] for n in G.nodes()])
        
    deg_dict = {}
    for deg in degrees:
        if deg in deg_dict.keys():
            deg_dict[deg] += 1
        else:
            deg_dict[deg] = 1
    
    x = list(deg_dict.keys())
    y = list(deg_dict.values())

    fig, ax = plt.subplots()
    plt.ylabel('Occurences')
    plt.xlabel('Degree')
    ax.scatter(x, y)
    plt.xscale('log')
    plt.show()

def show_community_dist(G):
    
    com_lens = {}

    for n in G.nodes():
        if G.nodes[n]['ground_truth'] in com_lens.keys():
            com_lens[G.nodes[n]['ground_truth']] += 1
        else:
            com_lens[G.nodes[n]['ground_truth']] = 1

    x = list(com_lens.keys())
    y = list(com_lens.values())
    
    fig, ax = plt.subplots()
    plt.ylabel('Community size')
    plt.xlabel('Community')
    ax.scatter(x, y)
    plt.yscale('log')
    plt.show()


def show_community_conductance(G):
    com_dict = {}

    for n in G.nodes():
        com = G.nodes[n]['ground_truth']
        if com not in com_dict.keys():
            com_dict[com] = [n]
        else:
            com_dict[com].append(n)
    
    communities = [com_dict[i] for i in range(len(com_dict.keys()))]

    ground_truth = NodeClustering(communities, graph=G, method_name='ground_truth')
    conductances = evaluation.conductance(G, ground_truth, summary=False)

    x = [len(c) for c in communities]
    y = conductances
    
    fig, ax = plt.subplots()
    plt.ylabel('Conductance')
    plt.xlabel('Community size')
    ax.scatter(x, y)
    plt.show()

"""
    The HICH-BA model uses the following parameters: 
    (i) n, i.e., the desired number of nodes,
    (ii) p_N , i.e., the probability of adding a node to the graph, with 
        probability 1 − pN an edge is added,
    (iii) r, list where each entry ri corresponds to the probability of a new node 
        belonging to community i,
    (iv) h, i.e., the homophily factor and represents the probability of a node 
        establishing an intra-community connection,
    (v) p_T, the probability to form a close triad connection,
    (vi) p_PA, the probability with which a new edge will be established using the 
        preferential attachment (PA)
"""
def hichba(n,r,h,p_PA,p_N,p_T):
    
    num_com = len(r)
    G = nx.Graph()
    nx.set_node_attributes(G, [], 'ground_truth')
    G.add_nodes_from(range(num_com))
    nodes = len(G.nodes())
    
    choices_c = {c:[] for c in range(num_com)}
    choices_weights_c = {c:{} for c in range(num_com)}
    
    c = 0
    for v in G.nodes():
        G.nodes[v]['ground_truth'] = c
        choices_c[c].append(v)
        choices_weights_c[c][v] = 1
        c += 1
    
    L_values, x_val = [], []
    pbar = tqdm(total=n, position=0, leave=True)
    pbar.update(len(G.nodes()))
    h_orig = h
    while nodes < n:
        # new node created
        if random.uniform(0,1) <= p_N:
            G.add_node(nodes)
            source = nodes
            nodes += 1
            c = random.choices(range(num_com), weights=r, k=1)[0]
            G.nodes[source]['ground_truth'] = c

            choices_c[c].append(source)
            choices_weights_c[c][source] = 1 

            # choices of nodes in same com
            choices = [x for x in choices_c[c] if x != source]
            
            # PA or not
            if random.uniform(0, 1) <= (1-p_PA): weights = [1 for v in choices]
            else: weights = [choices_weights_c[G.nodes[v]['ground_truth']][v] for v in choices]
            
            target = random.choices(choices, weights=weights, k=1)[0]

            G.add_edge(source, target)

            choices_weights_c[c][source] += 1
            choices_weights_c[c][target] += 1
            pbar.update(1)

        # new connection
        else:
            # triad connection
            if random.uniform(0,1) <= p_T:
                possible_v = [x for x in G.nodes() if G.degree(x) >= 2]
                if random.uniform(0,1) <= (1-p_PA):
                    if len(possible_v) == 0: continue
                    v = random.choice(possible_v)
                else:
                    if len(possible_v) == 0: continue
                    v = random.choices(possible_v, weights=[G.degree(x)+1 for x in G.nodes() if G.degree(x) >= 2],k=1)[0]
                    # v = random.choices(possible_v, weights=[G.degree(x) for x in G.nodes() if G.degree(x) >= 2],k=1)[0]
                
                target1 = random.choice(list(G.neighbors(v)))
                options = [y for y in G.neighbors(v) if not G.has_edge(target1, y)]
                if len(options) == 0: continue

                # homophily
                intra_inter = random.uniform(0, 1)
                if intra_inter <= h: choices = [x for x in options if G.nodes[v]['ground_truth'] == G.nodes[x]['ground_truth']]
                else: choices = [x for x in options if G.nodes[v]['ground_truth'] != G.nodes[x]['ground_truth']]
                
                if random.uniform(0, 1) <= (1-p_PA): weights = [1 for w in options]
                else: weights=[choices_weights_c[G.nodes[w]['ground_truth']][w] for w in options] 
                
                if len(options) == 0: print('no', intra_inter); continue
                target2 = random.choices(options, weights=weights, k=1)[0]
                
                G.add_edge(target1, target2)
                choices_weights_c[G.nodes[target1]['ground_truth']][target1] += 1
                choices_weights_c[G.nodes[target2]['ground_truth']][target2] += 1
                
            # random connection
            else:
                if random.uniform(0,1) <= (1-p_PA):
                    v = random.choice([x for x in G.nodes()])
                else:
                    v=random.choices([x for x in G.nodes()], weights=[G.degree(x)+1 for x in G.nodes()],k=1)[0]
                    # v = random.choices([x for x in G.nodes()], weights=[G.degree(x) for x in G.nodes()],k=1)[0]
                    
                neigh = list(G.neighbors(v))
                options = [x for x in G.nodes() if x not in neigh]
                intra_inter = random.uniform(0, 1)
                if intra_inter <= h: choices = [x for x in options if G.nodes[v]['ground_truth'] == G.nodes[x]['ground_truth']]
                else: choices = [x for x in options if G.nodes[v]['ground_truth'] != G.nodes[x]['ground_truth']]

                if random.uniform(0, 1) <= (1-p_PA): weights = [1 for v in choices]
                else: weights = [choices_weights_c[G.nodes[v]['ground_truth']][v] for v in choices] 

                if len(choices) == 0: continue
                target = random.choices(choices, weights=weights, k=1)[0]
                # if (intra_inter > h and random.uniform(0,1) <= r[G.nodes[target]['ground_truth']]/r[G.nodes[v]['ground_truth']]) or intra_inter<h:
                G.add_edge(v, target)

                choices_weights_c[G.nodes[v]['ground_truth']][v] += 1
                choices_weights_c[G.nodes[target]['ground_truth']][target] += 1
        
    return G


def original_hichba(n,r,h,p_PA,p_N,p_T):
    
    num_com=len(r)
    G= nx.Graph()
    nx.set_node_attributes(G, [], "ground_truth")
    G.add_nodes_from(range(num_com))
    nodes=len(G.nodes())
    
    choices_c={c:[] for c in range(num_com)}
    choices_weights_c={c:{} for c in range(num_com)}
    
    c=0
    for v in G.nodes():
        G.nodes[v]['ground_truth']= c
        choices_c[c].append(v)
        choices_weights_c[c][v]=1
        c+=1
        
    L_values,x_val=[],[]
    pbar = tqdm(total=n, position=0, leave=True)
    pbar.update(len(G.nodes()))
    h_orig=h
    while nodes<=n:
        if random.uniform(0,1)<=p_N:
            G.add_node(nodes-1)
            source=nodes-1
            nodes+=1
            c=random.choices(range(num_com), weights=r, k=1)[0]
            G.nodes[source]['ground_truth']= c

            choices_c[c].append(source)
            choices_weights_c[c][source]=1 

            choices=[x for x in choices_c[c] if x!=source]

            if random.uniform(0, 1)<=(1-p_PA):weights=[1 for v in choices]
            else:weights=[choices_weights_c[G.nodes[v]['ground_truth']][v] for v in choices]
            
            
            if len(choices)==0:continue
            target=random.choices(choices, weights=weights, k=1)[0]

            G.add_edge(source, target)

            choices_weights_c[c][source]+=1
            choices_weights_c[G.nodes[target]['ground_truth']][target]+=1
            pbar.update(1)

        else:
            if random.uniform(0,1)<=p_T:
                if random.uniform(0,1)<=(1-p_PA):
                    if len([x for x in G.nodes() if G.degree(x)>=2])==0:continue
                    v=random.choice([x for x in G.nodes() if G.degree(x)>=2])
                else:
                    if len([x for x in G.nodes() if G.degree(x)>=2])==0:continue
                    v=random.choices([x for x in G.nodes() if G.degree(x)>=2], weights=[G.degree(x)+1 for x in G.nodes() if G.degree(x)>=2],k=1)[0]
                
                target1=random.choice(list(G.neighbors(v)))
                options=[y for y in G.neighbors(v) if not G.has_edge(target1,y)]
                if len(options)==0: continue
                intra_inter= random.uniform(0, 1)
                if intra_inter<=h: choices=[x for x in options if G.nodes[v]['ground_truth']==G.nodes[x]['ground_truth']]
                else:choices=[x for x in options if G.nodes[v]['ground_truth']!=G.nodes[x]['ground_truth']]
                    
                if random.uniform(0, 1)<=(1-p_PA):weights=[1 for w in options]
                else: weights=[choices_weights_c[G.nodes[w]['ground_truth']][w] for w in options] 
                
                if len(options)==0: print("no ", intra_inter);continue
                target2=random.choices(options, weights=weights, k=1)[0]
                
                G.add_edge(target1, target2)
                choices_weights_c[G.nodes[target1]['ground_truth']][target1]+=1
                choices_weights_c[G.nodes[target2]['ground_truth']][target2]+=1
                
                
            else:
                if random.uniform(0,1)<=(1-p_PA):
                    v=random.choice([x for x in G.nodes() ])
                else:
                    v=random.choices([x for x in G.nodes() ], weights=[G.degree(x)+1 for x in G.nodes()],k=1)[0]
                    
                neigh=list( G.neighbors(v))
                options=[x for x in G.nodes() if x not in neigh]
                intra_inter= random.uniform(0, 1)
                if intra_inter<=h: choices=[x for x in options if G.nodes[v]['ground_truth']==G.nodes[x]['ground_truth']]
                else:choices=[x for x in options if G.nodes[v]['ground_truth']!=G.nodes[x]['ground_truth']]

                if random.uniform(0, 1)<=(1-p_PA):weights=[1 for v in choices]
                else:weights=[choices_weights_c[G.nodes[v]['ground_truth']][v] for v in choices] 

                if len(choices)==0:continue
                target=random.choices(choices, weights=weights, k=1)[0]
                if (intra_inter>h and random.uniform(0,1)<=r[G.nodes[target]['ground_truth']]/r[G.nodes[v]['ground_truth']]) or intra_inter<h :
                    G.add_edge(v, target)

                    choices_weights_c[G.nodes[v]['ground_truth']][v]+=1
                    choices_weights_c[G.nodes[target]['ground_truth']][target]+=1
        
    return G


def deepwalk_walks(H, walk_length, num_walks):
    '''
    Create random walks with deepwalk's method
    '''
    G = nx.Graph()
    G.add_nodes_from([str(n) for n in list(H.nodes)])
    G.add_edges_from([(str(e0), str(e1)) for e0, e1 in list(H.edges)])

    source_nodes = list(G)
    
    walks = []

    for _ in range(num_walks):

        for source in source_nodes:
            walk = [source]

            while len(walk) < walk_length:
                options = [n for n in G.neighbors(walk[-1])]
                next_node = np.random.choice(options)
                walk.append(next_node)
            walks.append(walk)

    return walks

def transform_sklearn_labels_to_communities(labels: list):
    pred_coms = [[] for i in range(len(np.unique(labels)))]
    for idx, label in enumerate(labels):
        pred_coms[label].append(idx)
    return pred_coms

def model_to_pred(G, model, n_clusters):
    emb_df = pd.DataFrame([model.wv.get_vector(str(n)) 
        for n in G.nodes()], index=G.nodes)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(emb_df)
    pred_coms = transform_sklearn_labels_to_communities(labels=kmeans.labels_)
    pred = NodeClustering(communities=pred_coms, graph=G)
    
    return pred

def walk_family(G, variant, n_clusters, dimensions, walk_length, num_walks):
    if variant == 'deepwalk':
        walks = deepwalk_walks(G, walk_length, num_walks) # walks is a list of walks (list of nodes)
        model = gensim.models.Word2Vec(walks, size=dimensions)

    elif variant == 'fairwalk':
        n = len(G.nodes())
        node2group = {node: group for node, group in zip(G.nodes(), 
            (5*np.random.random(n)).astype(int))}
        nx.set_node_attributes(G, node2group, 'group')

        model = FairWalk(G, dimensions=dimensions, walk_length=walk_length, 
            num_walks=num_walks, quiet=True)  # Use temp_folder for big graphs
        # model = model.fit(window=10, min_count=1, batch_words=4)
        model = model.fit()
    elif variant == 'node2vec':
        emb = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, 
            num_walks=num_walks, p=1, q=1, quiet=True)
        model = emb.fit() # model is a gensim.model.Word2Vec

    pred_coms = model_to_pred(G, model, n_clusters)
    return pred_coms


def remove_prefix(s, prefix):
    return s[len(prefix):] if s.startswith(prefix) else s


def path_prefix(dir_path, category, create=False):

    if category == 'data':
        input_prefix = 'data/'
    elif category == 'data_applied_methods':
        input_prefix = 'data_applied_methods/'
    elif category == 'results':
        input_prefix = 'results/'
    elif category == 'figures':
        input_prefix = 'figures/'
    else:
        print('Error: input correct category')
        exit()

    if input_prefix in dir_path:
        correct = dir_path
    else:
        correct = input_prefix + dir_path

    if not os.path.exists(correct):
        if not create:
            print('Directory does not exist:', correct)
            exit()
        else:
            os.makedirs(correct)
    return correct