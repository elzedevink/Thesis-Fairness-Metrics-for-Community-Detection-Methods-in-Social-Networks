import random
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from cdlib import NodeClustering, evaluation

from my_module import *


def generate_LFR(dir_path_LFR, n, graphs_per_category, mus):
    """
        Generate LFR graphs with the following parameters:
        - n: number of nodes
        - tau1: Power law exponent for degree distribution (>1)
        - tau2: Power law exponent for community size distribution (>1)
        - mu: mixing parameter, fraction of inter-community edges for each node (in 
            interval [0,1])
        - average_degree: average degree (either this or min_deg needs to be given)
        - min_degree: minimum degree (either this or avg_deg needs to be given)
        - max_degree: maximum degree
        - min_community: minimum community size
        - max_community: maximum community size
    """
    
    tau1 = 2 # 2 or 3, second paper sets 2
    tau2 = 2.5 # 2 or 3, per second paper
    average_degree = 20
    max_degree = 100
    min_community = 20 
    seed = 0

    for mu in mus:
        success = 0

        while success < graphs_per_category:
            random_state = np.random.RandomState(seed)

            try:
                G = nx.LFR_benchmark_graph(n=n, tau1=tau1, tau2=tau2, mu=mu, 
                    average_degree=average_degree, max_degree=max_degree, 
                    min_community=min_community, seed=random_state)

                communities = list({frozenset(G.nodes[v]["community"]) for v in G})
                communities.sort(key=len, reverse=True)
                
                for node in G.nodes():
                    for i, c in enumerate(communities):
                        if node in c:
                            G.nodes[node]['ground_truth'] = i
                    del G.nodes[node]['community']

                # show_degree_dist(G)
                # show_community_dist(G)
                # show_community_conductance(G)

                success += 1
                store_network(G, f'SG_mu{int(mu*10)}_s{seed}', dir_path_LFR)

            except Exception as error:
                print(error)

            seed += 1

# generate the 5 synthetic networks from the paper:
# 'Fairness-aware fake news mitigation using counter information propagation'
def generate_paper_networks():
    path = 'data/manual/'
    
    n = 10000
    r = [0.54, 0.3, 0.15, 0.005, 0.005]
    h_array = [0.9, 0.9, 0.9, 0.9, 0.2]
    p_PA_array = [0.9, 0.9, 0.1, 0.9, 0.9]
    p_N_array = [1/(40), 1/(10), 1/(20), 1/(20), 1/(20)]
    p_T_array = [0.9, 0.9, 0.9, 0.2, 0.9]

    for g in range(5):
        name = str(g)
        G = original_hichba(n,r,h_array[g],p_PA_array[g],p_N_array[g],p_T_array[g])
        store_network(G, name, path)


# manually set hichba parameters and store network
def manual_hichba():
    name = 'example'
    path = 'data/manual'

    n = 30
    # r = [0.1, 0.2, 0.3, 0.4]
    r = [1/6, 1/6, 4/6]
    h = 0.9
    p_PA = 0.8
    p_N = 1/4
    p_T = 0.3

    while True:
        G = hichba(n, r, h, p_PA, p_N, p_T)
        G.remove_edges_from(nx.selfloop_edges(G))

        com_dict = nx.get_node_attributes(G, 'ground_truth')
        num_gt0 = len([key for key in com_dict.keys() if com_dict[key] == 0])
        num_gt1 = len([key for key in com_dict.keys() if com_dict[key] == 1])
        num_gt2 = len([key for key in com_dict.keys() if com_dict[key] == 2])
        
        print(num_gt0)
        print(num_gt1)
        print(num_gt2)

        if num_gt0 == num_gt1 and num_gt0 == 5:
            break

    print(G)
    store_network(G, name, path)


def generate_HICH_BA(dir_path, graphs_per_category, n, r):
    h = 0.9
    p_PA = 0.8
    p_N = 1/10
    p_T = 0.3
    
    for i in range(graphs_per_category):
        net_name = 'SG_c' + str(len(r)) + '_n' + str(i)
        
        G = original_hichba(n, r, h, p_PA, p_N, p_T)
        print(nx.selfloop_edges(G))

        store_network(G, net_name, dir_path)

if __name__ == '__main__':
    # dir_path_LFR = 'data/synthetic/LFR_0_to_1'
    # mus = [x/10 for x in range(1, 11)]
    # generate_LFR(dir_path_LFR, 1000, 1, mus)

    # dir_path_LFR = 'data/synthetic/LFR_small_246'
    # generate_LFR(dir_path_LFR, 1000, 50, [0.2, 0.4, 0.6])

    # dir_path_LFR = 'data/synthetic/LFR_large_246'
    # generate_LFR(dir_path_LFR, 10000, 5, [0.2, 0.4, 0.6])

    # manual_hichba()
    

    # dir_path_HICH_BA = 'data/synthetic/HICH_BA/multiple_min'
    # r = [0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.9]
    # generate_HICH_BA(dir_path_HICH_BA, 5, 10000, r)

    # dir_path_HICH_BA = 'data/synthetic/HICH_BA/multiple_maj'
    # r = [0.3, 0.3, 0.3, 0.03, 0.03, 0.03, 0.003, 0.003, 0.003]
    # generate_HICH_BA(dir_path_HICH_BA, 5, 10000, r)


