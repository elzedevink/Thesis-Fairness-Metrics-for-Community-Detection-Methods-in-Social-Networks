import numpy as np
import networkx as nx
from abcd_graph import Graph, ABCDParams

from my_module import *

def get_zeta(n, max_degree):
    zeta = np.log(max_degree) / np.log(n)
    
    return zeta

def get_tau(n, max_community_size):
    tau = np.log(max_community_size) / np.log(n)

    return tau

def get_xi(G, comm_assignment_dict, mu):
    comm_assignment_dict2 = {node: value['ground_truth'] 
        for node, value in comm_assignment_dict.items()}

    num_coms = max(comm_assignment_dict2.values()) + 1
    volume_per_com = np.zeros(num_coms)

    for node, comm in comm_assignment_dict2.items():
        volume_per_com[comm] += nx.degree(G, node)
    total_volume = sum(volume_per_com)

    vol_sum = 0
    for i in range(num_coms):
        vol_sum += (volume_per_com[i] / total_volume) ** 2
    mu_0 = 1 - vol_sum

    xi = mu / mu_0

    return xi


def generate_ABCD(dir_path_LFR):
    graphs_per_category = 5

    # LFR settings
    LFR_tau1 = 2 # Power law exponent for degree distribution
    LFR_tau2 = 2.5 # Power law exponent for community size distribution
    LFR_average_degree = 20
    LFR_max_degree = 100
    LFR_min_community = 20

    # LFR values found from networks, see LFR_info()
    LFR_min_degree = 8
    LFR_max_community_size = 100

    """
        gamma	Power-law parameter for degrees, between 2 and 3
        delta	Min degree
        zeta	Parameter for max degree, between 0 and 1
        beta	Power-law parameter for community sizes, between 1 and 2
        s		Min community size
        tau		Parameter for max community size, between zeta and 1
        xi		Noise parameter, between 0 and 1
    """

    n = 10000
    mus = [0.2, 0.4, 0.6]

    # from LFR_info()
    xi_dict = {
        0.2: 0.20140185949631056,
        0.4: 0.4023125370927575, 
        0.6: 0.6029478696275621
    }
    
    count = 0
    for mu in mus:
        for _ in range(graphs_per_category):
            net_name = 'SG_xi'+str(int(mu*10))+'_n'+str(count)
            print(net_name)

            count += 1

            params = ABCDParams(
                gamma = LFR_tau1,
                delta = LFR_min_degree,
                zeta = get_zeta(n, LFR_max_degree),
                beta = LFR_tau2,
                s = LFR_min_community,
                tau = get_tau(n, 120),
                xi = xi_dict[mu]
            )

            graph = Graph(params, n=n, logger=True).build()
            
            communities = [community._vertices for community in graph._graph.communities]
            com_dict = {}

            for i in range(len(communities)):
                for node in communities[i]:
                    com_dict[node] = {'ground_truth': i}

            G = graph.exporter.to_networkx()
            
            nx.set_node_attributes(G, com_dict)
            
            store_network(G, net_name, dir_path_LFR)

"""
    Get values for xi by looking at comparable LFR networks
"""
def LRF_info():
    dir_path_input = 'data/synthetic/LFR_large_246'
    files = os.listdir(dir_path_input)
    net_names = sorted(list(set([file[:-10] for file in files 
            if file[-10:] == '_nodes.csv'])))
    mus = [float(net_name[5])/10 for net_name in net_names] # mus = [0.2, 0.4, 0.6]

    xi_dict = {0.2: [], 0.4: [], 0.6: []}
    for net_name, mu in zip(net_names, mus):
        G, comm_assignment_dict = get_network_communities(dir_path_input, net_name, 
            request_node_clustering=False)
        ground_truth = to_NodeClustering(G, comm_assignment_dict)

        min_degree = min([nx.degree(G, n) for n in G.nodes()])
        print('min_degree:', min_degree)

        max_community_size = max([len(community) for community in ground_truth.communities])
        print('max_com_size:', max_community_size)
        
        xi = get_xi(G, comm_assignment_dict, mu)
        xi_dict[mu].append(xi)

    xi_2_avg = np.average(xi_dict[0.2])
    xi_4_avg = np.average(xi_dict[0.4])
    xi_6_avg = np.average(xi_dict[0.6])
    print('xi_2_avg', xi_2_avg)
    print('xi_4_avg', xi_4_avg)
    print('xi_6_avg', xi_6_avg)


if __name__ == '__main__':
    # LRF_info()

    dir_path_LFR = 'data/synthetic/ABCD'
    generate_ABCD(dir_path_LFR)
