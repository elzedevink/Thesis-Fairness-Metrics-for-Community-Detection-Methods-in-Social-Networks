# Fairness Metrics for Community Detection Methods in Social Networks

This repository contains code written for my Master thesis titled "Fairness Metrics for Community Detection Methods in Social Networks". This thesis is available to read in pdf form. This work was supervised by Dr. Akrati Saxena and Dr. Frank Takes.

This work has been accepted to present at [NetSci-X 2025](https://netscix2025.iiti.ac.in/) and [NetSci-NL Symposium](https://www.netsci.nl/netscinl-symposium/). It will also be published as a full paper by [Complex Networks 2024](https://complexnetworks.org/). See:
[de Vink, E., Saxena, A.: Group fairness metrics for community detection methods in social networks (2024).](https://arxiv.org/abs/2410.05487)

A shortened version of my code is available [here](https://github.com/elzedevink/fairness-metrics-community-detection/). This only includes code to generate the results shown in the Complex Networks 2024 paper.

## Files
**generate_networks.py**: Generates LFR benchmark networks

**set_communities.py**: Applies community detection methods to network data

**get_results.py**: Calculates fairness metric Phi for size, density, and conductance

**create_figures.py**: gathers results, creates figures showing performance-fairness trade-off

**my_module.py**: Helper functions

## Folders
**data**: network data in csv files showing ground-truth communities in 'networkname_nodes.csv' and edge adjacency list in 'networkname_edges.csv'

**data_applied_methods**: same as **data** but with community assignments by the community detection methods

**results**: contains fairness metric Phi and performance values

**figures**: contains figures displaying the results. Only figures with NMI as the prediction accuracy metric are uploaded

**figures/thesis_text**: This folder includes figures present in the thesis and code used to create these figures

<img src="https://github.com/user-attachments/assets/d6db00f6-027d-45e3-a223-bbeafc4bcae2" alt="football_size_Phi_FCCN" width="450">

Example of fairness analysis for the football network regarding size with FCCN as the community-wise performance metric.


## Code usage
Example of using set_communities.py, get_results.py, and create_figures.py. We use the data in data/synthetic/example. The code requires the required directory to be provided after `-d`. set_community.py requires data from /data/, get_results.py requires data from /data_applied_methods.py/, and create_figures.py requires data from /results/.

Network generation
```
python3 generate_networks.py 
```

Applying community detection methods
```
python3 set_communities.py -d data/synthetic/example/
```

Get fairness metric and prediction accuracy results
```
python3 get_results.py -d data_applied_methods/synthetic/example/
```

Create figures showing fairness metric and prediction accuracy results
```
python3 create_figures.py -d results/synthetic/example/
```

