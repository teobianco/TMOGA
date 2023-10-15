import os
from tmoga import TMOGA, evaluation
import networkx as nx
import numpy as np

dataset = 'expansion_contraction'

adjs = np.load(f'/Users/teo/Desktop/Tesi Magistrale/FASE 1/TMOGA/tmoga/dataset/benchmark_rtu/{dataset}/graphs.npz',
               allow_pickle=True, encoding='latin1')['graph']
dynamic_network = [nx.from_scipy_sparse_matrix(x) for x in adjs.tolist()]

tmoga_model = TMOGA(dynamic_network, pop_size=100, max_gen=25)
solutions, solutions_population = tmoga_model.start()

os.mkdir(f'/Users/teo/Desktop/Tesi Magistrale/FASE 1/TMOGA/output/{dataset}')
np.savez(f'/Users/teo/Desktop/Tesi Magistrale/FASE 1/TMOGA/output/{dataset}/solutions.npz', solutions=solutions)

all_communities = []
for i in range(len(solutions)):
    communities = evaluation.parse_locus_solution(solutions[i])
    all_communities.append(communities)
    print(communities)

np.savez(f'/Users/teo/Desktop/Tesi Magistrale/FASE 1/TMOGA/output/{dataset}/communities.npz', communities=all_communities)
