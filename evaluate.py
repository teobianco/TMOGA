from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
import numpy as np


dataset = 'expansion_contraction'

ground_truth = np.load(f'/Users/teo/Desktop/Tesi Magistrale/FASE 1/TMOGA/tmoga/dataset/benchmark_rtu/{dataset}/labels.npz')['labels']
community_labels_dic = np.load(f'/Users/teo/Desktop/Tesi Magistrale/FASE 1/TMOGA/output/{dataset}/communities.npz', allow_pickle=True)['communities'].tolist()

for i in len(community_labels_dic):
    labels = []
    for (k, v) in community_labels_dic[i].items():
        for node in v:
            labels[node] = k
    print(f'NMI: {normalized_mutual_info_score(ground_truth, labels)}')
    print(f'ARI: {adjusted_rand_score(ground_truth, labels)}')
    # print(f'Silhouette: {silhouette_score(ground_truth, labels)}')
