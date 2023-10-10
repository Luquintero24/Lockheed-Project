from txtai.embeddings import Embeddings
import time
import json
import csv
import pandas as pd
from txtai.pipeline import Similarity
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster

em = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})


first_cluster = []
with open('first_clusters_full.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        conv = [int(element) for element in row]
        first_cluster.append(conv)

with open("data.json", "r") as f:
    json_data = json.load(f)
    data = pd.DataFrame(json_data)

def act_cluster(n):
    """Function clustering action texts of set n"""
    sort = []
    i = 0
    while i < len(n):
        if i < len(n["ACTION_TEXT"]):  # Check if i is within bounds
            n.reset_index(drop=True, inplace=True)
            res = em.similarity(n["ACTION_TEXT"][i], n["ACTION_TEXT"])
            sorted_res = sorted(res, key=lambda x: x[0])  # Sort the results
            second_elements = [item[1] for item in sorted_res]  # Extract second elements from tuples
            sort.append(second_elements)  # Append the second elements to the list
            i = i + 1
    matrix = np.array(sort)
    link = linkage(matrix, method = 'complete')

    threshold = 0.5
    clusters = fcluster(link, threshold, criterion='distance')

    # Create a dictionary to store the clusters
    cluster_dict = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = [idx]
        else:
            cluster_dict[cluster_id].append(idx)

    # Convert the dictionary of clusters into a list of lists
    clustered_indexes = list(cluster_dict.values())
    
    clustered_data = [[n.at[index, 'FAULT_ID'] for index in sublist] for sublist in clustered_indexes]
    return(clustered_data)

data = data.dropna(subset=['FAULT_ID'])
# Reset indexes to not break the range functions later on
data.reset_index(drop=True, inplace=True)
data['FAULT_ID'] = data['FAULT_ID'].astype(int)
em.index(data)

#Initialize New Dataframe
M = pd.DataFrame()
X = pd.DataFrame()
P = 0

#Only first 4 elements for testing purposes
first_cluster = [first_cluster[3]]
#print(first_cluster)
for sublist in first_cluster:
    #If the cluster is alone place it into M immediately
    if len(sublist)==1:
        filtered = data[data['FAULT_ID'] == sublist[0]]
        M = pd.concat([M, filtered], ignore_index=True)
    #If longer iterate over the list
    else:
        P = 0
        #Iterate over the cluster/sublist
        for j, fil in enumerate(sublist):
            #Create filtered dataframe containing only the current fault ID in the loop
            filtered = data[data['FAULT_ID'] == fil]
            #if len(filtered) >1 :
                #print("FIL: " ,filtered)
            #Set set Z to zero to hold all Removed Installed
            Z = pd.DataFrame()
            #Iterate over all fault ID rows 
            for index, row in filtered.iterrows():
                #If row has removed or installed values place in X to be clustered    
                if row['REMOVED_PART_NUMBER'] == '' and row['INSTALLED_PART_NUMBER'] == '':
                    X = pd.concat([X, row.to_frame().T])
                #Or place in Z to be counted
                else:
                    Z = pd.concat([Z, row.to_frame().T])
            if len(Z) > P:
                P = len(Z)
        print(X)
        ac = act_cluster(X)
        print(ac)
        


