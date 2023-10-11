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

#Import Clusters
first_cluster = []
with open('first_clusters_full.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        conv = [int(element) for element in row]
        first_cluster.append(conv)

#Import Original Data
with open("data.json", "r") as f:
    json_data = json.load(f)
    data = pd.DataFrame(json_data)

#Define action text clustering function
def act_cluster(n):
    """Function clustering action texts of set n"""
    sort = []
    i = 0
    #retrieve similarity scores
    while i < len(n):
        if i < len(n["ACTION_TEXT"]):  # Check if i is within bounds
            res = em.similarity(n["ACTION_TEXT"][i], n["ACTION_TEXT"])
            sorted_res = sorted(res, key=lambda x: x[0])  # Sort the results
            second_elements = [item[1] for item in sorted_res]  # Extract second elements from tuples
            sort.append(second_elements)  # Append the second elements to the list
            i = i + 1
    matrix = np.array(sort)
    link = linkage(matrix, method = 'complete')
    #Set threshold (lower means more similar)
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
    return(clustered_indexes)

#Drop lines without FAULT_ID
data = data.dropna(subset=['FAULT_ID'])
# Reset indexes to not break the range functions later on
data.reset_index(drop=True, inplace=True)
data['FAULT_ID'] = data['FAULT_ID'].astype(int)
em.index(data)

#Initialize New Dataframe
M = pd.DataFrame()
P = 0

#Only first 10 elements for testing purposes
first_cluster = first_cluster[:10]
#print(first_cluster)
for sublist in first_cluster:
    #Initilize W and Y which serve the same purpose as X and Z
    W = pd.DataFrame()
    #If the cluster is alone place it into M immediately
    if len(sublist)==1:
        Y = pd.DataFrame()
        #Find all rows with the single cluster faultID
        filtered = data[data['FAULT_ID'] == sublist[0]]
        #reset indexes to insure they can be iterated through
        filtered.reset_index(drop=True, inplace=True)
        for index, row in filtered.iterrows():
            #Only add rows without removed and installed to M
            if row['REMOVED_PART_NUMBER'] == '' and row['INSTALLED_PART_NUMBER'] == '':
                W = pd.concat([W, row.to_frame().T])
            else:
                #count number of removed installed rows
                Y = pd.concat([Y, row.to_frame().T])
        P = len(Y)
        #Add column P to all rows in W
        W['P'] = P
        #Add W to M
        M = pd.concat([M, W], ignore_index=True)
    #If longer iterate over the cluster
    else:
        P = 0
        X = pd.DataFrame()
        #Iterate over the cluster/sublist
        for j, fil in enumerate(sublist):
            #Create filtered dataframe containing only the current fault ID in the loop
            filtered = data[data['FAULT_ID'] == fil]
            
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
        #Reset X's index so that it works in the act_cluster function and so the correct rows can be added to M
        X.reset_index(drop=True, inplace=True)
        #Cluster Action Text
        ac = act_cluster(X)
        print(ac)
        #Take the clusters and use returned clustered indexs of X to store in M
        for sublist in ac:
            madd = X.iloc[sublist[0]]
            #Turn madd into a dataframe so we can add P and concat it
            madd = madd.to_frame().T
            madd["P"] = P
            M = pd.concat([M, madd], ignore_index=True)
print(M)
#Next Step: Turn M into a CSV file
