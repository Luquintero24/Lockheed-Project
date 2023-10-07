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

data.drop_duplicates(subset='FAULT_ID', inplace=True, keep="first")
data = data.dropna(subset=['FAULT_ID'])
# Reset indexes to not break the range functions later on
data.reset_index(drop=True, inplace=True)
data['FAULT_ID'] = data['FAULT_ID'].astype(int)
em.index(data)

#Initialize New Dataframe
M = pd.DataFrame()

#Iterate through clusters
for sublist in first_cluster:
    #If the cluster is alone place it into M immediately
    if len(sublist)==1:
        filtered = data[data['FAULT_ID'] == sublist[0]]
        M = pd.concat([M, filtered], ignore_index=True)

print(M)

