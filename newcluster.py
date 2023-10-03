# Import libraries
from txtai.embeddings import Embeddings
import time
import json
import pandas as pd
from txtai.pipeline import Similarity
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster


em = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})

# Load data into a DataFrame
with open("data.json", "r") as f:
    json_data = json.load(f)
    data = pd.DataFrame(json_data)

# Filter out duplicates
data.drop_duplicates(subset='FAULT_ID', inplace=True, keep="first")
# Reset indexes to not break the range functions later on
data.reset_index(drop=True, inplace=True)

em.index(data)




n = data[['FAULT_ID','FAULT_LINE_TEXT']][:10]

# i = 0

# n1 = []   #list of unique elements
# n2 = []   #list of nonunique elements and their similarity scores
# # while i <= 5:
#     #Produce Similarity scores
#     res = em.similarity(n['FAULT_LINE_TEXT'][i],n['FAULT_LINE_TEXT'])
#     #Filter scores
#     res = [t for t in res if t[1]>=0.5]
#     #Define m for this loop
#     m = []

#     #Remake the list of tuples into a list of lists
#     for tup in res:
#         res2 = list(tup)
#         m.append(res2)

#     #Replace Indexes with Fault ID
#     for a in range(len(m)):
#         m[a][0] = n['FAULT_ID'][a]

#     #Filter into N1 or N2 based on list length
#     if len(m) == 1:
#         #If unique add faultID to n1 and drop from n
#         n1.append(m[0][0])
#         dropped = m[0][0]
#         n = n[n['FAULT_ID'] != dropped]
#     else:
#         #If nonunique add full list to 
#         n2.append(m)
#     i=i+1

# #Remove from N2 any FAULT_IDs in N1 that were not removed previously in the Do-While loop
# filtered_n2 = [inner_list for inner_list in n2 if inner_list[0] not in n1]

# print(n2)

# # print("n1 will be", n1)
# start_time = time.time()
i=0
sorted_list = []
i = 0  # Initialize i
sort = []  # Initialize an empty list for sorting results

while i < 10:
    if i < len(n["FAULT_LINE_TEXT"]):  # Check if i is within bounds
        res = em.similarity(n["FAULT_LINE_TEXT"][i], n["FAULT_LINE_TEXT"])
        sorted_res = sorted(res, key=lambda x: x[0])  # Sort the results
        second_elements = [item[1] for item in sorted_res]  # Extract second elements from tuples
        sort.append(second_elements)  # Append the second elements to the list
    
    i = i + 1  # Increment i

# Convert 'sort' into a NumPy array to create a matrix
matrix = np.array(sort)

link = linkage(matrix, method = 'complete')

#Create dendrogram
dendrogram(link, labels=['0','1','2', '3', '4', '5','6','7', '8', '9'])
plt.xlabel('Indexes')
plt.ylabel('Distance')
plt.title('Hierarchial Clustering')
plt.show()


# # Convert the list of lists to a NumPy matrix
# matrix = np.array(second_elements_list)

# # Now 'matrix' contains the second elements as a NumPy matrix, where each row corresponds to one list of tuples
# print(matrix)
# end_time = time.time()

# # Calculate the time taken
# elapsed_time = end_time - start_time
# print(f"Time taken: {elapsed_time} seconds")




# Cut the dendrogram at a distance of 0.87
threshold = 0.87
clusters = fcluster(link, threshold, criterion='distance')

# Create a dictionary to store the clusters
cluster_dict = {}
for idx, cluster_id in enumerate(clusters):
    if cluster_id not in cluster_dict:
        cluster_dict[cluster_id] = [idx]
    else:
        cluster_dict[cluster_id].append(idx)

# Convert the dictionary of clusters into a list of lists
clustered_data = list(cluster_dict.values())

# # # Print the clustered data
# # for i, cluster in enumerate(clustered_data):
# #     print(f"Cluster {i + 1}:")
# #     for idx in cluster:
# #         print(f"FAULT_ID: {n['FAULT_ID'][idx]}, FAULT_LINE_TEXT: {n['FAULT_LINE_TEXT'][idx]}")
# #     print()

print(clustered_data)

# # Cut the dendrogram at a distance of 0.87
# threshold = 0.87
# clusters = fcluster(link, threshold, criterion='distance')

# # Create a dictionary to store the clusters
# cluster_dict = {}
# for idx, cluster_id in enumerate(clusters):
#     if cluster_id not in cluster_dict:
#         cluster_dict[cluster_id] = [n['FAULT_ID'][idx]]
#     else:
#         cluster_dict[cluster_id].append(n['FAULT_ID'][idx])

# # Convert the dictionary of clusters into a list of lists
# clustered_data = list(cluster_dict.values())

# # Print the list of lists containing only FAULT_ID values for each cluster
# print(clustered_data)






