# Import libraries
from txtai.embeddings import Embeddings
import json
import pandas as pd
from txtai.pipeline import Similarity
# Import pre-learned AI
embeddings = Embeddings({
    "path": "sentence-transformers/all-MiniLM-L6-v2"
})

# Load data into a DataFrame
with open("data.json", "r") as f:
    json_data = json.load(f)
    data = pd.DataFrame(json_data)

# Filter out duplicates
data.drop_duplicates(subset='FAULT_ID', inplace=True, keep="first")
# Reset indexes to not break the range functions later on
data.reset_index(drop=True, inplace=True)

#Have ai index all text
txtai_data = []
for text, row in data.iterrows():
    txtai_data.append((text, row['FAULT_LINE_TEXT'], None))
embeddings.index(txtai_data)



em = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})
n = data[['FAULT_ID','FAULT_LINE_TEXT']]
i = 0

n1 = []   #list of unique elements
n2 = []   #list of nonunique elements and their similarity scores
while i <= len(n):
    #Produce Similarity scores
    res = em.similarity(n['FAULT_LINE_TEXT'][i],n['FAULT_LINE_TEXT'])
    #Filter scores
    res = [t for t in res if t[1]>=0.5]
    #Define m for this loop
    m = []

    #Remake the list of tuples into a list of lists
    for tup in res:
        res2 = list(tup)
        m.append(res2)

    #Replace Indexes with Fault ID
    for a in range(len(m)):
        m[a][0] = n['FAULT_ID'][a]

    #Filter into N1 or N2 based on list length
    if len(m) == 1:
        #If unique add faultID to n1 and drop from n
        n1.append(m[0][0])
        dropped = m[0][0]
        n = n[n['FAULT_ID'] != dropped]
    else:
        #If nonunique add full list to 
        n2.append(m)
    i=i+1

#Remove from N2 any FAULT_IDs in N1 that were not removed previously in the Do-While loop
filtered_n2 = [inner_list for inner_list in n2 if inner_list[0] not in n1]


