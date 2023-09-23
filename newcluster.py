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
n = data[['FAULT_ID']]
i = 0
m = []
n1 = []
n2 = []

res = em.similarity(data['FAULT_LINE_TEXT'][0],data['FAULT_LINE_TEXT'])
res = [t for t in res if t[1]>=0.5]
res1 = []

for tup in res:
    res2 = list(tup)
    res1.append(res2)



for a in range(len(res1)):
    res1[a][0] = n['FAULT_ID'][a]

print(res1)

