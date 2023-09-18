from txtai.embeddings import Embeddings
import json
import pandas as pd

embeddings = Embeddings({
    
    "path": "sentence-transformers/all-MiniLM-L6-v2"
})

with open("filtered.json", "r") as f:
    data = json.load(f)["FAULT_LINE_TEXT"]

txtai_data = []
i=0
for text in data:
    txtai_data.append((i, text, None))
    i=i+1


embeddings.index(txtai_data)

for r in range(0,100):
    res = embeddings.search(data[r], 20)
    
    list= []
    for r in res:
        if r[1] > 0.5:
            list.append(r[0])

    print(list)






