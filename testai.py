#import libraries 
from txtai.embeddings import Embeddings
import json
import pandas as pd

#import prelearned ai
embeddings = Embeddings({
    
    "path": "sentence-transformers/all-MiniLM-L6-v2"
})

#load data into dataframe
with open("data.json", "r") as f:
    json_data = json.load(f)
    data = pd.DataFrame(json_data)

#filter out duplicates
data.drop_duplicates(subset = 'FAULT_ID', inplace = True, keep = "first")
#reset indexes to not break the range functions later on
data.reset_index(drop = True, inplace = True)

#Have ai index all text
txtai_data = []
for text, row in data.iterrows():
    txtai_data.append((text, row['FAULT_LINE_TEXT'], None))
embeddings.index(txtai_data)

def ULists():
    #Create List of all clusters with size 1
    UList1 = []
    for i in range(0,100):
        res = embeddings.search(data["FAULT_LINE_TEXT"][i], 20)
        list= []
        for r in res:
            #Similarity of 0.5 or more
            if r[1] > 0.5:
                list.append(data["FAULT_ID"][r[0]])
        if len(list) == 1:
            UList1.append(list)

    UList2 = []
    for i in range(0,100):
        res = embeddings.search(data["FAULT_LINE_TEXT"][i], 20)
        list= []
        for r in res:
            if r[1] > 0.6:
                list.append(data["FAULT_ID"][r[0]])
        if len(list) == 1:
            UList2.append(list)

    UList3 = []
    for i in range(0,100):
        res = embeddings.search(data["FAULT_LINE_TEXT"][i], 20)
        list= []
        for r in res:
            if r[1] > 0.7:
                list.append(data["FAULT_ID"][r[0]])
        if len(list) == 1:
            UList3.append(list)

    UList4 = []
    for i in range(0,100):
        res = embeddings.search(data["FAULT_LINE_TEXT"][i], 20)
        list= []
        for r in res:
            if r[1] > 0.8:
                list.append(data["FAULT_ID"][r[0]])
        if len(list) == 1:
            UList4.append(list)

    UList5 = []
    for i in range(0,100):
        res = embeddings.search(data["FAULT_LINE_TEXT"][i], 20)
        list= []
        for r in res:
            if r[1] > 0.9:
                list.append(data["FAULT_ID"][r[0]])
        if len(list) == 1:
            UList5.append(list)
            
#Printing Lists for now            
    print(UList1)
    print(UList2)
    print(UList3)
    print(UList4)
    print(UList5)

def GLists():
    #Find Generalized Lists 
    GList1 = []
    for i in range(0,100):
        res = embeddings.search(data["FAULT_LINE_TEXT"][i], 20)
        list= []
        for r in res:
            #Similarity of 0.5 or more
            if r[1] > 0.5:
                list.append(data["FAULT_ID"][r[0]])
        #Only add to GList if length is greater than 1
        if len(list) > 1:
            GList1.append(list)

    GList2 = []
    for i in range(0,100):
        res = embeddings.search(data["FAULT_LINE_TEXT"][i], 20)
        list= []
        for r in res:
            if r[1] > 0.6:
                list.append(data["FAULT_ID"][r[0]])
        if len(list) > 1:
            GList2.append(list)

    GList3 = []
    for i in range(0,500):
        res = embeddings.search(data["FAULT_LINE_TEXT"][i], 20)
        list= []
        for r in res:
            if r[1] > 0.7:
                list.append(data["FAULT_ID"][r[0]])
        if len(list) > 1:
            GList3.append(list)

    GList4 = []
    for i in range(0,100):
        res = embeddings.search(data["FAULT_LINE_TEXT"][i], 20)
        list= []
        for r in res:
            if r[1] > 0.8:
                list.append(data["FAULT_ID"][r[0]])
        if len(list) > 1:
            GList4.append(list)

    GList5 = []
    for i in range(0,100):
        res = embeddings.search(data["FAULT_LINE_TEXT"][i], 20)
        list= []
        for r in res:
            if r[1] > 0.9:
                list.append(data["FAULT_ID"][r[0]])
        if len(list) > 1:
            GList5.append(list)

#Printing Lists for now
    #return(GList1)
    #return(GList2)
    print(GList3)
    #return(GList4)
    #return(GList5)

#Call Functions wanted to run

def mech(GList):
    gen = []
    print(GList)
    for i in range(len(GList)):
        for j in range(len(GList)):
            if GList[i] != GList[j]:
                com = [item for item in GList[i] if item in GList[j]]
                if com:
                    gen.append(com)
    print(gen)

GLists()
