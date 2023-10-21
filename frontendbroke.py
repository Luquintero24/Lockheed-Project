import streamlit as st
import random
import csv
import pandas as pd
from txtai.embeddings import Embeddings
csv_file_path = 'Action_Texts.csv'
data = pd.read_csv(csv_file_path)

em = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})
st.title('Artificial Intelligence (AI) Assisted Form Generation')

title = st.text_input("Enter Fault: ")
index= None
placeholder= ""
ind = None

res = em.similarity(title, data["FAULT_LINE_TEXT"])
res = res[0:3]
option_1 = res[0][0]
option_2 = res[1][0]
option_3 = res[2][0] 
ind= None
option = st.selectbox('Which one is the best fit',(data["FAULT_LINE_TEXT"][option_1], data["FAULT_LINE_TEXT"][option_2], data["FAULT_LINE_TEXT"][option_3]), index=None, placeholder="Select best fit...")


if option == data["FAULT_LINE_TEXT"][option_1]:
    ind = option_1
    
elif option == data["FAULT_LINE_TEXT"][option_2]:
    ind = option_2
    
elif option == data["FAULT_LINE_TEXT"][option_3]:
    ind = option_3
    
if ind is not None:
    fault = st.text_input("Fault Line Text:", data['FAULT_LINE_TEXT'][ind])
    acserial = st.number_input("AC Serial: ")
    date = st.date_input("Fault Date: ")
    random_number = random.randint(1, 10000)
    faultid = st.number_input("Fault ID: ",random_number)
    datea = st.date_input("Action Date: ")
    action = st.text_input("Action Text: ", data['ACTION_TEXT'][ind]) 
    p= data['P'][ind]
    for i in range(1, p + 1):
        st.write(f"Data for Removed and Installed {i}")
        removenumber = st.text_input(f"Removed Part Number {i}")
        removeserial = st.text_input(f"Removed Part Serial {i}")
        installnumber = st.text_input(f"Installed Part Number {i}")
        installserial = st.text_input(f"Installed Part Serial {i}")
    wdc = st.text_input("WDC: ", data['WDC'][ind])
    wdc_description = st.text_input("WDC Description: ", data['WDC Description'][ind])
    tmc = st.text_input("TMC: ", data['TMC'][ind])
    tmc_description = st.text_input("TMC Description: ", data["TMC Description"][ind])
    hmc = st.text_input("HMC: ", data["HMC"][ind])
    hmc_description = st.text_input("HMC Description: ", data["HMC Description"][ind])


