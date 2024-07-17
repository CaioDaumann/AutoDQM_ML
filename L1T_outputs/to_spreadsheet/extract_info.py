import pandas as pd
import numpy as np
import ast
import re
from ast import literal_eval


# Reading the file
path = "../../paper_studies/ohne_algos/l1t_bb.csv"
read_file = pd.read_csv(path)

i = 0
for key in read_file.keys():
    print(key)

#exit()

#print( read_file["L1T/Run summary/L1TStage2uGMT/ugmtMuonPt_nEntries_score_betabinom"] )
#exit()

#calculate_number_of_entries(read_file)

run_nummer  = read_file["run_number"]
first_array = read_file["L1T/Run summary/L1TStage2CaloLayer1/ecalOccupancy_chi2_score_betabinom"]
    


# Initialize the dictionary
column_names = []
column_names.append("run_number")
column_names.append("year")
column_names.append("label")
column_names.append("train_label")
for key in read_file.keys():
    if "_chi2_score" in key or "pull_score_" in key or "nEntries_score" in key:
        column_names.append(key)
    
arrays = [np.array(read_file[column_name]) for column_name in column_names ]

data = {}
for column_name, array in zip(column_names, arrays):
    if isinstance(array[0], (int, np.integer)):
        data[column_name] = array
    else:
        data[column_name] = [f"{number:.3f}".replace('.', ',') for number in array]

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df.head())

# Save the DataFrame as a CSV file
df.to_csv('data.csv', index=False)





