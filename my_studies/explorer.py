# Script made to take a look at the output of the files produced by AutoDQM ML and setup the training
import pandas
import numpy as np

df = pandas.read_parquet("../tutorial/Muon.parquet")
print(df.columns)
print( df.run_number )

# Dealing with the histograms...

import awkward
histograms = awkward.from_parquet("../tutorial/Muon.parquet")
#print(histograms.fields)

# Number of entries
print('Entries:\n')
print(awkward.sum(histograms["DT/Run summary/03-LocalTrigger-TM/Task/TM_TrigEffNum_W0"], axis = -1))

# Number of bins
print('Bins:\n')
print(awkward.count(histograms["DT/Run summary/03-LocalTrigger-TM/Task/TM_TrigEffNum_W0"],axis=-1))

print('Shape:\n')
print(np.shape(histograms["DT/Run summary/03-LocalTrigger-TM/Task/TM_TrigEffNum_W0"][0]))
#print(awkward.shape(histograms["DT/Run summary/03-LocalTrigger-TM/Task/TM_TrigEffNum_W0"]))