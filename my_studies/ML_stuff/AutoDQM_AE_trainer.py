import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.linalg import svd

import yaml
from yaml import Loader

import plotting
import utils
import flow_for_anomalies

"""
TODO:

- Make the model more expressive
    - RES nets shoudl do it
    - Should I also try 1d conv?
- Try the normalized AR
- Try better pre-processing (std, log, etc)
- Should we also give the bin as input? Or only the values?
- READ THE BEST VALIDATION EPOCH MODEL!!
"""

# For some reason pytorch vannila dtaaset was saying the batches were lists???
class CustomTensorDataset(TensorDataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors):
        """
        Arguments:
            tensors (Tensor): Contains sample tensors.
        """
        # Ensure the input is a tensor
        if not isinstance(tensors, torch.Tensor):
            raise TypeError('The input data should be a PyTorch tensor.')
        self.tensors = tensors

    def __getitem__(self, index):
        # Return a single tensor sample at given index
        return self.tensors[index]

    def __len__(self):
        return self.tensors.size(0)  # The number of items in the dataset


# First we need to load the data
file = pd.read_parquet("../../tutorial/Muon.parquet")
print('file read')

## Lets concatenate all the histograms into a single array
good_runs_index = file["label"] == 0
bad_runs_index  = file["label"] == 1

concatenated_good_runs = np.array([])
for key in file.keys():
    if "hResDist_" in key:
        concatenated_good_runs = np.concatenate([concatenated_good_runs, file[key][good_runs_index].values])

# Since it's already properly structured, you can directly reshape:
"""
histograms = np.vstack([(array/(np.sum(array)+0.1)).reshape(1, -1) for array in concatenated_good_runs])

concatenated_good_runs = histograms

# Lets clean the histograms of any NaNs or None values
concatenated_good_runs = [arr for arr in concatenated_good_runs if arr is not None 
                  and not np.isnan(arr).any() 
                  and not np.isinf(arr).any()]
""" 

# Assuming concatenated_good_runs is your array of arrays
# Assuming concatenated_good_runs is your array of arrays
cleaned_arrays = [arr for arr in concatenated_good_runs if arr is not None 
                  and not np.isnan(arr).any() 
                  and not np.isinf(arr).any()]

cleaned_arrays = cleaned_arrays/np.sum(cleaned_arrays)

# Identify rows with inf or NaN entries
rows_to_keep = ~(np.isnan(cleaned_arrays).any(axis=1) | np.isinf(cleaned_arrays).any(axis=1))

# Filter out rows with inf or NaN entries
cleaned_arrays = cleaned_arrays[rows_to_keep]

# Creating some anomalies!!
made_anomalies = plotting.create_and_plot_anomalies(cleaned_arrays)
#made_anomalies = made_anomalies / np.sum(made_anomalies)

global_mean = np.mean(cleaned_arrays)
global_std = np.std(cleaned_arrays)

# Apply Z-score normalization
normalized_arrays    = [(arr - global_mean) / global_std for arr in cleaned_arrays]
normalized_anomalies = [(arr - global_mean) / global_std for arr in made_anomalies]

# Normalizing them to one! This has to be done since we added more event into the anomalies!
#normalized_anomalies = normalized_anomalies / np.sum(normalized_anomalies, axis=1)[:, None]
#normalized_arrays    = normalized_arrays / np.sum(normalized_arrays, axis=1)[:, None]
#print(np.sum(normalized_anomalies,axis =1), np.sum( made_anomalies, axis = 1 )  ,  np.sum( normalized_arrays, axis = 1 ) )

plotting.validate_nominal_and_anomalies(normalized_arrays , normalized_anomalies)
############################################################################################################

concatenated_good_runs = normalized_arrays
concatenated_good_runs = np.nan_to_num(concatenated_good_runs)

# Split the data into validation and test sets
train_data, test_data = train_test_split(concatenated_good_runs, test_size=0.15, random_state=42)

# Convert arrays to PyTorch tensors
train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor  = torch.tensor(test_data, dtype=torch.float32)
anomalies_tensor = torch.tensor(normalized_anomalies, dtype=torch.float32)

## Lets save these tensors so one does not have to run the above code again
# Save train_tensor
torch.save(train_tensor, 'train_tensor.pth')

# Save test_tensor
torch.save(test_tensor, 'test_tensor.pth')

print('Number of training examples:', len(train_tensor), ' and Testing examples:', len(test_tensor))

# Create TensorDataset
train_dataset = CustomTensorDataset(train_tensor)
test_dataset = CustomTensorDataset(test_tensor)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)

# Loop to read over network condigurations from the yaml file: 
stream     = open('config.yaml' , 'r')
dictionary = yaml.load(stream,Loader)

avaliable_models = ["autoencoder", "1dconv_flow" ,"flow","feature_flow","pca"]
# Checking if the entries on yaml are avaliable
for key in dictionary:
    if key["name"] not in avaliable_models:
        print('The model you are trying to use is not avaliable! Please choose one of the following: ', avaliable_models)
        exit()

# Loop over the models and performing the training
for key in dictionary:
    
    if key["name"] == "flow":
        print('normalizing flow model training was selected! Begin training!') 

        # This is the 'simple' flow model!
        flow_model = flow_for_anomalies.normalizing_flow(train_loader, test_loader, anomalies_tensor , n_flow_layers = key["n_transforms"] , n_hidden_features = key["n_nodes"], n_hidden = key["n_layers"], input_dim = len(concatenated_good_runs[0]), n_epochs = 1)

    if key["name"] == "feature_flow":
        print('Feature normalizing flow model training was selected! Begin training!') 
        
        # This is not working yet! The log likelihood does not make sense!!
        feature_flow_model = flow_for_anomalies.feature_extraction_flow(train_loader, test_loader,anomalies_tensor,  n_flow_layers = key["n_transforms"] , n_hidden_features = key["n_nodes"] , n_hidden = key["n_layers"], input_dim = len(concatenated_good_runs[0]), n_epochs = 10)

    if key["name"] == "autoencoder":
        print('Autoencoder model training was selected! Begin training!') 

        Autoencoder = utils.train_AE(train_loader, test_loader, input_size = len(concatenated_good_runs[0]), encoding_dim = 16, epochs = 10, global_mean = global_mean, global_std = global_std)
        model_ae    = Autoencoder.train_AE()

    if key["name"] == "1dconv_flow":
        print('1D Convolutional flow model training was selected! Begin training!') 

        ConAutoencoder = utils.train_ConvAE(train_loader, test_loader, input_size = len(concatenated_good_runs[0]), encoding_dim = 16, epochs = 10)
        model_Convae = ConAutoencoder.train_AE()
        
    else:
        print('ERROR! The model you are trying to use is not avaliable! Please choose one of the following: ', avaliable_models)
        exit()
