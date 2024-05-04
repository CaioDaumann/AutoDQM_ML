import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.linalg import svd

import utils

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


global_mean = np.mean(cleaned_arrays)
global_std = np.std(cleaned_arrays)

# Apply Z-score normalization
normalized_arrays = [(arr - global_mean) / global_std for arr in cleaned_arrays]

concatenated_good_runs = normalized_arrays
concatenated_good_runs = np.nan_to_num(concatenated_good_runs)

# Split the data into validation and test sets
train_data, test_data = train_test_split(concatenated_good_runs, test_size=0.1, random_state=42)

# Convert arrays to PyTorch tensors
train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor  = torch.tensor(test_data, dtype=torch.float32)

## Lets save these tensors so one does not have to run the above code again
# Save train_tensor
torch.save(train_tensor, 'train_tensor.pth')

# Save test_tensor
torch.save(test_tensor, 'test_tensor.pth')

print('Number of training examples:', len(train_tensor), ' and Testing examples:', len(test_tensor))

# Would also be probably nice to std scale the data and maybe a log transform? Because some values are way higher than others! 
# And they have very diferent ranges ... 

# Create TensorDataset
train_dataset = CustomTensorDataset(train_tensor)
test_dataset = CustomTensorDataset(test_tensor)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=640, shuffle=False)

# Training the AE model!
print('Training the AE model with linear layers!')
Autoencoder = utils.train_AE(train_loader, test_loader, input_size = len(concatenated_good_runs[0]), encoding_dim = 16, epochs = 10, global_mean = global_mean, global_std = global_std)
model_ae    = Autoencoder.train_AE()

# Training the AE model with 1D convolutions!
print('Training the AE model with 1D convolutions!')
ConAutoencoder = utils.train_ConvAE(train_loader, test_loader, input_size = len(concatenated_good_runs[0]), encoding_dim = 16, epochs = 10)
model_Convae = ConAutoencoder.train_AE()

# Training the PCA model!
from sklearn.decomposition import PCA

# Train PCA
# Perform PCA with desired number of components
# Train PCA

# de-standardize the data
train_data, test_data = train_data * global_std + global_mean, test_data * global_std + global_mean

print('Begining of the PCA training!')
pca = PCA(n_components=8)
pca.fit(train_data[:8])
print('PCA model trained!')

# Project data onto lower-dimensional space
train_data_projected = pca.transform(train_data)
test_data_projected  = pca.transform(test_data)

# Reconstruct data by transforming projected data back
train_data_reconstructed = pca.inverse_transform(train_data_projected)
test_data_reconstructed  = pca.inverse_transform(test_data_projected)

# Calculate sum of squared errors (SSE)
train_sse = np.sum((train_data - train_data_reconstructed) ** 2)/len(train_data)
test_sse = np.sum((test_data - test_data_reconstructed) ** 2)/len(test_data)

print("Train SSE:", np.mean(train_sse))
print("Test SSE:", np.mean(test_sse))

# Now lets validate this here!
# Assuming test_indices is a list of indices of histograms in test_data that you want to compare
num_examples = min(4, len(test_data))  # Number of histograms to compare (up to 5)

plt.figure(figsize=(15, 4 * num_examples))
for i in range(num_examples):
    index = np.random.randint(len(test_data))  # Randomly select an index from test_data
    
    # Original histogram
    plt.subplot(num_examples, 2, 2*i+1)
    plt.plot(test_data[index], label='Original', color='blue', linewidth=2)
    plt.title(f'Original Histogram {index}')
    plt.xlabel('Bin')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Reconstructed histogram
    plt.subplot(num_examples, 2, 2*i+2)
    plt.plot(test_data_reconstructed[index], label='Reconstructed', color='red', linewidth=2)
    plt.title(f'Reconstructed Histogram {index}')
    plt.xlabel('Bin')
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.savefig('PCA_reconstruction.png')

# Training the Flow model!

# Training the normalized AE model!