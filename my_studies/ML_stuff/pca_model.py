import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.linalg import svd

# Training the PCA model!
from sklearn.decomposition import PCA

# Train PCA
# Perform PCA with desired number of components
# Train PCA

# Just the skeleton!!
class train_pca:
    def __init__(self, data):
        self.data = data
    def train_pca(self):
        return 0
    def evaluate_performance(self):
        return 0


# de-standardize the data
train_data, test_data = train_data * global_std + global_mean, test_data * global_std + global_mean

print('Begining of the PCA training!')
pca = PCA(n_components=6)
pca.fit(train_data[:10])
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