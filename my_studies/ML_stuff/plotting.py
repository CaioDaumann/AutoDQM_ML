import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt

def custom_uniform_sample(num_samples):
    samples = np.zeros(num_samples)
    #range1 = (0.15, 0.45)
    #range2 = (0.55, 0.85)

    range1 = (0.15, 0.55)
    range2 = (0.55, 0.85)   
    
    # Calculate lengths of the intervals
    length1 = range1[1] - range1[0]
    length2 = range2[1] - range2[0]
    total_length = length1 + length2
    
    # Probability of choosing from each range
    prob_range1 = length1 / total_length
    
    for i in range(num_samples):
        if np.random.rand() < prob_range1:
            # Sample from the first range
            samples[i] = np.random.uniform(range1[0], range1[1])
        else:
            # Sample from the second range
            samples[i] = np.random.uniform(range2[0], range2[1])
            
    return samples

def create_and_plot_anomalies(test_set):
    
    ## Lets create a list to store all of the generated anomalies!
    anomaly_list, anomaly_list_stacked = [],[]
    number_of_anomalies = 12000
    
    for i in range(int(number_of_anomalies)):

        np.random.seed()

        # Set parameters for the Gaussian distribution
        mean    =  custom_uniform_sample(1) #np.random.uniform(0.15, 0.85) #0.8
        std_dev =  np.random.uniform(0.01, 0.1) #0.1
        num_samples = 12000

        # Assuming 'test_set' is defined somewhere else and contains valid data
        y = test_set[i]  # Ensure this index access is valid

        # Define the range of your data
        min_edge = 0
        max_edge = 1

        # Generate bin edges from the specified range
        num_bins = len(y)
        bin_edges = np.linspace(min_edge, max_edge, num_bins + 1)

        # Generate Gaussian distributed data
        data = np.random.normal(mean, std_dev, num_samples)

        # Compute histogram counts using predefined bins
        counts, _ = np.histogram(data, bins=bin_edges)

        # Normalize and scale counts based on 'y'
        normalized_counts = 0.2 * np.sum(y) * counts / np.sum(counts)

        # Prepare stacked data
        stacked_y = y + normalized_counts
        
        stacked_y = np.sum(y) * stacked_y / np.sum(stacked_y)
        # Maybe we should normalize the data to one?
        #cleaned_arrays = cleaned_arrays/np.sum(cleaned_arrays, axis =1 , keepdims = True)
        
        anomaly_list.append(normalized_counts)
        anomaly_list_stacked.append(stacked_y)

    # Making it into a array
    anomaly_list         = np.array(anomaly_list)
    anomaly_list_stacked = np.array(anomaly_list_stacked)

    n_plots = 6
    # Plotting
    for plot in range(n_plots):
    
        plt.figure(figsize=(10, 8))

        # Plot the original 'y' data
        plt.step(bin_edges[:-1], test_set[plot], where='post', label='Original Y', linewidth=1.5, linestyle='--', color='blue')

        # Plot the Gaussian histogram as a separate line
        plt.step(bin_edges[:-1], anomaly_list[plot], where='post', label='Gaussian Histogram', linewidth=1.5, linestyle=':', color='green')

        # Plot the stacked data
        plt.step(bin_edges[:-1], anomaly_list_stacked[plot], where='post', label='Stacked Data', linewidth=2, color='red')

        # Adding plot decorations
        plt.xlabel('Bin Edges')
        plt.ylabel('Counts')
        plt.title('Comparison of Original, Gaussian, and Stacked Histograms')
        plt.legend()

        # Save the plot to a file
        plt.savefig('plots/anomalies/comparison_histogram_' +str(plot)+'.png')
    
    # Lets return the anomalies
    return anomaly_list_stacked

def validate_nominal_and_anomalies(test_set, anomaly_list_stacked):
    
    n_plots = 6
    
    # Define the range of your data
    min_edge = 0
    max_edge = 1

    # Generate bin edges from the specified range
    num_bins = len(test_set[1])
    bin_edges = np.linspace(min_edge, max_edge, num_bins + 1)
    
    for plot in range(n_plots):
    
        plt.figure(figsize=(10, 8))

        # Plot the Gaussian histogram as a separate line
        #plt.step(bin_edges[:-1], anomaly_list[plot], where='post', label='Gaussian Histogram', linewidth=1.5, linestyle=':', color='green')

        # Plot the stacked data
        plt.step(bin_edges[:-1], anomaly_list_stacked[plot], where='post', label='anomaly', linewidth=2, color='red')

        # Plot the original 'y' data
        plt.step(bin_edges[:-1], test_set[plot], where='post', label='Original', linewidth=1.5, linestyle='--', color='blue')

        # Adding plot decorations
        plt.xlabel('Bin Edges')
        plt.ylabel('Counts')
        plt.title('Comparison of Original, Gaussian, and Stacked Histograms')
        plt.legend()

        # Save the plot to a file
        plt.savefig('plots/anomalies/comparison_' +str(plot)+'.png')
    
    return 0

def plot_sse_scores(test_set, test_data_reconstructed, anomalous_data ,anomalous_data_reconstructed):
    return 0

