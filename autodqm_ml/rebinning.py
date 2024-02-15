from numpy.testing import assert_allclose
import numpy as np

def merge_below_threshold_with_indices(arr, threshold):
    merged_indices = [] # this pertains to the original array element positions, not the iterated version following each merge
    i = 0
    while i < len(arr) - 1:
        if arr[i] < threshold:
            merged_value = arr[i] + arr[i + 1]
            arr[i + 1] = merged_value
            merged_indices.append((i, i + 1))
            arr = np.delete(arr, i)
        else:
            i += 1
    
    return arr, merged_indices

def merge_bins_in_arrays(array_of_arrays, pairs_of_indices):
    new_array_of_arrays = []
    for arr in array_of_arrays:
        # Make a copy of the original array to perform merges
        merged_array = np.copy(arr)
    
        # Iterate through each pair of indices
        for pair in pairs_of_indices:
            # Merge the elements at the given indices
            merged_value = merged_array[pair[0]] + merged_array[pair[1]]
            # Update the value at the first index and remove the value at the second index
            merged_array[pair[0]] = merged_value
            merged_array = np.delete(merged_array, pair[1])
        
            # Update the pairs of indices based on the merge operation
            for i in range(len(pairs_of_indices)):
                pair_indices = list(pairs_of_indices[i])  # Convert tuple to list
                # Adjust the indices based on the merge operation
                for j in range(len(pair_indices)):
                    if pair_indices[j] == pair[1]:
                        pair_indices[j] = pair[0]  # Update the index
                # Convert the list back to tuple
                pairs_of_indices[i] = tuple(pair_indices)
                
        merged_array = np.array([np.array(sub_arr) for sub_arr in merged_array])
        new_array_of_arrays.append(np.array(merged_array))

    new_array_of_arrays = np.array(new_array_of_arrays)
    return new_array_of_arrays

def rebinning_min_occupancy(df_ver_hist, min_occ_threshold):
    hist = np.array(df_ver_hist)
    normalized_hist = hist / np.sum(hist, axis=(1, 2), keepdims=True)
    #print(normalized_hist.shape) # normalise histogram shape
    flattened_hist = normalized_hist.reshape(len(hist), -1)
    #print("Flattened Shape:", flattened_hist.shape)
    combined_hist = np.sum(flattened_hist, axis=0)
    #print("Combined Shape:", combined_hist.shape) # 648 bins per hist, and totals number of hists (308)
    hist_copy = hist.reshape(len(hist), -1)
    #print(len(hist[0])*len(hist[0][0]))
    assert_allclose(sum(combined_hist), len(hist), atol=1e-3)

    track_merges = []
    print("-----")
    #print("len combined hist: ", len(combined_hist))
    new_comb_hist, track_merges = merge_below_threshold_with_indices(combined_hist, min_occ_threshold * len(hist[0])*len(hist[0][0]))

    #print(len(new_comb_hist))
    #print('track_merges:', track_merges)

    #print("len flattened hist: ", len(flattened_hist))
    #print(track_merges)

    merged_flattened_hist = merge_bins_in_arrays(flattened_hist, track_merges)
    #print(merged_flattened_hist)

    return merged_flattened_hist
