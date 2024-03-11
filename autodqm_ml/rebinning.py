from numpy.testing import assert_allclose
import numpy as np

def merge_below_threshold_with_indices(arr, threshold):
    merged_indices = [] # this pertains to the original array element positions, not the iterated version following each merge
    i = 0
    removed_val = 0
    while i < len(arr) - 1:
        if arr[i] < threshold:
            merged_value = arr[i] + arr[i + 1]
            arr[i + 1] = merged_value
            merged_indices.append((i + removed_val, i + removed_val + 1))
            arr = np.delete(arr, i)
            removed_val += 1
        else:
            i += 1
    return arr, merged_indices

def count_instances_before_first_pair_changes(arr):
    if not arr:
        return 0, []

    if (arr[0] != (0,1)):
        return 0, arr
    
    count = 0
    for i in range(len(arr) - 1):
        if arr[i][1] == arr[i + 1][0]:
            count += 1
        else:
            break

    if count >= 1: count += 1
    new_array = arr[count:]

    return count, new_array

def count_instances_before_last_pair_changes(arr,hist_len):
    if not arr:
        return 0, []

    if (arr[-1] != (hist_len - 2,hist_len - 1)):
        return 0, arr

    count = 0
    for i in range(len(arr) - 1, -1, -1):
        if arr[i][0] == arr[i - 1][1]:
            count += 1
        else:
            break

    if count >= 1: count += 1
    new_array = arr[:-count]

    return count, new_array

def merge_first_n_bins(array_of_arrays, n):
    if n <= 0:
        return array_of_arrays
    merged_bins = [np.concatenate(([sum(subarray[:n])], subarray[n:])) for subarray in array_of_arrays]
    return merged_bins

def merge_last_n_bins(array_of_arrays, n):
    if n <= 0:
        return array_of_arrays
    merged_bins = [np.concatenate((subarray[:-n], [sum(subarray[-n:])])) for subarray in array_of_arrays]
    return merged_bins

def merge_bins_in_arrays(array_of_arrays, pairs_of_indices, empty_bins_removed):
    new_array_of_arrays = []
    for arr in array_of_arrays:
        # Make a copy of the original array to perform merges
        merged_array = np.copy(arr)
        #print(len(merged_array))
    
        # Iterate through each pair of indices
        for pair in pairs_of_indices:
            pair = (pair[0] - empty_bins_removed, pair[1] - empty_bins_removed)
            # Merge the elements at the given indices
            merged_value = merged_array[pair[0]] + merged_array[pair[1]]
            # Update the value at the first index and remove the value at the second index
            merged_array[pair[0]] = -5
            merged_array[pair[1]] = merged_value
            #merged_array = np.compress(merged_array != 0, merged_array)
            #merged_array = np.delete(merged_array, pair[1])
            '''
            # Update the pairs of indices based on the merge operation
            for i in range(len(pairs_of_indices)):
                pair_indices = list(pairs_of_indices[i])  # Convert tuple to list
                # Adjust the indices based on the merge operation
                for j in range(len(pair_indices)):
                    if pair_indices[j] == pair[1]:
                        pair_indices[j] = pair[0]  # Update the index
                # Convert the list back to tuple
                pairs_of_indices[i] = tuple(pair_indices)
            '''
        merged_array = np.compress(merged_array != -5, merged_array)
        #merged_array = np.array([np.array(sub_arr) for sub_arr in merged_array])
        new_array_of_arrays.append(np.array(merged_array))
    new_array_of_arrays = np.array(new_array_of_arrays)
    return new_array_of_arrays

def rebinning_min_occupancy(df_ver_hist, min_occ_threshold):
    hist = np.array(df_ver_hist)
    number_of_first_empty = 0
    normalized_hist = hist / np.sum(hist, axis=(1, 2), keepdims=True)
    #print("Normalised hist shape:", normalized_hist.shape) # normalise histogram shape
    flattened_hist = normalized_hist.reshape(len(hist), -1)
    print("Flattened Shape:", flattened_hist.shape)
    combined_hist = np.sum(flattened_hist, axis=0)
    #print("Combined Shape:", combined_hist.shape) # XXX bins per hist, and totals number of hists (308)
    hist_copy = hist.reshape(len(hist), -1)
    #assert_allclose(sum(combined_hist), len(hist), atol=1e-3)
    track_merges = []
    new_comb_hist, track_merges = merge_below_threshold_with_indices(combined_hist, min_occ_threshold * len(hist)) # / (0.2*np.log10(len(flattened_hist[0]))))
    #print("STARTING FULL MANIPULATION PROGRAMMING")
    #print("ORIGINAL HISTOGRAM DIMENSIONS MULITPLIED: ",len(hist[0])*len(hist[0][0]))
    #print("LENGTH OF MODIFIED HISTOGRAM THAT SURVIVES THRESHOLD SELECTION: ", len(new_comb_hist))
    #print('NUMBER OF INTENDED TRACK MERGES: ', len(track_merges))
    number_of_first_empty, track_merges = count_instances_before_first_pair_changes(track_merges)
    #print("NUMBER OF FIRST BINS BELOW OCC THRESHOLD: ",number_of_first_empty)
    #print("NUMBER OF MERGES REMAINING: ",len(track_merges))
    number_of_last_empty, track_merges = count_instances_before_last_pair_changes(track_merges,len(combined_hist))
    #print("NUMBER OF LAST BINS BELOW OCC THRESHOLD: ",number_of_last_empty)
    #print("NUMBER OF MERGES REMAINING: ",len(track_merges))
    flattened_hist = merge_first_n_bins(flattened_hist,number_of_first_empty)
    flattened_hist = merge_last_n_bins(flattened_hist,number_of_last_empty)
    #print("BEFORE FULL MERGE, HISTOGRAMS ARE OF LENGTH ",len(flattened_hist[0]))
    merged_flattened_hist = merge_bins_in_arrays(flattened_hist, track_merges, number_of_first_empty)
    #print("AFTER FULL MERGE, HISTOGRAMS ARE OF LENGTH ",len(merged_flattened_hist[0]))

    return merged_flattened_hist
