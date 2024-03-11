'''
Macro to use post-scripts/assess.py to convert the lsit of SSE scores to ROC curves for studies over a large data set
Requires input directory where bad_runs_sse_scores.csv is located (this is also the output directory) and list of bad
runs as labelled by data certification reports or similar (i.e. not algos!) (Runs not in the list of bad runs are 
considered good runs by default)
'''

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import sys
import json
import argparse
import awkward

from autodqm_ml.utils import expand_path
from autodqm_ml.constants import kANOMALOUS, kGOOD
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--input_file",
    help = "input file (i.e. output from fetch_data.py) to use for training the ML algorithm",
    type = str,
    required = True,
    default = None
  )
  return parser.parse_args()

def count_number_of_hists_above_threshold(Fdf, Fthreshold_list):
  runs_list = Fdf['run_number']
  Ft_list = np.array([float(Fthreshold_list_item) for Fthreshold_list_item in Fthreshold_list])
  hist_bad_count = 0
  bad_hist_array = []
  for run in runs_list:
    run_row = Fdf.loc[Fdf['run_number'] == run].drop(columns=['run_number'])
    run_row = run_row.iloc[0].values
    hist_bad_count = sum(hist_sse > hist_thresh for hist_sse, hist_thresh in zip(run_row, Ft_list))
    bad_hist_array.append(hist_bad_count)
  return bad_hist_array

# returns mean number of runs with SSE above the given threshold
def count_mean_runs_above(Fdf, Fthreshold_list, type, single):
  hists_flagged_per_run = count_number_of_hists_above_threshold(Fdf, Fthreshold_list)
  if (sum(hists_flagged_per_run) > 1.5 * len(Fdf['run_number'])) & (type == "good") & (single == 1):
    print("[COUNT_MEAN_RUNS_ABOVE] Tracing the SSE score thresholds for each histogram at the HF data point")
    for entry in Fthreshold_list: print(str(entry) + ",")
    single = 0
  #  print("[COUNT_MEAN_RUNS_ABOVE] Number of histogram flags for each run")
  #  print(hists_flagged_per_run)
  #  single = 0
  mean_hists_flagged_per_run = sum(hists_flagged_per_run) / len(Fdf['run_number'])
  #print(mean_hists_flagged_per_run)
  return mean_hists_flagged_per_run, single

# returns fraction of runs with SSE above the given threshold
def count_fraction_runs_above(Fdf, Fthreshold_list, N_bad_hists, type, single):
  hists_flagged_per_run = count_number_of_hists_above_threshold(Fdf, Fthreshold_list)
  count = len([i for i in hists_flagged_per_run if i > N_bad_hists])
  if (N_bad_hists == 3) & (count > 0.1 * len(Fdf['run_number'])) & (type == "good") & (single == 1):
    print("[COUNT_FRACTIONS_RUNS_ABOVE] Tracing the SSE score thresholds for each histogram at the RF data point")
    for entry in Fthreshold_list: print(str(entry) + ",")
    #print("[COUNT_FRACTIONS_RUNS_ABOVE] Followed by number of histogram flags for each good run")
    #print(hists_flagged_per_run)
    single = 0
  count_per_run = count / len(Fdf['run_number'])
  return count_per_run, single

def find_closest_index(arr, dpthres):
  closest_value = min(arr, key=lambda x: abs(x - dpthres))
  closest_index = arr.index(closest_value)
  return closest_index

def count_run_most_flags(df, score_thresholds):
  #print(df)
  #print(score_thresholds)
  result_df = pd.DataFrame(columns=['run_number', 'count_exceeds'])
  for index, row in df.iterrows():
    run = row['run_number']
    thresholds_iterator = iter(score_thresholds)
    threshold = next(thresholds_iterator, None)

    count_exceeds = 0

    for histogram_column in df.columns[1:]:
        score = row[histogram_column]

        if score > threshold:
            count_exceeds += 1

        threshold = next(thresholds_iterator, None)
    result_df = result_df.append({'run_number': run, 'count_exceeds': count_exceeds}, ignore_index=True)
  result_df['run_number'] = result_df['run_number'].astype(int).astype(str) + ','
  result_df = result_df.sort_values(by='count_exceeds', ascending=False)
  return result_df

def main(args):

  sse_df = pd.read_csv(args.input_file)
  algorithm_name = str(sse_df['algo'].iloc[0]).upper()
  if algorithm_name == "BETAB": algorithm_name = "Beta_Binomial"

  sse_df = sse_df.loc[:,~sse_df.columns.duplicated()].copy()
  hist_cols = [col for col in sse_df.columns if '_score_' in col]
  hist_dict = {each_hist: "max" for each_hist in hist_cols}

  sse_df = sse_df.groupby(['run_number','label'])[hist_cols].agg(hist_dict).reset_index()
  sse_df = sse_df.sort_values(['label']).reset_index()
  sse_df = sse_df[['run_number','label'] + [col for col in sse_df.columns if (col != 'run_number')&(col != 'label')]]

  sse_df_good = sse_df.loc[sse_df['label'] == 0].reset_index()
  sse_df_bad = sse_df.loc[sse_df['label'] == 1].reset_index()
  sse_df_good = sse_df_good[['run_number'] + hist_cols]
  sse_df_bad = sse_df_bad[['run_number'] + hist_cols]

  print("[LOGGER] Calculating SSE thresholds for each histogram for algorithm :" + algorithm_name)

  cutoffs_across_hists = []
  for histogram in hist_cols:
    sse_ordered = sorted(sse_df_good[histogram], reverse=True)
    cutoff_0 = sse_ordered[0] + 0.5*(sse_ordered[0] - sse_ordered[1])
    cutoff_thresholds = []
    cutoff_thresholds.append(cutoff_0)
    for ii in range(len(sse_ordered)-1):
      cutoff_ii = 0.5*(sse_ordered[ii]+sse_ordered[ii+1])
      cutoff_thresholds.append(cutoff_ii)
    cutoffs_across_hists.append(cutoff_thresholds)

  cutoffs_across_hists = np.array(cutoffs_across_hists)
  print("[LOGGER] There are " + str(len(cutoffs_across_hists)) + " histograms, " + str(len(sse_df_good['run_number'])) + " good runs and " + str(len(sse_df_bad['run_number']))  + " bad runs")

  print("[LOGGER] Counting the number of histogram flags at FR threshold > 0.1, N = 3")

  nbh_ii = 3
  tFRF_ROC_good_X = []
  tFRF_ROC_bad_Y = []
  single = 1
  for cutoff_index in range(len(cutoffs_across_hists[0,:])):
    #if nbh_ii == 3: print("HERE:" + str(cutoff_index))
    t_cutoff_index_g_FRF_rc, single = count_fraction_runs_above(sse_df_good, cutoffs_across_hists[:,cutoff_index], nbh_ii, "good", single)
    t_cutoff_index_b_FRF_rc, single = count_fraction_runs_above(sse_df_bad, cutoffs_across_hists[:,cutoff_index], nbh_ii, "bad", single)
    tFRF_ROC_good_X.append(t_cutoff_index_g_FRF_rc)
    tFRF_ROC_bad_Y.append(t_cutoff_index_b_FRF_rc)
    if single == 0: break

  tFRF_ROC_good_X = sorted(tFRF_ROC_good_X)
  tFRF_ROC_bad_Y = sorted(tFRF_ROC_bad_Y)

  tMHF_ROC_good_X = []
  tMHF_ROC_bad_Y = []

  single = 1
  for cutoff_index in range(len(cutoffs_across_hists[0,:])):
    t_cutoff_index_g_MHF_rc, single = count_mean_runs_above(sse_df_good, cutoffs_across_hists[:,cutoff_index], "good", single)
    t_cutoff_index_b_MHF_rc, single = count_mean_runs_above(sse_df_bad, cutoffs_across_hists[:,cutoff_index], "bad", single)
    tMHF_ROC_good_X.append(t_cutoff_index_g_MHF_rc)
    tMHF_ROC_bad_Y.append(t_cutoff_index_b_MHF_rc)
    if single == 0: break

  #print(tMHF_ROC_good_X)
  tMHF_ROC_good_X = sorted(tMHF_ROC_good_X)
  tMHF_ROC_bad_Y = sorted(tMHF_ROC_bad_Y)


  #algo = "pca"
  algo = "pca"
  if algo == "pca":
    thresholds_mh = [1.33880645e-06,1.71567021e-05,2.07432838e-04,2.66117342e-04,1.53229424e-04,1.55259134e-04,7.96096127e-05,2.56020621e-04,7.21273529e-04,7.51130839e-04,8.94609148e-05,1.21253016e-03,1.82645994e-04,4.68996706e-03,3.04298805e-05,4.29786637e-04,1.97987234e-04,1.22143826e-03,2.29235532e-04,2.26782300e-04,1.70092400e-04,1.90973911e-04,1.95007126e-04,2.04282311e-04,2.13468611e-04,1.69465104e-04,3.49285997e-04,1.69873415e-04,3.38607530e-04,6.96611175e-04,9.32676717e-04,1.10196514e-03,6.60170297e-04,6.76419962e-04,1.22296337e-03,1.10896094e-03,2.34616131e-04,4.75172914e-05,5.61562970e-02,5.29765262e-02,6.42649239e-03,8.03505818e-03,3.62255976e-03,4.34978594e-03,1.19777153e-02,1.40878144e-02,3.25955373e-02,3.77295784e-02,4.55771287e-02,5.45613752e-02,5.30430759e-03,3.64479532e-03,5.89764503e-03,4.54672408e-03,1.28201471e-02,1.00821667e-02,1.42514024e-02,1.16427332e-02,1.82288481e-02,1.58886293e-02,1.91253485e-02,1.83733684e-02]
    thresholds_fr = [1.23321025e-06,1.49943958e-05,1.49286891e-04,1.88155069e-04,1.37752523e-04,1.41880690e-04,6.42989790e-05,2.28385199e-04,6.13848811e-04,6.77197251e-04,7.61906472e-05,1.12812445e-03,1.48219262e-04,3.17847931e-03,2.54402776e-05,3.28967227e-04,1.73371775e-04,1.02633538e-03,1.78704300e-04,1.84717121e-04,1.47557857e-04,1.67662806e-04,1.49062244e-04,1.49726581e-04,1.67555471e-04,1.33698367e-04,2.76963667e-04,1.42156407e-04,2.53627973e-04,5.84408675e-04,8.82823846e-04,9.84422267e-04,3.79128118e-04,5.85359283e-04,1.04462715e-03,9.89620541e-04,2.16569021e-04,3.92015482e-05,4.10701618e-02,4.30546707e-02,5.89479274e-03,6.98787478e-03,3.11337826e-03,3.60097170e-03,9.78355221e-03,1.23230026e-02,2.46475766e-02,3.25677169e-02,3.77774200e-02,5.13555984e-02,4.31542490e-03,3.14183580e-03,4.97391103e-03,3.35085450e-03,9.48770967e-03,8.02717054e-03,1.15562015e-02,1.01348733e-02,1.45747964e-02,1.39596159e-02,1.68925103e-02,1.63308314e-02]
  
  if algo == "ae":
    thresholds_mh = [3.43235308e-04,6.62314856e-05,9.85145902e-03,1.05648505e-03,1.58853967e-03,1.89324233e-04,1.09722362e-04,3.50171594e-03,7.75700216e-04,4.23791384e-03,4.12679020e-04,2.05688427e-02,3.04456103e-04,2.59582113e-02,1.95408631e-04,2.72414456e-02,2.73722362e-04,6.89270544e-03,2.24029017e-03,3.90413689e-03,2.82081729e-01,8.77772082e-04,2.32376275e-04,1.49818297e-03,2.72066078e-04,2.37863248e-02,4.28827785e-04,2.85802584e-02,3.82918177e-04,4.15537740e-03,1.35855561e-03,1.25847083e-03,7.39616055e-03,1.64155478e-03,1.26473719e-03,1.16164908e-03,1.45719745e-03,1.44145096e-04,7.71774847e-02,6.28571785e-02,8.15585082e-03,9.73947628e-03,3.95112632e-03,4.74834296e-03,1.39885435e-02,1.56613237e-02,3.52744863e-02,3.94445789e-02,5.36912058e-02,6.61840856e-02,6.15910197e-03,4.40185316e-03,6.54050092e-03,4.93548246e-03,1.49373987e-02,1.17950470e-02,1.65506627e-02,1.36560220e-02,2.27486150e-02,1.86681047e-02,2.20956146e-02,2.03984596e-02]
    thresholds_fr = [3.43235308e-04,6.62314856e-05,9.85145902e-03,1.05648505e-03,1.58853967e-03,1.89324233e-04,1.09722362e-04,3.50171594e-03,7.75700216e-04,4.23791384e-03,4.12679020e-04,2.05688427e-02,3.04456103e-04,2.59582113e-02,1.95408631e-04,2.72414456e-02,2.73722362e-04,6.89270544e-03,2.24029017e-03,3.90413689e-03,2.82081729e-01,8.77772082e-04,2.32376275e-04,1.49818297e-03,2.72066078e-04,2.37863248e-02,4.28827785e-04,2.85802584e-02,3.82918177e-04,4.15537740e-03,1.35855561e-03,1.25847083e-03,7.39616055e-03,1.64155478e-03,1.26473719e-03,1.16164908e-03,1.45719745e-03,1.44145096e-04,7.71774847e-02,6.28571785e-02,8.15585082e-03,9.73947628e-03,3.95112632e-03,4.74834296e-03,1.39885435e-02,1.56613237e-02,3.52744863e-02,3.94445789e-02,5.36912058e-02,6.61840856e-02,6.15910197e-03,4.40185316e-03,6.54050092e-03,4.93548246e-03,1.49373987e-02,1.17950470e-02,1.65506627e-02,1.36560220e-02,2.27486150e-02,1.86681047e-02,2.20956146e-02,2.03984596e-02]

  metric = "fr"
  #metric = "mh"
  if metric == "fr":
    thresholds_to_study = thresholds_fr
  if metric == "mh":
    thresholds_to_study = thresholds_mh

  print("[LOGGER] Now calculating the number of histogram flags per good and bad run, with algorithm " + algo.upper() + " and metric " + metric.upper() + " options selected")

  run_flags_good = count_run_most_flags(sse_df_good,thresholds_to_study)
  run_flags_bad = count_run_most_flags(sse_df_bad,thresholds_to_study)

  print("Good")
  print(run_flags_good.to_string(index=False))

  print("Bad")
  print(run_flags_bad.to_string(index=False))

  sse_df_good_runs_only = sse_df_good.loc[(sse_df_good['run_number'] == 355913) | (sse_df_good['run_number'] == 356386) | (sse_df_good['run_number'] == 356956)]

  result_dict = {}

  for index, row in sse_df_good_runs_only.iterrows():
    run = row['run_number']
    run_data = row.drop('run_number')

    # Find histograms above the corresponding scores
    above_threshold = run_data.index[run_data > thresholds_to_study].tolist()

    # Store the result in the dictionary
    result_dict[run] = above_threshold

  result_df = pd.DataFrame(list(result_dict.items()), columns=['Run', 'Histograms'])

  # prints histograms flagged in a particular run, following selection of runs above (good runs with high numbers of flags)
  for index, row in result_df.iterrows():
    run = row['Run']
    histograms = row['Histograms']
    
    print(f"Run {run}:")
    for hist in histograms:
        print(hist[17:(-14 - len(algo))])
    print()
  

  # index of point in array closest to MH = 1.5 from good runs
  # array of sse scores all hists and runs
  # extract sse scores per hist, order, and calculate thresholds
  # for each threshold, find number of sse scores above this for each hist and matching hist threshold per 265 runs, for each of good and bad run df
  # sum across runs and divide by number of runs

  result_data_mh = {'Histogram': [], 'MH': []}
  result_data_fr = {'Histogram': [], 'FR': []}
  result_data_mh_bad = {'Histogram': [], 'MH': []}
  result_data_fr_bad = {'Histogram': [], 'FR': []}

  for column, threshold in zip(sse_df.columns[1:], thresholds_mh):
    graph_name = column
    threshold_count = (sse_df[column] > threshold).sum()
    result_data_mh['Histogram'].append(graph_name)
    result_data_mh['MH'].append(threshold_count)

  result_df_mh = pd.DataFrame(result_data_mh)

  for column, threshold in zip(sse_df.columns[1:], thresholds_fr):
    graph_name = column
    threshold_count = (sse_df[column] > threshold).sum()
    result_data_fr['Histogram'].append(graph_name)
    result_data_fr['FR'].append(threshold_count)

  result_df_fr = pd.DataFrame(result_data_fr)
  result_df = pd.merge(result_df_mh, result_df_fr, on='Histogram')

  result_df.to_csv('./hist_flag_freq_all_runs_'+algo+'.csv', index=False)


  for column, threshold in zip(sse_df_bad.columns[1:], thresholds_mh):
    graph_name = column
    threshold_count = (sse_df_bad[column] > threshold).sum()
    result_data_mh_bad['Histogram'].append(graph_name)
    result_data_mh_bad['MH'].append(threshold_count)

  result_df_mh_bad = pd.DataFrame(result_data_mh_bad)
  #print(result_df_mh_bad)

  for column, threshold in zip(sse_df_bad.columns[1:], thresholds_fr):
    graph_name = column
    threshold_count = (sse_df_bad[column] > threshold).sum()
    result_data_fr_bad['Histogram'].append(graph_name)
    result_data_fr_bad['FR'].append(threshold_count)

  result_df_fr_bad = pd.DataFrame(result_data_fr_bad)
  result_df_bad = pd.merge(result_df_mh_bad, result_df_fr_bad, on='Histogram')


  print("[LOGGER] Now calculating the frequency at which histograms are flagged in bad run, as flagged by the " + algo.upper() + " algorithm")
  print(result_df_bad[["MH","FR"]].to_string(index=False, sep=','))
  result_df_bad.to_csv('./hist_flag_freq_bad_runs_'+algo+'.csv', index=False)

  index_mh1p5 = find_closest_index(tMHF_ROC_good_X, 1.5)
  dist_of_sse_at_mh1p5 = np.array([sub_array[index_mh1p5] for sub_array in cutoffs_across_hists])
  log_scale_mh1p5 = np.log10(dist_of_sse_at_mh1p5)

  index_rf0p1 = find_closest_index(tFRF_ROC_good_X, 0.1)
  dist_of_sse_at_rf0p1 = np.array([sub_array[index_rf0p1] for sub_array in cutoffs_across_hists])
  log_scale_rf0p1 = np.log10(dist_of_sse_at_rf0p1)

  bin_width = 0.1
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
  ax1.hist(log_scale_mh1p5, bins=np.arange(-6, 0, step=bin_width), alpha=0.5, edgecolor='red')
  ax1.set_title(r'SSE distribution for MH$_{\mathrm{good}}$ = 1.5')
  ax1.set_xlabel('SSE score')
  ax1.set_ylabel('Frequency')

  ax2.hist(log_scale_rf0p1, bins=np.arange(-6, 0, step=bin_width), alpha=0.5, edgecolor='green')
  ax2.set_title(r'SSE distribution for RF$_{\mathrm{good}}$ = 0.1')
  ax2.set_xlabel('SSE score')
  ax2.set_ylabel('Frequency')

  plt.tight_layout()

  plt.show()

if __name__ == "__main__":
  args = parse_arguments()
  main(args)
