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
    print("MH " + type)
    print(Fthreshold_list)
    print(hists_flagged_per_run)
    single = 0
  mean_hists_flagged_per_run = sum(hists_flagged_per_run) / len(Fdf['run_number'])
  print(mean_hists_flagged_per_run)
  return mean_hists_flagged_per_run, single

# returns fraction of runs with SSE above the given threshold
def count_fraction_runs_above(Fdf, Fthreshold_list, N_bad_hists, type, single):
  hists_flagged_per_run = count_number_of_hists_above_threshold(Fdf, Fthreshold_list)
  count = len([i for i in hists_flagged_per_run if i > N_bad_hists])
  if (N_bad_hists == 3) & (count > 0.1 * len(Fdf['run_number'])) & (type == "good") & (single == 1):
    print("FR " + type)
    print(Fthreshold_list)
    print(hists_flagged_per_run)
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

  #N_bad_hists = [5,3,1]
  N_bad_hists = [3]
  tFRF_ROC_good_X = []
  tFRF_ROC_bad_Y = []

  for nbh_ii in N_bad_hists:
    tFRF_ROC_good_X_init = []
    tFRF_ROC_bad_Y_init = []
    single = 1
    for cutoff_index in range(len(cutoffs_across_hists[0,:])):
      #if nbh_ii == 3: print("HERE:" + str(cutoff_index))
      t_cutoff_index_g_FRF_rc, single = count_fraction_runs_above(sse_df_good, cutoffs_across_hists[:,cutoff_index], nbh_ii, "good", single)
      t_cutoff_index_b_FRF_rc, single = count_fraction_runs_above(sse_df_bad, cutoffs_across_hists[:,cutoff_index], nbh_ii, "bad", single)
      tFRF_ROC_good_X_init.append(t_cutoff_index_g_FRF_rc)
      tFRF_ROC_bad_Y_init.append(t_cutoff_index_b_FRF_rc)

    tFRF_ROC_good_X_init = sorted(tFRF_ROC_good_X_init)
    tFRF_ROC_bad_Y_init = sorted(tFRF_ROC_bad_Y_init)

    tFRF_ROC_good_X.append(tFRF_ROC_good_X_init)
    tFRF_ROC_bad_Y.append(tFRF_ROC_bad_Y_init)
    

  tMHF_ROC_good_X = []
  tMHF_ROC_bad_Y = []

  single = 1
  for cutoff_index in range(len(cutoffs_across_hists[0,:])):
    t_cutoff_index_g_MHF_rc, single = count_mean_runs_above(sse_df_good, cutoffs_across_hists[:,cutoff_index], "good", single)
    t_cutoff_index_b_MHF_rc, single = count_mean_runs_above(sse_df_bad, cutoffs_across_hists[:,cutoff_index], "bad", single)
    tMHF_ROC_good_X.append(t_cutoff_index_g_MHF_rc)
    tMHF_ROC_bad_Y.append(t_cutoff_index_b_MHF_rc)

  print("AFTER THE FUNCTIONS")
  #print(tMHF_ROC_good_X)

  tMHF_ROC_good_X = sorted(tMHF_ROC_good_X)
  tMHF_ROC_bad_Y = sorted(tMHF_ROC_bad_Y)


  #algo = "pca"
  algo = "ae"
  if algo == "pca":
    thresholds_mh = [1.23321025e-06,1.50930279e-05,1.49286891e-04,1.88155069e-04,1.37752523e-04,1.47867315e-04,7.83042127e-05,2.28385199e-04,6.30925470e-04,6.77197251e-04,7.76971854e-05,1.12812445e-03,1.67914317e-04,3.17847931e-03,4.77704011e-05,3.28967227e-04,1.78754919e-04,1.02633538e-03,1.78704300e-04,1.84717121e-04,1.47557857e-04,1.67662806e-04,1.49062244e-04,1.49726581e-04,1.67555471e-04,1.33698367e-04,2.76963667e-04,1.42156407e-04,2.53627973e-04,5.84408675e-04,8.82823846e-04,9.84422267e-04,3.79128118e-04,5.85359283e-04,1.04462715e-03,9.89620541e-04,2.16569021e-04,3.92015482e-05,4.10701618e-02,4.30546707e-02,5.91853398e-03,6.71062478e-03,3.29411974e-03,3.51575081e-03,9.57488908e-03,1.26203323e-02,2.59618679e-02,3.55206677e-02,3.81228905e-02,5.28686726e-02,4.57241002e-03,3.22627175e-03,5.30213895e-03,3.74273862e-03,1.00866140e-02,8.54201608e-03,1.21205140e-02,1.08614010e-02,1.56596262e-02,1.42228641e-02,1.89596373e-02,1.74686020e-02]
    thresholds_fr = [1.33880645e-06,1.71005517e-05,2.07432838e-04,2.66117342e-04,1.53229424e-04,1.74173277e-04,8.90376781e-05,2.56020621e-04,7.37766365e-04,7.51130839e-04,9.92637028e-05,1.21253016e-03,2.07025065e-04,4.68996706e-03,5.05289716e-05,4.29786637e-04,2.03680806e-04,1.22143826e-03,2.29235532e-04,2.26782300e-04,1.70092400e-04,1.90973911e-04,1.95007126e-04,2.04282311e-04,2.13468611e-04,1.69465104e-04,3.49285997e-04,1.69873415e-04,3.38607530e-04,6.96611175e-04,9.32676717e-04,1.10196514e-03,6.60170297e-04,6.76419962e-04,1.22296337e-03,1.10896094e-03,2.34616131e-04,4.75172914e-05,5.61562970e-02,5.29765262e-02,7.74857577e-03,8.63094534e-03,3.67583637e-03,4.51919091e-03,1.21420367e-02,1.48391655e-02,3.45248856e-02,3.87003590e-02,4.67129799e-02,5.66864263e-02,5.73300184e-03,3.73491494e-03,6.19481455e-03,4.59445989e-03,1.37037914e-02,1.07576198e-02,1.49759833e-02,1.21937127e-02,1.91201691e-02,1.80337773e-02,2.05148831e-02,1.95208514e-02]
  
  if algo == "ae":
    thresholds_mh = [3.43235308e-04,6.32704961e-05,1.21560002e-02,1.10584158e-03,2.49256570e-04,1.72538927e+00,8.90918916e+06,4.04723681e-03,1.13164362e+07,7.90779055e-03,1.37364361e+06,2.10385729e-02,9.48842615e+02,2.51544865e-02,9.26887607e+05,2.67243432e-02,2.84718639e+06,7.35490642e-03,2.24029017e-03,3.90413689e-03,2.82017854e-02,3.12499911e-03,2.35614017e-04,1.37801106e-02,2.79267887e-04,2.82977162e-01,4.45697806e-04,1.98522037e-02,3.85285611e-04,5.93037919e-03,1.57751789e-03,1.24156223e-03,5.78747791e-03,1.63056218e-03,1.26515594e-03,1.22085912e-03,4.32205942e-04,1.20878434e-04,7.72103765e-02,6.23229780e-02,1.06981523e-01,1.65030603e-02,1.04873488e-01,2.41992558e-02,2.93121238e-02,5.88366336e-02,4.14468502e-02,7.06497670e-02,1.81868000e-01,5.17372589e-01,6.04952044e-03,4.33471317e-02,7.60607355e-02,9.05287890e-03,1.25771026e-01,1.23315050e-02,4.13688186e-01,2.21849840e+01,1.70747272e-01,2.02735651e-01,7.79511363e+01,8.07983347e+01]
    thresholds_fr = [3.45784982e-04,7.18441549e-05,1.32854789e-02,1.29294316e-03,2.66538588e-04,2.22242527e+00,9.47408080e+06,4.18054456e-03,1.27242410e+07,7.99147306e-03,1.73384307e+06,2.17228089e-02,1.13181223e+03,3.00884209e-02,9.43803204e+05,2.76265196e-02,3.34901870e+06,7.74733924e-03,2.28152040e-03,3.95771757e-03,3.21016176e-02,3.26686532e-03,2.48405074e-04,1.40528112e-02,2.97953845e-04,2.95177950e-01,4.89878585e-04,2.33485642e-02,4.68326525e-04,6.06619073e-03,1.81558892e-03,1.29455791e-03,6.22331421e-03,2.16641479e-03,1.36219487e-03,1.39317963e-03,4.40724532e-04,1.22319700e-04,8.41518050e-02,6.95396490e-02,1.08259410e-01,1.84118339e-02,1.19769289e-01,4.17237221e-02,3.71517175e-02,7.11543543e-02,5.15580223e-02,1.24368375e-01,2.24870679e-01,8.16979036e-01,8.53139196e-03,1.01338313e-01,8.81065211e-02,9.91997432e-03,2.36691047e-01,1.40691508e-02,4.81930945e-01,2.36482632e+01,3.53155503e-01,4.13411270e-01,1.03975066e+02,9.57220059e+01]

  #metric = "fr"
  metric = "mh"
  if metric == "fr":
    thresholds_to_study = thresholds_fr
  if metric == "mh":
    thresholds_to_study = thresholds_mh

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
        print(hist[17:-15])
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

  result_df_bad.to_csv('./hist_flag_freq_bad_runs_'+algo+'.csv', index=False)

  index_mh1p5 = find_closest_index(tMHF_ROC_good_X, 1.5)
  dist_of_sse_at_mh1p5 = np.array([sub_array[index_mh1p5] for sub_array in cutoffs_across_hists])
  log_scale_mh1p5 = np.log10(dist_of_sse_at_mh1p5)

  index_rf0p1 = find_closest_index(tFRF_ROC_good_X[1], 0.1)
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
