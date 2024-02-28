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
  #print(mean_hists_flagged_per_run)
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
    #print(run)
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
  N_bad_hists = 3
  tFRF_ROC_good_X = []
  tFRF_ROC_bad_Y = []

  for nbh_ii in N_bad_hists:
    tFRF_ROC_good_X_init = []
    tFRF_ROC_bad_Y_init = []
    single = 1
    for cutoff_index in range(len(cutoffs_across_hists[0,:])):
      #if nbh_ii == 3: print(cutoff_index)
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

  tMHF_ROC_good_X = sorted(tMHF_ROC_good_X)
  tMHF_ROC_bad_Y = sorted(tMHF_ROC_bad_Y)



  algo = "pca"
  #algo = "ae"
  if algo == "pca":
    thresholds_mh = [1.33880645e-06,3.70879538e-06,2.07432838e-04,2.66117342e-04,1.53229424e-04,1.50588933e-04,7.60579091e-05,2.56020621e-04,7.17893025e-04,7.51130839e-04,7.71975298e-05,1.21253016e-03,1.73886274e-04,4.68996706e-03,2.82594840e-05,4.29786637e-04,1.87090182e-04,1.22143826e-03,2.29235532e-04,2.26782300e-04,1.70092400e-04,1.90973911e-04,1.95007126e-04,2.04282311e-04,2.13468611e-04,1.69465104e-04,3.49285997e-04,1.69873415e-04,3.38607530e-04,6.96611175e-04,9.32676717e-04,1.10394778e-03,6.60170297e-04,6.76419962e-04,1.22298216e-03,1.10916584e-03,2.34616131e-04,4.75172915e-05,5.57723951e-02,5.29767061e-02,6.71458421e-03,8.19580717e-03,3.56364660e-03,4.33776876e-03,1.21058412e-02,1.42543040e-02,3.26262834e-02,3.79098123e-02,4.55855036e-02,5.45623678e-02,5.32348925e-03,3.60698042e-03,5.95885243e-03,4.54829651e-03,1.28365046e-02,1.00816028e-02,1.42640427e-02,1.16641336e-02,1.82043639e-02,1.58835050e-02,1.91096728e-02,1.83462891e-02]
    thresholds_fr = [1.21923653e-06,3.39235805e-06,1.46611637e-04,1.86651626e-04,1.34228888e-04,1.34414060e-04,5.52984813e-05,2.04392598e-04,5.91147559e-04,6.57175545e-04,6.21854772e-05,1.09874233e-03,1.38764334e-04,3.04778903e-03,2.05674547e-05,3.20099752e-04,1.59766659e-04,9.72542278e-04,1.76607148e-04,1.71897154e-04,1.39978724e-04,1.52245923e-04,1.37982468e-04,1.43233418e-04,1.50150669e-04,1.28892336e-04,2.56956056e-04,1.31040269e-04,2.43951748e-04,5.71627414e-04,8.59285632e-04,9.21611290e-04,3.72258677e-04,5.72983971e-04,9.82457296e-04,9.82599023e-04,1.98513105e-04,3.86858372e-05,3.84118727e-02,4.27476439e-02,5.76547988e-03,6.60202868e-03,3.03147035e-03,3.33583893e-03,9.41221220e-03,1.21282967e-02,2.44694405e-02,3.21343906e-02,3.70272178e-02,4.99990694e-02,4.28180460e-03,3.04120583e-03,4.82706018e-03,3.27151354e-03,9.19279936e-03,7.68303605e-03,1.11989272e-02,9.55544609e-03,1.39589441e-02,1.34185967e-02,1.65757293e-02,1.60363727e-02]
  
  if algo == "ae":
    thresholds_mh = [4.25565811e-06,2.72662318e-05,1.22452560e-02,1.12961949e-03,2.48105838e-04,1.59000401e-04,9.29551184e-05,3.04813516e-03,7.85855788e-04,6.64200152e-03,2.61354390e-04,1.97794150e-02,1.93488271e-04,2.59127835e-02,1.38232329e-04,2.60387157e-02,2.85075333e-04,7.07133440e-03,1.23303601e-03,1.50356244e-03,2.37041451e-02,1.56838456e-03,2.81271250e-04,1.44898730e-03,2.94147092e-04,2.50652520e-02,4.32353252e-04,2.07793869e-02,4.10492134e-04,1.52113746e-01,1.41945898e-03,1.24526710e-03,3.45554296e-02,2.24815997e-03,1.26829066e-03,1.15859112e-03,4.17991751e-04,1.64827064e-04,7.70464223e-02,6.26892118e-02,8.32875034e-03,9.60306614e-03,5.44727114e-03,6.29202680e-03,1.43760371e-02,1.57707936e-02,3.59110147e-02,3.93577079e-02,5.41346806e-02,6.64641657e-02,6.14407728e-03,3.95507947e-03,6.64433037e-03,5.00297150e-03,1.51480623e-02,1.17949134e-02,1.66293178e-02,1.35661483e-02,2.27381337e-02,1.91478194e-02,2.19890951e-02,2.03942992e-02]
    thresholds_fr = [4.07938840e-06,2.65272426e-05,1.13604429e-02,1.11095458e-03,2.44459096e-04,1.55373752e-04,8.56873278e-05,2.98882210e-03,7.40703994e-04,6.58689315e-03,2.54363257e-04,1.97337078e-02,1.73677626e-04,2.54428656e-02,1.37468819e-04,2.55057127e-02,2.80031608e-04,7.05595115e-03,1.22261531e-03,1.48614878e-03,2.25002668e-02,1.52170926e-03,2.70846493e-04,1.40277526e-03,2.82806131e-04,2.38418947e-02,4.10689956e-04,2.06457327e-02,3.85074294e-04,1.50723591e-01,1.33363721e-03,1.17306572e-03,3.43795579e-02,2.11243511e-03,1.24652611e-03,1.13112958e-03,4.11502194e-04,1.61705247e-04,7.22285041e-02,5.90187745e-02,7.91861903e-03,8.81976572e-03,5.34482952e-03,6.10680130e-03,1.28192236e-02,1.51737748e-02,3.44470912e-02,3.89189023e-02,4.71833989e-02,5.68616941e-02,5.73964175e-03,3.67919797e-03,6.58799543e-03,4.83185422e-03,1.46592416e-02,1.09744590e-02,1.55580370e-02,1.31766596e-02,2.08115834e-02,1.89783244e-02,2.13012894e-02,1.95514855e-02]

  metric = "fr"
  #metric = "mh"
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
        print(hist[17:-17])
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
