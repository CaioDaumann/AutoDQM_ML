#### THIS IS THE VERSION WITH THE FORMAT FOR THE PAPER #####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.set_printoptions(threshold=sys.maxsize)
## !!!TODO::change this to take arguments of csv files instead of hard code xlsx file
N = 3
algo =   'chi2' # 'pull' #
## plot the pull and chi2 using rob's plotting scripts
fig, axs = plt.subplots(ncols=2,nrows=1,figsize=(12,6))
fig0, ax0 = plt.subplots(figsize=(6,6))
fig1, ax1 = plt.subplots(figsize=(6,6))

for numref, marker, color in zip(['1_REF', '4_REF', '8_REF'], ['-D', '-o', '-^'], ["#e42536", "#f89c20", "#5790fc",]):#["#5790fc", "#f89c20","#7a21dd"]):#['purple', 'yellow', 'orange']):
    plotkwargs = {'label':numref,  'marker':'.'}
    # assumed all csv files are same name but in different directories
    df = pd.read_csv('HLT_l1tShift_{}/L1T_HLTPhysics.csv'.format(numref))

    ## split into good (0,-1) and bad runs (1)
    df_g = df[df['label'] != 1]
    df_b = df[df['label'] == 1]


    ## remove bad rows (-99), col = ['label', 'run_number']
    df_g = df_g.filter(regex=f'{algo}_score') #'score')
    df_g = df_g[df_g != -99].dropna(how='all')
    df_b = df_b.filter(regex=f'{algo}_score')  # 'score')
    df_b = df_b[df_b != -99].dropna(how='all')

    ## sort descending
    sorted_df_g = -np.sort(-df_g, axis=0)

    ## calculate thresholds
    cuts = np.array([(col[1:] + col[:-1])/2 for col in sorted_df_g.T]).T
    print(df_g.shape)
    zerothcut = sorted_df_g[0,:] + (sorted_df_g[0, :] - sorted_df_g[1,:])/2
    cuts = np.insert(cuts, 0, zerothcut, axis=0)

    ## get counts and mean
    counts_g = np.array([np.count_nonzero(df_g >= cut, axis=1) for cut in cuts])
    counts_b = np.array([np.count_nonzero(df_b >= cut, axis=1) for cut in cuts])
    avg_cnt_g = counts_g.mean(axis=1)
    avg_cnt_b = counts_b.mean(axis=1)

    # --------------- percent runs given N fails -----------------
    perc_g = np.count_nonzero(counts_g > N, axis=1)/counts_g.shape[1]
    perc_b = np.count_nonzero(counts_b > N, axis=1)/counts_b.shape[1]

    ##--------- plotting the output in the same way as rob --------------
    # algorithm_name = "combined"
    algorithm_name = "chi2" if algo=='chi2' else "max pull"
    xaxislabelsize = 14
    yaxislabelsize = xaxislabelsize
    cmssize = 18
    luminositysize = 14
    axisnumbersize = 12
    annotatesize=16
    labelpad=10
    linewidth=1.5
    legendlabel = f'{numref[0]} reference runs' if numref[0] != '1' else f'{numref[0]} reference run'
    ax1.tick_params(axis='both', which='major', labelsize=axisnumbersize)
    ax1.set_xlabel(f'Fraction of good runs with ≥{N} histogram flags', fontsize=xaxislabelsize, labelpad=labelpad)
    ax1.set_ylabel(f'Fraction of bad runs with ≥{N} histogram flags', fontsize=yaxislabelsize, labelpad=labelpad)
    ax1.axline((0, 0), slope=1, linestyle='--', linewidth=linewidth, color='#964a8b', zorder=0)
    ax1.plot(perc_g, perc_b, marker, mfc=color, color=color, mec='k', markersize=8, linewidth=1, label=legendlabel)
    ax1.axis(xmin=0,xmax=0.4,ymin=0,ymax=0.8)
    ax1.annotate(f"Beta-binomial {algorithm_name} test", xy=(0.05, 0.98), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', fontsize=annotatesize, fontstyle='italic')
    ax1.legend(loc='lower right', fontsize=annotatesize)
    #ax1.text(0, 1.02, "CMS", fontsize=cmssize, weight='bold',transform=ax1.transAxes)
    #ax1.text(0.15, 1.02, "Preliminary", fontsize=cmssize-4, fontstyle='italic', transform=ax1.transAxes)
    ax1.text(0, 1.02, "Private work (CMS data)", fontsize=cmssize-4, fontstyle='italic', transform=ax1.transAxes)
    ax1.text(0.63, 1.02, "2022 (13.6 TeV)", fontsize=luminositysize, transform=ax1.transAxes)

    ax0.tick_params(axis='both', which='major', labelsize=axisnumbersize)
    ax0.set_xlabel('Mean histogram flags per good run', fontsize=xaxislabelsize, labelpad=labelpad)
    ax0.set_ylabel('Mean histogram flags per bad run', fontsize=yaxislabelsize, labelpad=labelpad)
    ax0.axline((0, 0), slope=1, linestyle='--',linewidth=linewidth,color='#964a8b', zorder=0)
    ax0.plot(avg_cnt_g, avg_cnt_b, marker, mfc=color, color=color, mec='k', markersize=8, linewidth=1, label=legendlabel)
    ax0.annotate(f"Beta-binomial {algorithm_name} test", xy=(0.05, 0.98), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', fontsize=annotatesize, fontstyle='italic')
    ax0.axis(xmin=0,xmax=8,ymin=0,ymax=25)
    ax0.legend(loc='center left', fontsize=annotatesize, bbox_to_anchor=(0.05, 0.75), bbox_transform=ax0.transAxes)
    #ax0.text(0, 1.02, "CMS", fontsize=cmssize, weight='bold', transform=ax0.transAxes)
    #ax0.text(0.15, 1.02, "Preliminary", fontsize=cmssize-4, fontstyle='italic', transform=ax0.transAxes)
    ax0.text(0, 1.02, "Private work (CMS data)", fontsize=cmssize-4, fontstyle='italic', transform=ax0.transAxes)
    ax0.text(0.63, 1.02, "2022 (13.6 TeV)", fontsize=luminositysize, transform=ax0.transAxes)
    ## --------------------------------------------------------------------

#for cut in cuts[:4]:
#    print(cut)
fig0.savefig("HF_ROC_comparison_" + algorithm_name + ".png",bbox_inches='tight')
#print("SAVED: " + args.output_dir + "/RF_HF_ROC_comparison_" + algorithm_name + ".pdf")
fig1.savefig("RF_ROC_comparison_" + algorithm_name + ".png",bbox_inches='tight')
