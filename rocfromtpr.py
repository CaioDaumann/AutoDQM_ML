import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

fns = [
    '/afs/cern.ch/work/c/csutanta/ML_metastudy/rob_csv/rob_ae_roc.csv',
    #'/afs/cern.ch/work/c/csutanta/ML_metastudy/rob_csv/rob_combined_roc.csv',
    '/afs/cern.ch/work/c/csutanta/ML_metastudy/rob_csv/rob_pca_roc.csv'
]
Ns = [1,3,5]
algorithm_names = ['ae', 'pca'] #'combined',

xaxislabelsize = 14
yaxislabelsize = xaxislabelsize
cmssize = 18
luminositysize = 14
axisnumbersize = 12
annotatesize=16
labelpad=10
linewidth=1.5



for fn, algorithm_name, label in zip(fns, algorithm_names, ['Autoencoder', 'PCA']):
    fig, axs = plt.subplots(ncols=2,nrows=1,figsize=(12,6))
    fig0, ax0 = plt.subplots(figsize=(6,6))
    fig1, ax1 = plt.subplots(figsize=(6,6))
    df = pd.read_csv(fn)
    hf_good = df[f'{algorithm_name}_hf_roc_good']
    hf_bad = df[f'{algorithm_name}_hf_roc_bad']

    ax0.tick_params(axis='both', which='major', labelsize=axisnumbersize)
    ax0.set_xlabel('Mean histogram flags per good run', fontsize=xaxislabelsize, labelpad=labelpad)
    ax0.set_ylabel('Mean histogram flags per bad run', fontsize=yaxislabelsize, labelpad=labelpad)
    ax0.axline((0, 0), slope=1, linestyle='--',linewidth=linewidth,color='#964a8b', zorder=0)
    ax0.plot(hf_good, hf_bad, '-D', mfc="#e42536", color='#e42536', mec='k', markersize=8, linewidth=1)
    ax0.annotate(f"{label} test", xy=(0.05, 0.98), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', fontsize=annotatesize, fontstyle='italic')
    ax0.axis(xmin=0,xmax=8,ymin=0,ymax=25)
    #ax0.legend(loc='center left', fontsize=annotatesize, bbox_to_anchor=(0.05, 0.75), bbox_transform=ax0.transAxes)
    ax0.text(0, 1.02, "CMS", fontsize=cmssize, weight='bold', transform=ax0.transAxes)
    ax0.text(0.15, 1.02, "Preliminary", fontsize=cmssize-4, fontstyle='italic', transform=ax0.transAxes)
    ax0.text(0.63, 1.02, "2022 (13.6 TeV)", fontsize=luminositysize, transform=ax0.transAxes)
    fig0.savefig("HF_ROC_comparison_" + algorithm_name + ".pdf",bbox_inches='tight')

    for N, marker, color in zip(Ns, ['-D', '-o', '-^'], ["#e42536", "#f89c20", "#5790fc",]):
        rf_good = df[f'{algorithm_name}_rf_n{N}_roc_good']
        rf_bad = df[f'{algorithm_name}_rf_n{N}_roc_bad']
        ax1.tick_params(axis='both', which='major', labelsize=axisnumbersize)
        ax1.set_xlabel(f'Fraction of good runs with ≥N histogram flags', fontsize=xaxislabelsize, labelpad=labelpad)
        ax1.set_ylabel(f'Fraction of bad runs with ≥N histogram flags', fontsize=yaxislabelsize, labelpad=labelpad)
        ax1.axline((0, 0), slope=1, linestyle='--', linewidth=linewidth, color='#964a8b', zorder=0)
        ax1.plot(rf_good, rf_bad, marker, mfc=color, color=color, mec='k', markersize=8, linewidth=1, label=f'SSE thresholds, N = {N}')
        ax1.axis(xmin=0,xmax=0.4,ymin=0,ymax=0.8)
        ax1.annotate(f"{label} test", xy=(0.05, 0.98), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', fontsize=annotatesize, fontstyle='italic')
        ax1.legend(loc='lower right', fontsize=annotatesize)
        ax1.text(0, 1.02, "CMS", fontsize=cmssize, weight='bold',transform=ax1.transAxes)
        ax1.text(0.15, 1.02, "Preliminary", fontsize=cmssize-4, fontstyle='italic', transform=ax1.transAxes)
        ax1.text(0.63, 1.02, "2022 (13.6 TeV)", fontsize=luminositysize, transform=ax1.transAxes)


    fig1.savefig("RF_ROC_comparison_" + algorithm_name + ".pdf",bbox_inches='tight')
