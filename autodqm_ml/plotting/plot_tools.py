import matplotlib.pyplot as plt 
import numpy as np 
from pathlib import Path
import awkward
import os

import pandas as pd

from yahist import Hist1D, Hist2D

from matplotlib import colors

from datetime import datetime

import logging
logger = logging.getLogger(__name__)

def make_sse_plot(name, recos, save_name, **kwargs):
    x_label = "Anomaly Score"
    y_label = "Fraction of runs"
    log_y = kwargs.get("log_y", False)

    # Append all sse's together to get a common binning
    all = awkward.concatenate([x["score"] for k, x in recos.items()])
    h_all = Hist1D(all)
    bins = h_all.edges
    
    hists = []

    for reco, info in recos.items():
        mean = awkward.mean(info["score"])
        std = awkward.std(info["score"])
        h = Hist1D(info["score"], bins = bins, label = reco + " [N = %d, SSE = %.2E +/- %.2E]" % (len(info["score"]), mean, std))
        h = h.normalize()
        hists.append(h)

    fig, ax = plt.subplots(1, figsize=(8,6))

    for idx, h in enumerate(hists):
        h.plot(ax=ax, color = "C%d" % (idx+1), errors = True, linewidth=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xscale("log")
    ax.set_yscale("log")

    logger.debug("[plot_tools.py : make_sse_plot] Writing plot to file '%s'." % (save_name))
    plt.savefig(save_name)
    plt.savefig(save_name.replace(".pdf", ".png"))
    plt.clf()


def make_original_vs_reconstructed_plot(name, original, recos, run, save_name, hist_layout, **kwargs): 
    n_dim = len(np.array(original).shape)

    if n_dim == 1:
        make_original_vs_reconstructed_plot1d(name, original, recos, run, save_name, **kwargs)

    elif n_dim == 2:
        if hist_layout == 'flatten':
            original_flat = awkward.flatten(original, axis = -1)
            recos_flat = {}
            for algorithm, reco in recos.items():
                recos_flat[algorithm] = {
                          "reco" : awkward.flatten(reco["reco"], axis = -1),
                          "score" : reco["score"]
                }
            make_original_vs_reconstructed_plot1d(name, original_flat, recos_flat, run, save_name, **kwargs)     
        elif hist_layout == '2d':
            make_original_vs_reconstructed_plot2d(name, original, recos, run, save_name, **kwargs)
        else:
            message = "[plot_tools.py : make_original_vs_reconstructed_plot] Please specify a valid histogram layout option: flatten (default), 2d"
            logger.exception(message)
            raise RuntimeError()

    else:
        message = "[plot_tools.py : make_original_vs_reconstructed_plot] Plotting not implemented for histograms with dimension %d." % (n_dim)
        logger.exception(message)
        raise RuntimeError()

def make_original_vs_reconstructed_plot1d(name, original, recos, run, save_name, **kwargs):
    bins = "%s, 0, 1" % (len(original))
    x_label = name + " (a.u.)"
    y_label = "Fraction of events"
    
    rat_lim = kwargs.get("rat_lim", [0.0, 2.0])
    log_y = kwargs.get("log_y", False)

    h_orig = Hist1D(original, bins = bins, label = "original")
    h_orig._counts = original
    h_reco = []
   
    for reco, info in recos.items():
        h = Hist1D(info["reco"], bins = bins, label = "%s [sse : %.2E]" % (reco, info["score"]))
        h._counts = info["reco"]
        h_reco.append(h)

    fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(8,6), gridspec_kw=dict(height_ratios=[3, 1]))
    plt.grid()

    h_orig.plot(ax=ax1, color="black", errors = False, linewidth=2)
    plt.sca(ax1)

    for idx, h in enumerate(h_reco):
        if "pca" in save_name: idx = idx
        if ("ae" in save_name) or ("autoencoder" in save_name): idx = idx + 1

    for idx, h in enumerate(h_reco):
        if "pca" in save_name: idx = idx
        if ("ae" in save_name) or ("autoencoder" in save_name): idx = idx + 1
        ratio = h.divide(h_orig)
        ratio.metadata["label"] = None
        ratio.plot(ax=ax2, color = "C%d" % (idx+1), errors = False, linewidth=2)

    ax1.set_ylabel(y_label)
    ax2.set_ylabel("ML Reco / Original")
    ax2.set_xlabel(x_label)
    ax2.set_ylim(rat_lim)
    #ax1.set_ylim([0.0, awkward.max(original) * 1.5])

    if log_y:
        ax1.set_yscale("log")

    logger.debug("[plot_tools.py : make_original_vs_reconstructed_plot1d] Writing plot to file '%s'. " % (save_name))
    plt.savefig(save_name)
    plt.savefig(save_name.replace(".pdf", ".png"))
    plt.clf()

def make_original_vs_reconstructed_plot2d(name, original, recos, run, save_name, **kwargs):
    x_label = name + " (a.u.)"
    y_label = "Fraction of events"
    extent = (0, 1, 0, 1)
    color_map = plt.cm.Purples
    rat_lim = kwargs.get("rat_lim", [0.0, 2.0])
    log_y = kwargs.get("log_y", False)
    h_reco = []
    labels = []
    base_vmax = awkward.max(original)
    base_vmin = awkward.min(original)
    ratio_vmax = -10
    ratio_vmin = 10
    ratios = []
    for reco, info in recos.items():
        h_reco.append(info["reco"])
        ratio = np.abs(info["reco"] - original)
        ratios.append(ratio)
        base_vmax = np.max((base_vmax, awkward.max(info["reco"])))
        base_vmin = np.min((base_vmin, awkward.min(info["reco"])))
        ratio_vmax = np.max((ratio_vmax, awkward.max(ratio)))
        ratio_vmin = np.min((ratio_vmin, awkward.min(ratio)))
        labels.append("%s [sse : %.2E]" % (reco, info["score"]))
    
    fig, axes = plt.subplots(2, len(h_reco) + 1, figsize=(5 + len(h_reco)*5, 6), gridspec_kw=dict(height_ratios=[3, 1]), sharey = True, sharex=True)
     
    if log_y:
        base_norm = colors.LogNorm(base_vmin, base_vmax)
        ratio_norm = colors.LogNorm(ratio_vmin, ratio_vmax)
    else:
        base_norm = colors.Normalize(base_vmin, base_vmax)
        ratio_norm = colors.Normalize(ratio_vmin, ratio_vmax)
    #plt.grid()
    axes[0][0].imshow(original, norm=base_norm, cmap=color_map, extent = extent, aspect = 'auto')
    axes[0][0].set_title("Original")
    #plt.colorbar(mesh, ax = axes[0][0])
    axes[0][0].grid()
    for idx, h in enumerate(h_reco):
        if idx == len(h_reco) - 1:
            pos = axes[0][idx+1].imshow(h, norm=base_norm, cmap=color_map, extent = extent, aspect = 'auto')
            cax = axes[0][idx+1].inset_axes([1.1, 0, 0.1, 1])
            plt.colorbar(pos, cax = cax, ax = axes[0][idx+1])
            pos = axes[1][idx+1].imshow(ratios[idx], norm=ratio_norm, cmap=color_map, extent = extent, aspect = 'auto')
            cax = axes[1][idx+1].inset_axes([1.1, 0, 0.1, 1])
            plt.colorbar(pos, cax = cax, ax = axes[1][idx+1])
        else:
            pos = axes[0][idx+1].imshow(h, norm=base_norm, cmap=color_map, extent = extent, aspect = 'auto')
            pos = axes[1][idx+1].imshow(ratios[idx], norm=ratio_norm, cmap=color_map, extent = extent, aspect = 'auto')
        axes[0][idx+1].set_title(labels[idx])
        axes[0][idx+1].grid()
        axes[1][idx+1].grid()
    axes[1][0].remove()
    axes[0][0].set_ylabel(y_label)
    axes[1][1].set_ylabel("ML Reco - Original")
    axes[0][0].set_xlabel(x_label)


    logger.debug("[plot_tools.py : make_original_vs_reconstructed_plot1d] Writing plot to file '%s'. " % (save_name))
    plt.savefig(save_name, bbox_inches='tight')
    plt.savefig(save_name.replace(".pdf", ".png"), bbox_inches='tight')
    plt.clf()
    
def plot1D(original_hist, reconstructed_hist, run, hist_path, algo, threshold):    
    """
    plots given original and recontructed histogram. Will plot the MSE plot if the SSE is over the threshold. 

    :param original_hist: original histogram to be plotted  
    :type original_hist: numpy array of shape (n, )
    :param reconstructed_hist: reconstructed histogram from the ML algorithm
    :type reconstructed_hist: numpy array of shape (n, )
    :param hist_path: name of histogram
    :type hist_path: str
    :param algo: name of algorithm used. This is used to label the folder. Can use self.name to be consistent between this and plotMSESummary
    :param type: str
    :param threshold: threshold to determind histogram anomaly
    :type threshold: int
    """
    fig, ax = plt.subplots()
    mse = np.mean(np.square(original_hist - reconstructed_hist))
    
    # for bin edges
    binEdges = np.linspace(0, 1, original_hist.shape[0])
    width = binEdges[1] - binEdges[0]
    # plot original/recon 
    ax.bar(binEdges, original_hist, alpha=0.5, label='original', width=width)
    ax.bar(binEdges, reconstructed_hist, alpha=0.5, label='reconstructed', width=width)
    plotname = hist_path.split('/')[-1]
    ax.set_title(f'{plotname} {run} {algo}')
    leg = ax.legend(loc='upper right')
    text = '\n'.join((
        f'mse: {mse:.4e}',
        ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(.8, 0.1*max(original_hist.max(), reconstructed_hist.max()), text, wrap=True, bbox=props )
    # create directory to save plot
    Path(f'plots/{algo}/{run}').mkdir(parents=True, exist_ok=True)
    fig.savefig(f'plots/{algo}/{run}/{plotname}.png')
    plt.close('all')
    
    if mse > threshold: 
        fig2, ax2 = plt.subplots()
        ax2.bar(binEdges, np.square(original_hist - reconstructed_hist), alpha=0.5, width=width)
        ax2.set_title(f'MSE {plotname} {run}')
        fig2.savefig(f'plots/{algo}/{run}/{plotname}-MSE.png')
        plt.close('all')
    

def plotMSESummary(original_hists, reconstructed_hists, threshold, hist_paths, runs, algo): 
    """ 
    Plots all the MSE on one plot that also shows how many passese threhold
    

    :param original_hist: original histogram to be plotted  
    :type original_hist: numpy array of shape (n, )
    :param reconstructed_hist: reconstructed histogram from the ML algorithm
    :type reconstructed_hist: numpy array of shape (n, )
    :param threshold: threshold to determind histogram anomaly
    :type threshold: int
    :param hist_path: list of name of histograms
    :type hist_path: list
    :param runs: list of runs used for testing. Must be same as list passed into the pca or autoencoder.plot function 
    :param type: list
    :param algo: name of algorithm used. This is used to place the plot in the correct folder. Can use self.name to be consistent between this and plot1D
    :param type: str

    """ 
    ## convert to awkward array as not all hists have the same length
    original_hists = awkward.Array(original_hists)
    reconstructed_hists = awkward.Array(reconstructed_hists)
    
    fig, ax = plt.subplots()
    mse = np.mean(np.square(original_hists - reconstructed_hists), axis=1)
    
    ## count number of good and bad histogrms
    num_good_hists = np.count_nonzero(mse < threshold)
    num_bad_hists = np.count_nonzero(mse > threshold)
    
    ## get names of top 5 highest mse histograms
    ## plot_names[argsort[-1]] should be the highest mse histogram
    sortIdx = np.argsort(mse)

    hist,_, _ = ax.hist(mse)
    ax.set_xlabel('MSE values') 
    ax.set_title('Summary of all MSE values')
    
    numHistText = [f'num good hists: {num_good_hists}', f'num bad hists: {num_bad_hists}']
    ## mse summary is for all plots in all runs, so need to make names accordingly
    hist_names = [hist_path.split('/')[-1] for hist_path in hist_paths]
    hist_names_runs = [f'{hist_name} ({run})' for run in runs for hist_name in hist_names]


    ## account for case less than 5 plots were tested
    maxidx = min(5, len(hist_names_runs))+1
    ## plot_name is the whole directory, so we only need the last for histname
    rankHistText = [f'{hist_names_runs[sortIdx[i]].split("/")[-1]}: {mse[sortIdx[i]]:.4e}' for i in range(-1, -maxidx, -1)]
    text = '\n'.join(
        numHistText + rankHistText
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(1.1*max(mse), 0.5*max(hist), text, wrap=True, bbox=props)
    
    fig.savefig(f'plots/{algo}/MSE_Summary.png', bbox_inches='tight')


def plot_roc_curve(h_name, results, save_name, **kwargs):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.yaxis.set_ticks_position('both')
    ax1.grid(True)

    log = kwargs.get("log", False)

    idx = 0
    for algo, res in results.items():
        ax1.plot(
                res["fpr"],
                res["tpr"],
                color = "C%d" % (idx+1),
                label = algo + " [AUC = %.3f +/- %.3f]" % (res["auc"], res["auc_unc"])
        )
        ax1.fill_between(
                res["fpr"],
                res["tpr"] - (res["tpr_unc"] / 2.),
                res["tpr"] + (res["tpr_unc"] / 2.),
                color = "C%d" % (idx+1),
                alpha = 0.25
        )
        idx += 1

    if not log:
        plt.ylim(0,1)
        plt.xlim(0,1)
    else:
        plt.xlim(0.005, 1)
        plt.ylim(0,1)
        ax1.set_xscale("log")

    plt.xlabel("False Anomaly Rate (FPR)")
    plt.ylabel("Anomaly Detection Efficiency (TPR)")

    legend = ax1.legend(loc='lower right')

    logger.debug("[plot_tools.py : plot_roc_curve] Writing plot to file '%s'." % (save_name))
    plt.savefig(save_name)
    plt.savefig(save_name.replace(".pdf", ".png"))
    plt.clf()

def plot_rescaled_score_hist(data, hist, savename):
    
    fig, axes = plt.subplots(len(data['score']), 1, figsize = (12, 4*len(data['score'])))
    if len(data['score']) == 1:
        axes = [axes]
    for i in range(len(data['score'])):
        score = data['score'][i]
        score_min = awkward.min(score)
        score_max = awkward.max(score) - score_min
        score = score - awkward.min(score)
        score = score/awkward.max(score)
        axes[i].hist(score, bins = np.logspace(np.log10(1e-4),np.log10(1.0), 100), color = 'tab:blue', alpha = 0.8, label = 'All Runs')
        axes[i].set_ylabel(data['algo'][i])
        axes[i].set_yscale('log')
        axes[i].set_xscale('log')
        if 'bad' in data:
            badax = axes[i].twinx()
            bad = data['bad'][i]
            bad = bad - score_min
            bad = bad/score_max
            badax.hist(bad, bins = np.logspace(np.log10(1e-4),np.log10(1.0), 100), range = (0, 1), color = 'tab:orange', alpha = 0.8, label ='Bad Runs')
            axes[i].spines['left'].set_color('tab:blue')
            badax.xaxis.label.set_color('tab:orange')
            axes[i].xaxis.label.set_color('tab:blue')
            if i == 0:
               badax.set_ylabel('Anomalous Runs')
            badax.spines['right'].set_color('tab:orange')
            badax.set_xscale('log')    
    fig.suptitle(hist)
    axes[0].legend()
    axes[0].set_title('Min-Max Scaled Anomaly Scores')
    fig.savefig(savename, bbox_inches = 'tight')
    fig.savefig(savename.replace('.png', '.pdf'), bbox_inches = 'tight')
    
def make_training_plots(history, hist, save_file):
        epochs = range(len(history['loss']))
        print(len(history.columns))
        fig, axes = plt.subplots(1, len(history.columns), figsize = (len(history.columns)*9, 9))
        i = 0
        fig.suptitle(hist, fontsize = 22)
        for stat, y in history.items():
            axes[i].plot(epochs, y)
            axes[i].set_xlabel('Epoch', fontsize = 15)
            axes[i].set_title(stat, fontsize = 18)
            axes[i].set_yscale('log')
            i += 1
        plt.savefig(save_file, bbox_inches = 'tight')
def multi_exp_plots(paths, xlabel, x, title, legend = None, logx = False, logy = False):
    fig, axes = plt.subplots(1, 4, figsize = (36, 9))
    i = 0
    fig.suptitle(title, fontsize = 22)
    if not legend:
        legend = [None]*len(x)
    if type(paths) == list:
        for i in range(len(paths)):
            make_one_var_exp_plots(paths[i], xlabel, x, axes, legend[i], logx)
        if paths[0][len(paths[0]) - 1] == '/':
            savepath = paths[0][:len(paths[0]) - 1]
        else:
            savepath = paths[0]
        filename = title.replace(' ', '_')
        for c in '()[]{}/.,:;?!@#$^&*':
            filename = filename.replace(c, '')
        savepath = savepath[:str.rindex(savepath, '/')] + '/' + filename + '_plots.png'
        print(savepath)
    else:
        make_one_var_exp_plots(paths, xlabel, x, axes, legend, logx)
        plt.savefig(paths + 'plots.png', bbox_inches = 'tight')

def make_one_var_exp_plots(path, xlabel, x, axes, label = None, logx = False):
    data = {'Epochs Trained':[], 'Epochs Trained Std':[],
            'Best Train Loss':[], 'Best Train Loss Std':[],
            'Best Validation Loss':[], 'Best Validation Loss Std':[],
            'Ending Learning Rate':[], 'Ending Learning Rate Std':[]}
    levels = [dir for dir in os.listdir(path) if not '.png' in dir]
    if not label:
        if path[len(path) - 1] == '/':
            label = path[:len(path) - 1]
            label = label[label.rindex('/') + 1:]
        else:
            label = path[path.rindex('/')]
    for level in levels:
        dirs = os.listdir(path+level)
        files = [file for file in dirs if '.csv' in file]
        subset = {'Epochs Trained':[], 'Best Train Loss':[], 'Best Validation Loss':[],'Ending Learning Rate':[]}
        for file in files:
            filepath = path + level + '/' + file
            df = pd.read_csv(filepath)
            subset['Epochs Trained'].append(len(df['loss']))
            subset['Best Train Loss'].append(np.min(df['loss']))
            subset['Best Validation Loss'].append(np.min(df['val_loss']))
            subset['Ending Learning Rate'].append(df['lr'].iloc[len(df['loss']) - 1])
        for item, values in subset.items():
            data[item].append(np.mean(values))
            data[item + ' Std'].append(np.std(values))
    i = 0
    for item in data:
        if not 'Std' in item:
          axes[i].errorbar(x, data[item], data[item + ' Std'], label = label)
          axes[i].set_title(item, fontsize = 18)
          if logx:
            axes[i].set_xscale('log')
          if i == 3:
            axes[i].set_yscale('log')
          axes[i].set_xlabel(xlabel, fontsize = 15)
          i += 1


def multi_exp_bar_plots(paths, xlabel, title, legend = None):
    b = 9
    s = b/(len(xlabel) + .5)
    fig, axes = plt.subplots(1, 5, figsize = (45, 9))
    fig.suptitle(title, fontsize = 22)
    if not legend:
        legend = [None]*len(paths)
    if type(paths) == list:
        for i in range(len(paths)):
            make_one_var_exp_bar_plots(paths[i], xlabel, axes, i, b, s, len(paths), legend[i])
        if paths[0][len(paths[0]) - 1] == '/':
            savepath = paths[0][:len(paths[0]) - 1]
        else:
            savepath = paths[0]
        filename = title.replace(' ', '_')
        for c in '()[]{}/.,:;?!@#$^&*':
            filename = filename.replace(c, '')
        savepath = savepath[:str.rindex(savepath, '/')] + '/' + filename + '_plots.png'
        print(savepath)
    else:
        
        make_one_var_exp_bar_plots(paths, xlabel, axes, 0, b, s, 1, legend[0])
        plt.savefig(paths + 'plots.png', bbox_inches = 'tight')

def make_one_var_exp_bar_plots(path, xlabel, axes, i, b, s, n, label = None):
    data = {'Epochs Trained':[], 'Epochs Trained Std':[],
            'Best Train Loss':[], 'Best Train Loss Std':[],
            'Best Validation Loss':[], 'Best Validation Loss Std':[],
            'Ending Learning Rate':[], 'Ending Learning Rate Std':[]}
    levels = [dir for dir in os.listdir(path) if not '.png' in dir and not 'assess' in dir]
    if not label:
        if path[len(path) - 1] == '/':
            label = path[:len(path) - 1]
            label = label[label.rindex('/') + 1:]
        else:
            label = path[path.rindex('/')]
    for level in levels:
        dirs = os.listdir(path+level)
        files = [file for file in dirs if '.csv' in file]
        subset = {'Epochs Trained':[], 'Best Train Loss':[], 'Best Validation Loss':[],'Ending Learning Rate':[]}
        for file in files:
            filepath = path + level + '/' + file
            df = pd.read_csv(filepath)
            subset['Epochs Trained'].append(len(df['loss']))
            subset['Best Train Loss'].append(np.min(df['loss']))
            subset['Best Validation Loss'].append(np.min(df['val_loss']))
            subset['Ending Learning Rate'].append(df['lr'].iloc[len(df['loss']) - 1])
            if 'mse' in df.columns:
                if 'Best MSE' not in data:
                    data['Best MSE'] = []
                    data['Best MSE Std'] = []
                if 'Best MSE' not in subset:
                    subset['Best MSE'] = []
                subset['Best MSE'].append(np.min(df['mse']))
        for item, values in subset.items():
            data[item].append(np.mean(values))
            data[item + ' Std'].append(np.std(values))
            if 'MSE' in item:
                print(level, np.mean(values))
                print(values)
    m = 0
    for item in data:
        if not 'Std' in item:
          x = [s + 2*s*i + j*b for j in range(len(xlabel))]
          print(data[item])
          axes[m].bar(x, data[item], yerr = data[item + ' Std'], width = (b-.5)/n, label = label)
          axes[m].set_title(item, fontsize = 18)
          if 'Loss' in item:
            axes[m].set_yscale('log')
          axes[m].set_xticks(x, xlabel, fontsize = 15)
          m += 1
