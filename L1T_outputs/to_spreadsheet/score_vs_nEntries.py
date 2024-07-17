import pandas as pd
import numpy as np
import ast
import re
from ast import literal_eval
import matplotlib.pyplot as plt

def chi2_and_nEntries(df):

    # Reading the Chi2 scores and the nEntries scores
    column_names_chi2, column_names_pull, column_names_nEntries = [], [], []
    for key in df.keys():
        if "_chi2_score" in key:
            column_names_chi2.append(key)
        if "_pull_score" in key:
            column_names_pull.append(key)
        if "nEntries_score" in key:
            column_names_nEntries.append( key  )

    Chi2_scores = [np.array(df[column_name]) for column_name in column_names_chi2 ]
    pull_scores = [np.array(df[column_name]) for column_name in column_names_pull ]
    nEntries_scores = [np.array(df[column_name]) for column_name in column_names_nEntries ]

    # Flatten the lists of arrays
    Chi2_scores_flat = np.concatenate(Chi2_scores)
    pull_scores_flat = np.concatenate(pull_scores)
    nEntries_scores_flat = np.concatenate(nEntries_scores)

    return Chi2_scores_flat, pull_scores_flat, nEntries_scores_flat

def ranked_chi2_and_nEntries(df):

    # Reading the Chi2 scores and the nEntries scores
    column_names_chi2, column_names_pull, column_names_nEntries = [], [], []
    for key in df.keys():
        if "_chi2_score" in key:
            column_names_chi2.append(key)
        if "_pull_score" in key:
            column_names_pull.append(key)
        if "nEntries_score" in key:
            column_names_nEntries.append( key  )

    Chi2_scores     = [np.argsort(np.argsort(np.array(df[column_name]))) for column_name in column_names_chi2 ]
    pull_scores     = [np.argsort(np.argsort(np.array(df[column_name]))) for column_name in column_names_pull ]
    nEntries_scores = [np.argsort(np.argsort(np.array(df[column_name]))) for column_name in column_names_nEntries ]

    # Flatten the lists of arrays
    Chi2_scores_flat = np.concatenate(Chi2_scores)
    nEntries_scores_flat = np.concatenate(nEntries_scores)
    pull_scores_flat = np.concatenate(pull_scores)

    return Chi2_scores_flat, pull_scores_flat, nEntries_scores_flat

def pull_and_nEntries(df):

    # Reading the Chi2 scores and the nEntries scores
    column_names_chi2, column_names_nEntries = [], []
    for key in df.keys():
        if "_pull_score" in key:
            column_names_chi2.append(key)
        if "nEntries_score" in key:
            column_names_nEntries.append( key  )

    Chi2_scores = [np.array(df[column_name]) for column_name in column_names_chi2 ]
    nEntries_scores = [np.array(df[column_name]) for column_name in column_names_nEntries ]

    # Flatten the lists of arrays
    Chi2_scores_flat = np.concatenate(Chi2_scores)
    nEntries_scores_flat = np.concatenate(nEntries_scores)

    return Chi2_scores_flat, nEntries_scores_flat

def chi2_and_LS(df):

    ls_array = df['run_ls_rec']
    # Reading the Chi2 scores and the nEntries scores
    column_names_chi2, column_names_pull, column_names_nEntries = [], [], []
    Chi2_scores, pull_scores, LS_scores = [], [], []
    for key in df.keys():
        if "_chi2_score" in key:
            Chi2_scores.append(np.array(df[key]))
            LS_scores.append( np.array(ls_array) ) 
        if "_pull_score" in key:
            pull_scores.append(np.array(df[key]))

    # Flatten the lists of arrays
    Chi2_scores_flat = np.concatenate(Chi2_scores)
    pull_scores_flat = np.concatenate(pull_scores)
    LS_scores_flat = np.concatenate(LS_scores)

    return Chi2_scores_flat, pull_scores_flat, LS_scores_flat

def chi2_and_rec_lumi(df):

    rec_lumi_array = df['rec_lumi']
    # Reading the Chi2 scores and the nEntries scores
    column_names_chi2, column_names_pull , column_names_nEntries = [], [], []
    Chi2_scores, pull_scores, rec_lumi_scores = [], [], []
    for key in df.keys():
        if "_chi2_score" in key:
            Chi2_scores.append(np.array(df[key]))
            rec_lumi_scores.append( np.array(rec_lumi_array) )   
        if "_pull_score" in key:
            pull_scores.append(np.array(df[key]))        

    # Flatten the lists of arrays
    Chi2_scores_flat = np.concatenate(Chi2_scores)
    rec_lumi_scores_flat = np.concatenate(rec_lumi_scores)
    pull_scores_flat = np.concatenate(pull_scores)

    return Chi2_scores_flat, pull_scores_flat, rec_lumi_scores_flat

def main_plotter(Chi2_1d_good, nEntries_1d_good, Chi2_1d_bad, nEntries_1d_bad, Chi2_2d_good, nEntries_2d_good, Chi2_2d_bad, nEntries_2d_bad, IsLS = False, LogScale = False, IsnEntries = False, IsLumi = False, algorithm = 'Chi2'):
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Scatter plot in the top left subplot (1, 1)
    axs[0, 0].scatter([nEntries_1d_good],[Chi2_1d_good], alpha=0.5, color = 'blue', s=10)
    axs[0, 0].set_title('1D Histograms - Good Runs', fontsize=18)
    #axs[0, 0].set_xlabel('Ranked nEntries')
    if IsLumi:
        axs[0, 0].set_ylabel(f'{algorithm} Scores', fontsize=16)
    elif IsLS:
        axs[0, 0].set_ylabel(f'{algorithm} Scores', fontsize=16)
    elif IsnEntries:
        axs[0, 0].set_ylabel(f'{algorithm} Scores', fontsize=16)
    else:
        axs[0, 0].set_ylabel(f'Ranked {algorithm} Scores', fontsize=16)
    
    if LogScale:
        axs[0, 0].set_xscale('log')
        axs[0, 0].set_yscale('log')

    # Scatter plot in the top right subplot (1, 2)
    axs[0, 1].scatter([nEntries_1d_bad], [Chi2_1d_bad], alpha=0.5, color = 'red', s=10)
    axs[0, 1].set_title('1D Histograms - Bad Runs', fontsize=18)
    #axs[0, 1].set_xlabel('Ranked nEntries')
    #axs[0, 1].set_ylabel('Ranked Chi2 Scores')
    if LogScale:
        axs[0, 1].set_xscale('log')
        axs[0, 1].set_yscale('log')

    # Scatter plot in the bottom left subplot (2, 1)
    axs[1, 0].scatter([nEntries_2d_good], [Chi2_2d_good], alpha=0.5, color = 'blue', s=10)
    axs[1, 0].set_title('2D Histograms - Good Runs', fontsize=18)
    if IsLumi:
        axs[1, 0].set_xlabel('dcs_rec_lumi', fontsize=16)
        axs[1, 0].set_ylabel(f'{algorithm} Scores', fontsize=16)
    elif IsLS:
        axs[1, 0].set_xlabel('run_ls_rec', fontsize=16)
        axs[1, 0].set_ylabel(f'{algorithm} Scores', fontsize=16)
    elif IsnEntries:
        axs[1, 0].set_xlabel('nEntries', fontsize=16)
        axs[1, 0].set_ylabel(f'{algorithm} Scores', fontsize=16)
    else:
        axs[1, 0].set_xlabel('Ranked nEntries', fontsize=16)
        axs[1, 0].set_ylabel(f'Ranked {algorithm} Scores', fontsize=16)
    if LogScale:
        axs[1, 0].set_xscale('log')
        axs[1, 0].set_yscale('log')

    # Scatter plot in the bottom right subplot (2, 2)
    axs[1, 1].scatter([nEntries_2d_bad], [Chi2_2d_bad], alpha=0.5, color = 'red', s=10)
    axs[1, 1].set_title('2D Histograms - Bad Runs', fontsize=18)
    if IsLumi:
        axs[1, 1].set_xlabel('dcs_rec_lumi', fontsize=16)
    elif IsLS:
        axs[1, 1].set_xlabel('run_ls_rec', fontsize=16)
    elif IsnEntries:
        axs[1, 1].set_xlabel('nEntries', fontsize=16)
    else:
        axs[1, 1].set_xlabel('Ranked nEntries', fontsize=16)
    #axs[1, 1].set_ylabel('Ranked Chi2 Scores')
    if LogScale:
        axs[1, 1].set_xscale('log')
        axs[1, 1].set_yscale('log')

    # Retrieve the limits from the [0, 0] plot
    x_min, x_max = axs[0, 0].get_xlim()
    y_min, y_max = axs[0, 0].get_ylim()

    # Set the same x and y axis range for all plots based on the [0, 0] plot
    if LogScale:
        for ax in axs.flat:
            #ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    # Add a title to the entire figure
    fig.suptitle('Algorithms - Hot bins and Rebin', fontsize=20)

    #plt.title('Scatter Plot of Ranked Chi2 Scores vs Ranked nEntries Scores - With Hot bin/Rebin')
    # Adjust layout
    
    plt.tight_layout()
    if IsLS:
        plt.savefig(f'{algorithm}_2d_scatter_plots_LS.png')
    elif IsLumi:
        plt.savefig(f'{algorithm}_2d_scatter_plots_Lumi.png')
    elif IsnEntries:
        plt.savefig(f'{algorithm}_2d_scatter_plots_nEntries.png')
    else:
        plt.savefig(f'{algorithm}_2d_scatter_plots.png')
    
def extract_1d_and_2d_df(df):
    
    column_names = []
    column_dim   = []
    for key in df.keys():
        if "nDims" in key:
            column_names.append(key[:-22])
            column_dim.append(df[key].values[0])
                
    df_1d, df_2d = df, df  
    histos_1d = 0
    for column_name, dim in zip(column_names, column_dim):
        if dim == 2:
            df_1d = df_1d.drop(columns=[col for col in df_1d.columns if column_name in col])
        elif dim == 1:
            df_2d = df_2d.drop(columns=[col for col in df_2d.columns if column_name in col])
            histos_1d = histos_1d + 1
        
    return df_1d, df_2d


def add_lumi_related_variables(df):

    run_number = [357767,355921,360141,357756,357734,357802,357542,357688,362653,362618,361239,359762,360116,357479,356321,357758,360017,361400,356970,360413,360131,357110,360892,357778,359808,362757,361197,355988,356814,360992,360126,356531,356478,357078,357482,357538,362058,357332,359810,356998,356943,362728,360887,361333,356074,361580,357706,357809,362166,357759,362438,359871,361957,360393,356569,356577,357101,356489,357271,355872,357610,356043,360895,361443,362060,360876,359806,361297,357814,361318,356956,360296,360295,360459,360125,361364,357804,361475,356386,356323,356968,357550,357776,360400,361468,357813,357777,359763,356375,359717,362087,357893,360820,360225,362617,359699,360946,359998,360128,357106,356615,362758,356788,361107,356580,360945,356944,360224,356824,356581,357898,356309,357699,361366,357333,356563,356947,360951,357001,360019,362696,356383,361363,356381,357079,355912,360392,357701,362760,356937,362148,360888,356948,360825,360919,356999,357268,356815,361990,357440,356969,356721,361569,360890,356077,361054,361193,361280,356709,357900,362106,361044,359693,360927,361303,359685,362720,357812,357081,357805,357803,361579,359602,362059,362596,361105,356476,361223,357700,356071,357735,357472,362107,361045,359812,356568,361052,356619,356378,355865,357885,356997,362167,357330,360893,360428,362614,361573,360941,357442,360075,361361,356570,361110,362062,360090,362657,360460,357401,357441,360991,361320,361020,361948,357807,357000,361188,360796,360327,357696,360794,360826,362091,356908,356951,359814,356954,361083,359686,362654,357612,360889,359694,362698,361417,361954,360761,357406,356582,362616,357698,362655,360856,357112,357754,356719,357329,357438,356812,359809,356919,356433,362105,361272,360491,357331,362161,362597,359718,357077,359661,357899,357705,355913,359764,360942,361994,360127,361365,356955,361956,361512,360088,357732,357815,361284,357328,360874,361106,356789,356488,362153,361240,360439,361971,356076,360490,360921,362615,355892,360948,360437,356946,356005,362009,360486,362061,362154,356316,356576,360795,360950,357080,361989,357808,356446,361091,357613,356523,357611,362695,356578,360441,362104,361362,357779,362163,359751]
    run_ls_rec = [355872,355892,355912,355913,355921,356005,356076,356077,356309,356316,356323,356378,356381,356386,356433,356446,356523,356563,356569,356570,356576,356578,356582,356615,356619,356812,356814,356919,356937,356946,356947,356948,356951,356954,356955,356956,356968,356969,356970,357001,357080,357081,357112,357271,357329,357330,357332,357333,357401,357406,357438,357440,357441,357442,357479,357482,357542,357610,357611,357612,357613,357688,357696,357700,357701,357705,357706,357732,357734,357735,357754,357756,357758,357777,357778,357802,357803,357805,357807,357813,357814,357815,357898,357899,357900,359661,359685,359686,359693,359694,359699,359718,359751,359762,359763,359764,359806,359810,359812,359814,359871,359998,360019,360075,360090,360116,360125,360126,360127,360128,360131,360141,360224,360225,360295,360296,360327,360393,360400,360413,360459,360460,360490,360491,360761,360794,360795,360820,360825,360826,360876,360887,360888,360890,360892,360895,360919,360921,360927,360942,360945,360946,360948,360950,360951,360991,360992,361020,361044,361045,361054,361091,361105,361107,361110,361188,361197,361223,361239,361240,361272,361280,361297,361303,361318,361320,361333,361365,361366,361400,361417,361443,361468,361475,361512,361569,361573,361579,361580,361957,361971,361989,361990,361994,362058,362060,362061,362062,362087,362091,362104,362105,362106,362107,362148,362153,362154,362161,362166,362167,362597,362614,362615,362616,362617,362618,362653,362655,362657,362695,362696,362698,362720,362728,362758,362760,355872,355892,355912,355913,355921,355988,356005,356043,356076,356077,356309,356316,356323,356378,356381,356383,356386,356433,356446,356523,356531,356563,356568,356569,356570,356576,356578,356580,356582,356615,356619,356812,356814,356824,356908,356919,356937,356946,356947,356948,356951,356954,356955,356956,356968,356969,356970,356999,357000,357001,357079,357080,357081,357106,357112,357268,357271,357328,357329,357330,357331,357332,357333,357401,357406,357438,357440,357441,357442,357472,357479,357482,357538,357542,357550,357610,357611,357612,357613,357688,357696,357698,357699,357700,357701,357705,357706,357732,357734,357735,357754,357756,357758,357759,357777,357778,357779,357802,357803,357804,357805,357807,357808,357809,357812,357813,357814,357815,357898,357899,357900,359602,359661,359685,359686,359693,359694,359699,359718,359751,359762,359763,359764,359806,359808,359809,359810,359812,359814,359871,359998,360017,360019,360075,360090,360116,360125,360126,360127,360128,360131,360141,360224,360225,360295,360296,360327,360392,360393,360400,360413,360428,360437,360459,360460,360490,360491,360761,360794,360795,360820,360825,360826,360856,360874,360876,360887,360888,360889,360890,360892,360895,360919,360921,360927,360941,360942,360945,360946,360948,360950,360951,360991,360992,361020,361044,361045,361052,361054,361083,361091,361105,361106,361107,361110,361188,361193,361197,361223,361239,361240,361272,361280,361284,361297,361303,361318,361320,361333,361361,361362,361363,361365,361366,361400,361417,361443,361468,361475,361512,361569,361573,361579,361580,361957,361971,361989,361990,361994,362058,362059,362060,362061,362062,362087,362091,362104,362105,362106,362107,362148,362153,362154,362161,362163,362166,362167,362597,362614,362615,362616,362617,362618,362653,362654,362655,362657,362695,362696,362698,362720,362728,362757,362758,362760,360486,355865,360088,361954,361948,361956,356071,356074,356375,356478,356476,356489,356488,356581,356577,356709,356719,356721,356815,356944,356998,357101,357776,357767,357893,360441,360439,356321,356943,356997,357078,357077,357110,359717,360796,361364,362009,362438,362596,356788,356789,357885,360893]
    dcs_cms_Nls = [1216,184,158,106,405,187,152,472,124,141,649,304,1193,122,310,614,901,308,250,98,183,865,104,1297,173,115,672,88,119,129,350,88,284,365,380,109,185,236,366,183,662,759,519,1580,668,157,430,207,617,125,196,354,84,1373,1046,220,251,191,412,736,256,380,379,758,344,203,161,128,300,1126,118,425,85,86,358,164,154,88,244,293,212,944,313,638,516,81,141,934,614,1080,1856,368,1007,235,474,1111,369,184,228,366,228,1863,1488,792,794,356,188,369,201,163,378,751,225,355,1441,519,352,124,80,97,1339,1077,1199,179,157,682,412,885,359,119,347,123,297,519,307,700,798,218,1893,348,99,369,123,717,190,236,99,399,125,947,461,381,420,156,197,141,2177,695,1056,1015,131,272,933,2240,457,382,302,489,201,456,1622,2274,1881,1233,1538,569,305,872,422,1163,2343,115,204,116,171,89,314,214,146,1418,120,184,83,785,720,144,784,124,337,646,139,153,153,358,244,274,266,356,198,334,1715,374,903,388,215,728,1216,184,158,106,405,48,187,65,152,472,124,141,649,304,1193,36,122,310,614,901,56,308,23,250,98,183,865,50,104,1297,173,115,672,66,26,88,119,129,350,88,284,365,380,109,185,236,366,58,50,183,22,662,759,60,519,74,1580,62,668,157,24,430,207,617,125,196,354,84,1373,27,1046,220,26,251,36,191,412,736,256,380,379,63,30,758,344,203,161,128,300,1126,118,425,85,70,86,358,81,164,154,23,88,244,17,41,50,293,212,944,313,638,516,74,81,141,934,614,1080,1856,368,1007,235,474,1111,369,46,44,184,228,366,228,1863,31,1488,792,794,356,188,369,201,163,378,751,225,355,1441,519,352,65,124,80,97,80,44,1339,1077,1199,179,157,682,412,885,359,119,56,53,347,123,297,36,519,307,700,798,218,1893,76,348,99,369,123,717,190,236,99,399,125,947,65,461,54,381,420,31,156,197,141,29,2177,695,1056,1015,131,272,36,933,2240,457,382,302,47,31,27,489,201,456,1622,2274,1881,1233,1538,569,305,872,422,1163,2343,115,204,116,171,67,89,314,214,146,1418,120,184,83,785,720,144,784,124,65,337,646,139,153,153,358,244,274,266,26,356,198,334,1715,374,903,388,66,215,728,431,287,314,190,345,146,362,81,147,671,28,475,231,93,31,122,890,33,219,23,16,130,562,17,15,145,15,85,197,123,162,91,28,339,556,160,19,89,35,128,117,96,16]
    dcs_rec_lumi = [70.976,10.924,9.106,5.910,23.071,12.310,17.350,50.371,11.076,9.399,72.933,43.024,132.922,8.806,48.381,93.899,169.023,55.209,45.846,19.321,26.322,156.373,15.324,233.770,24.281,32.247,158.692,21.426,28.653,34.914,91.705,20.189,61.221,68.792,64.082,15.146,51.261,67.040,96.039,51.476,206.244,178.680,135.803,424.072,219.769,42.968,99.092,43.544,169.338,33.717,62.501,115.002,25.104,313.360,331.908,50.718,84.347,45.660,111.390,157.137,46.068,130.476,130.816,210.809,69.917,38.924,27.146,41.299,109.792,322.444,33.660,149.673,25.473,23.462,87.988,53.533,53.255,29.888,78.722,80.145,51.516,196.390,106.366,188.409,119.376,9.177,21.131,114.977,164.151,217.971,413.549,122.746,313.368,84.746,156.599,265.505,144.757,62.814,71.102,101.714,99.130,576.254,471.287,312.584,201.562,139.463,68.686,120.811,58.461,44.517,88.463,293.394,81.270,139.288,523.430,133.299,133.399,49.577,23.999,36.249,461.025,235.193,375.634,40.729,9.149,269.149,149.479,349.531,114.620,34.114,21.256,43.211,123.331,195.420,96.691,177.556,324.443,79.605,662.281,142.238,38.520,142.448,41.796,211.486,44.858,27.328,10.846,25.876,46.298,379.783,132.244,49.982,146.106,55.437,65.709,54.825,686.442,273.949,425.948,293.223,50.626,110.970,380.716,773.255,189.057,149.279,121.018,193.981,74.411,184.623,616.365,781.085,705.547,487.725,583.544,229.059,123.051,346.739,142.028,327.848,814.148,42.248,85.198,47.985,62.573,36.154,129.374,86.404,61.346,502.542,41.904,74.354,32.881,317.822,292.404,58.078,268.360,49.761,139.245,258.914,58.378,65.070,68.310,160.454,109.087,101.298,103.313,144.545,81.396,130.171,592.699,81.903,360.052,145.363,87.675,285.735,70.976,10.924,9.106,5.910,23.071,3.040,12.310,5.653,17.350,50.371,11.076,9.399,72.933,43.024,132.922,3.068,8.806,48.381,93.899,169.023,9.079,55.209,3.237,45.846,19.321,26.322,156.373,7.829,15.324,233.770,24.281,32.247,158.692,9.594,7.217,21.426,28.653,34.914,91.705,20.189,61.221,68.792,64.082,15.146,51.261,67.040,96.039,16.969,11.047,51.476,7.500,206.244,178.680,17.228,135.803,21.294,424.072,16.587,219.769,42.968,5.629,99.092,43.544,169.338,33.717,62.501,115.002,25.104,313.360,7.269,331.908,50.718,5.495,84.347,11.865,45.660,111.390,157.137,46.068,130.476,130.816,18.745,9.284,210.809,69.917,38.924,27.146,41.299,109.792,322.444,33.660,149.673,25.473,20.669,23.462,87.988,17.060,53.533,53.255,7.771,29.888,78.722,4.359,12.167,14.701,80.145,51.516,196.390,106.366,188.409,119.376,5.497,9.177,21.131,114.977,164.151,217.971,413.549,122.746,313.368,84.746,156.599,265.505,144.757,16.656,15.517,62.814,71.102,101.714,99.130,576.254,8.878,471.287,312.584,201.562,139.463,68.686,120.811,58.461,44.517,88.463,293.394,81.270,139.288,523.430,133.299,133.399,24.388,49.577,23.999,36.249,28.062,16.468,461.025,235.193,375.634,40.729,9.149,269.149,149.479,349.531,114.620,34.114,3.083,3.057,21.256,43.211,123.331,8.658,195.420,96.691,177.556,324.443,79.605,662.281,25.089,142.238,38.520,142.448,41.796,211.486,44.858,27.328,10.846,25.876,46.298,379.783,20.868,132.244,7.174,49.982,146.106,10.644,55.437,65.709,54.825,11.107,686.442,273.949,425.948,293.223,50.626,110.970,14.835,380.716,773.255,189.057,149.279,121.018,15.109,6.410,8.032,193.981,74.411,184.623,616.365,781.085,705.547,487.725,583.544,229.059,123.051,346.739,142.028,327.848,814.148,42.248,85.198,47.985,62.573,25.669,36.154,129.374,86.404,61.346,502.542,41.904,74.354,32.881,317.822,292.404,58.078,268.360,49.761,26.694,139.245,258.914,58.378,65.070,68.310,160.454,109.087,101.298,103.313,10.739,144.545,81.396,130.171,592.699,81.903,360.052,145.363,19.947,87.675,285.735,161.481,15.765,100.197,53.930,29.274,44.941,39.145,9.424,16.092,83.358,3.692,71.505,33.773,9.028,5.759,16.902,206.756,6.803,43.215,5.281,4.633,41.930,176.831,5.920,3.663,59.213,5.656,9.109,56.445,34.080,58.171,27.658,7.669,125.596,162.554,64.983,4.306,33.434,9.649,32.855,30.710,21.775,4.471]

    # Create a mapping dictionary
    dcs_cms_Nls_mapping  = dict(zip(run_number, dcs_cms_Nls))
    dcs_rec_lumi_mapping = dict(zip(run_number, dcs_rec_lumi))
    run_ls_rec_mapping  = dict(zip(run_number, run_ls_rec))

    df['cms_Nls'] = df['run_number'].map(dcs_cms_Nls_mapping)
    df['rec_lumi'] = df['run_number'].map(dcs_rec_lumi_mapping)
    df['run_ls_rec'] = df['run_number'].map(run_ls_rec_mapping)

    return df

def scatter_plot_vs_LS(df):
    
    df_1d, df_2d = extract_1d_and_2d_df(df)

    # Extracting 1d chi2 metrics
    df_1d_good = df_1d[(df_1d["label"] == 0) | (df_1d["label"] == -1)]
    Chi2_scores_flat_1d, pull_scores_flat_1d, nEntries_scores_flat_1d  = chi2_and_LS(df_1d_good)
    
    df_1d_bad = df_1d[df_1d["label"] == 1]
    Chi2_scores_flat_bad_1d, pull_scores_flat_bad_1d, nEntries_scores_flat_bad_1d  = chi2_and_LS(df_1d_bad)
    
    # Now two dimensional!
    df_2d_good = df_2d[(df_2d["label"] == 0) | (df_2d["label"] == -1)]
    Chi2_scores_flat_2d, pull_scores_flat_2d, nEntries_scores_flat_2d  = chi2_and_LS(df_2d_good)
    
    df_2d_bad = df_2d[df_2d["label"] == 1]
    Chi2_scores_flat_bad_2d, pull_scores_flat_bad_2d, nEntries_scores_flat_bad_2d  = chi2_and_LS(df_2d_bad)
        
    # Now lets plot this Chi2 metrics!
    main_plotter(Chi2_scores_flat_1d, nEntries_scores_flat_1d, Chi2_scores_flat_bad_1d, nEntries_scores_flat_bad_1d, Chi2_scores_flat_2d, nEntries_scores_flat_2d, Chi2_scores_flat_bad_2d, nEntries_scores_flat_bad_2d, IsLS = True, LogScale = True)
    main_plotter(pull_scores_flat_1d, nEntries_scores_flat_1d, pull_scores_flat_bad_1d, nEntries_scores_flat_bad_1d, pull_scores_flat_2d, nEntries_scores_flat_2d, pull_scores_flat_bad_2d, nEntries_scores_flat_bad_2d, IsLS = True, LogScale = True, algorithm = 'MaxPull')

def scatter_plot_vs_nEntires(df):
    
    df_1d, df_2d = extract_1d_and_2d_df(df)

    # Extracting 1d chi2 metrics
    df_1d_good = df_1d[(df_1d["label"] == 0) | (df_1d["label"] == -1)]
    Chi2_scores_flat_1d, pull_scores_flat_1d, nEntries_scores_flat_1d  = chi2_and_nEntries(df_1d_good)
    
    df_1d_bad = df_1d[df_1d["label"] == 1]
    Chi2_scores_flat_bad_1d, pull_scores_flat_bad_1d , nEntries_scores_flat_bad_1d  = chi2_and_nEntries(df_1d_bad)
    
    # Now two dimensional!
    df_2d_good = df_2d[(df_2d["label"] == 0) | (df_2d["label"] == -1)]
    Chi2_scores_flat_2d, pull_scores_flat_2d, nEntries_scores_flat_2d  = chi2_and_nEntries(df_2d_good)
    
    df_2d_bad = df_2d[df_2d["label"] == 1]
    Chi2_scores_flat_bad_2d, pull_scores_flat_bad_2d, nEntries_scores_flat_bad_2d  = chi2_and_nEntries(df_2d_bad)
        
    # Now lets plot this Chi2 metrics!
    main_plotter(Chi2_scores_flat_1d, nEntries_scores_flat_1d, Chi2_scores_flat_bad_1d, nEntries_scores_flat_bad_1d, Chi2_scores_flat_2d, nEntries_scores_flat_2d, Chi2_scores_flat_bad_2d, nEntries_scores_flat_bad_2d, IsnEntries = True, LogScale = True)
    main_plotter(pull_scores_flat_1d, nEntries_scores_flat_1d, pull_scores_flat_bad_1d, nEntries_scores_flat_bad_1d, pull_scores_flat_2d, nEntries_scores_flat_2d, pull_scores_flat_bad_2d, nEntries_scores_flat_bad_2d, IsnEntries = True, LogScale = True,  algorithm = 'MaxPull')

def scatter_plot_vs_rec_lumi(df):
    
    df_1d, df_2d = extract_1d_and_2d_df(df)

    # Extracting 1d chi2 metrics
    df_1d_good = df_1d[(df_1d["label"] == 0) | (df_1d["label"] == -1)]
    Chi2_scores_flat_1d, pull_scores_flat_1d, nEntries_scores_flat_1d  = chi2_and_rec_lumi(df_1d_good)
    
    df_1d_bad = df_1d[df_1d["label"] == 1]
    Chi2_scores_flat_bad_1d, pull_scores_flat_bad_1d, nEntries_scores_flat_bad_1d  = chi2_and_rec_lumi(df_1d_bad)
    
    # Now two dimensional!
    df_2d_good = df_2d[(df_2d["label"] == 0) | (df_2d["label"] == -1)]
    Chi2_scores_flat_2d, pull_scores_flat_2d ,nEntries_scores_flat_2d  = chi2_and_rec_lumi(df_2d_good)
    
    df_2d_bad = df_2d[df_2d["label"] == 1]
    Chi2_scores_flat_bad_2d, pull_scores_flat_bad_2d, nEntries_scores_flat_bad_2d  = chi2_and_rec_lumi(df_2d_bad)
        
    # Now lets plot this Chi2 metrics!
    main_plotter(Chi2_scores_flat_1d, nEntries_scores_flat_1d, Chi2_scores_flat_bad_1d, nEntries_scores_flat_bad_1d, Chi2_scores_flat_2d, nEntries_scores_flat_2d, Chi2_scores_flat_bad_2d, nEntries_scores_flat_bad_2d, IsLS = False, IsLumi = True, LogScale = True)
    main_plotter(pull_scores_flat_1d, nEntries_scores_flat_1d, pull_scores_flat_bad_1d, nEntries_scores_flat_bad_1d, pull_scores_flat_2d, nEntries_scores_flat_2d, pull_scores_flat_bad_2d, nEntries_scores_flat_bad_2d, IsLS = False, IsLumi = True, LogScale = True, algorithm = 'MaxPull')

def scatter_ranked_chi2_vs_ranked_nEntries(df):
    
    # Removing runs with Chi2 scores less than 0
    df_1d, df_2d = extract_1d_and_2d_df(df)

    # Extracting 1d chi2 metrics
    df_1d_good = df_1d[(df_1d["label"] == 0) | (df_1d["label"] == -1)]
    Chi2_scores_flat_1d, pull_scores_flat_1d, nEntries_scores_flat_1d  = ranked_chi2_and_nEntries(df_1d_good)
    
    df_1d_bad = df_1d[df_1d["label"] == 1]
    Chi2_scores_flat_bad_1d, pull_scores_flat_bad_1d, nEntries_scores_flat_bad_1d  = ranked_chi2_and_nEntries(df_1d_bad)
    
    # Now two dimensional!
    df_2d_good = df_2d[(df_2d["label"] == 0) | (df_2d["label"] == -1)]
    Chi2_scores_flat_2d, pull_scores_flat_2d,nEntries_scores_flat_2d  = ranked_chi2_and_nEntries(df_2d_good)
    
    df_2d_bad = df_2d[df_2d["label"] == 1]
    Chi2_scores_flat_bad_2d, pull_scores_flat_bad_2d, nEntries_scores_flat_bad_2d  = ranked_chi2_and_nEntries(df_2d_bad)
        
    # Now lets plot this Chi2 metrics!
    main_plotter(Chi2_scores_flat_1d, nEntries_scores_flat_1d, Chi2_scores_flat_bad_1d, nEntries_scores_flat_bad_1d, Chi2_scores_flat_2d, nEntries_scores_flat_2d, Chi2_scores_flat_bad_2d, nEntries_scores_flat_bad_2d)
    main_plotter(pull_scores_flat_1d, nEntries_scores_flat_1d, pull_scores_flat_bad_1d, nEntries_scores_flat_bad_1d, pull_scores_flat_2d, nEntries_scores_flat_2d, pull_scores_flat_bad_2d, nEntries_scores_flat_bad_2d, algorithm = 'MaxPull')

def main():

    # Reading the file
    path = "../with_dimensions_in_df/l1t_bb.csv"
    read_file = pd.read_csv(path)

    # Adding the run_ls_rec and other lumi related variables
    add_lumi_related_variables(read_file)
    
    read_file = read_file[ read_file["L1T/Run summary/L1TObjects/L1TMuon/timing/First_bunch/muons_eta_phi_bx_firstbunch_0_chi2_score_betabinom"] > 0 ]
    
    # Now, producing the diferent kind of scatter plots
    scatter_plot_vs_LS(read_file)
    scatter_plot_vs_rec_lumi(read_file)
    scatter_plot_vs_nEntires(read_file)
    scatter_ranked_chi2_vs_ranked_nEntries(read_file)



if __name__ == "__main__":
    main()