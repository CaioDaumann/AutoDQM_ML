#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import uproot
import numpy as np
import scipy.stats as stats
from scipy.special import gammaln
from autodqm.rebin_utils import rebin_pull_hist

def comparators():
    return {
        'beta_binomial' : beta_binomial
    }

def beta_binomial(histpair, pull_cap=15, chi2_cut=10, pull_cut=10, min_entries=1, tol=0.01, norm_type='all', **kwargs):
    """beta_binomial works on both 1D and 2D"""
    data_hist_orig = histpair.data_hist
    ref_hists_orig = [rh for rh in histpair.ref_hists if len(rh) == len(data_hist_orig)]

    ## Observed that float64 is necessary for betabinom to preserve precision with arrays as input.
    ## (Not needed for single values.)  Deep magic that we do not undertand - AWB 2022.08.02
    data_hist_raw = np.round(np.copy(np.float64(data_hist_orig)))
    ref_hists_raw = np.round(np.array([np.copy(np.float64(rh)) for rh in ref_hists_orig]))

    ## Can't use bin centers: instead treat 'xmin' and 'xmax' as indices
    # x_bins = data_hist_orig.axes[0].edges()

    ## Concatenate multiple histograms together
    do_concat = histpair.data_concat and histpair.ref_concat
    if do_concat:
        for dhc in histpair.data_concat:
            data_hist_raw = np.concatenate((data_hist_raw, np.round(np.copy(np.float64(dhc)))))

        ref_hists_concat = []
        for ii in range(len(ref_hists_raw)):
            iRef_concat = np.copy(ref_hists_raw[ii])
            for rhc in histpair.ref_concat[ii]:
                iRef_concat = np.concatenate((iRef_concat, np.copy(np.float64(rhc))))
            ref_hists_concat.append(iRef_concat)
        ref_hists_raw = np.array(ref_hists_concat)

    ## Delete empty reference histograms
    ref_hists_raw = np.array([rhr for rhr in ref_hists_raw if np.sum(rhr) > 0])
    nRef = len(ref_hists_raw)

    ## Does not run beta_binomial if data or ref is 0
    if np.sum(data_hist_raw) <= 0 or nRef == 0:
        return np.NAN, np.NAN#None, None

    ## Adjust x-axis range for 1D plots if option set in config file
    if len(data_hist_raw) > 4 and not do_concat:
        binLo, binHi = 0, len(data_hist_raw) - 1
        if 'xmin' in histpair.config.keys() and histpair.config['xmin'] <= len(data_hist_raw) - 2:
            binLo = max(binLo, histpair.config['xmin'])
        if 'xmax' in histpair.config.keys() and histpair.config['xmax'] >= binLo + 1:
            binHi = min(binHi, histpair.config['xmax'])

        ## Check if new binning makes data or sum of references have all empty bins
        if np.sum(data_hist_raw[binLo:binHi+1]) <= 0 or sum(np.sum(r[binLo:binHi+1]) > 0 for r in ref_hists_raw) == 0:
            binLo, binHi = 0, len(data_hist_raw) - 1

        data_hist_raw = data_hist_raw[binLo:binHi+1]
        ref_hists_raw = np.array([r[binLo:binHi+1] for r in ref_hists_raw if np.sum(r[binLo:binHi+1]) > 0])

    ## Update nRef and again don't run on empty histograms
    nRef = len(ref_hists_raw)
    if nRef == 0:
        return np.NAN, np.NAN#None, None

    ## Summed ref_hist
    ref_hist_sum = ref_hists_raw.sum(axis=0)

    ## Delete leading and trailing bins of 1D plots which are all zeros
    if len(data_hist_raw) > 20 and not do_concat:
        binHi = max( min( np.nonzero(data_hist_raw + ref_hist_sum > 0)[0][-1] + 1, len(data_hist_raw) - 1 ), 20 )
        binLo = min( max( np.nonzero(data_hist_raw + ref_hist_sum > 0)[0][0] - 1, 0 ), binHi - 20 )

        data_hist_raw = data_hist_raw[binLo:binHi+1]
        ref_hists_raw = np.array([r[binLo:binHi+1] for r in ref_hists_raw])
        ref_hist_sum  = ref_hist_sum[binLo:binHi+1]

    ## num entries
    data_hist_Entries = np.sum(data_hist_raw)
    ref_hist_Entries = [np.sum(rh) for rh in ref_hists_raw]
    ref_hist_Entries_avg = np.round(np.sum(ref_hist_Entries) / nRef)

    # ## normalized ref_hist
    # ref_hist_norm = np.zeros_like(ref_hist_sum)
    # for ref_hist_raw in ref_hists_raw:
    #     ref_hist_norm = np.add(ref_hist_norm, (ref_hist_raw / np.sum(ref_hist_raw)))
    # ref_hist_norm = ref_hist_norm * data_hist_Entries / nRef

    ## only filled bins used for chi2
    nBinsUsed = np.count_nonzero(np.add(ref_hist_sum, data_hist_raw))
    nBins = data_hist_raw.size

    ## calculte pull and chi2, and get probability-weighted reference histogram
    [pull_hist, ref_hist_prob_wgt] = pull(data_hist_raw, ref_hists_raw, tol)
    pull_hist = pull_hist*np.sign(data_hist_raw-ref_hist_prob_wgt)
    chi2 = np.square(pull_hist).sum()/nBinsUsed
    max_pull = maxPullNorm(np.amax(pull_hist), nBinsUsed)
    min_pull = maxPullNorm(np.amin(pull_hist), nBinsUsed)
    if abs(min_pull) > max_pull:
        max_pull = min_pull

    # For test propouses! - should we do it only for Occupancy plots??
    # lets check if the histogram is 2d
    #if 'TH2' in str(type(data_hist_orig)):
    if pull_hist.ndim > 1:
        pass
        #chi2 = rebin_pull_hist(pull_hist, histpair, chi2, nBinsUsed)
    #if 'Occupancy' in histpair.data_name:
    #    chi2 = rebin_pull_hist(pull_hist, histpair)

    print('DEBUG: chi2 = %f, max_pull = %f\n' % (chi2, max_pull))

    return chi2, max_pull, data_hist_Entries, pull_hist.ndim


def pull(D_raw, R_list_raw, tol=0.01):
    nRef = len(R_list_raw)
    probs = []

    for R_raw in R_list_raw:
        ## Compute per-bin probabilities with beta-binomial function
        ## Protect against zero values with a floor at 10^-300 (37 sigma)
        probs.append( np.maximum(ProbRel(D_raw, R_raw, 'BetaB', tol), pow(10, -300)) )

    ## Per-bin probability is the per-bin average over all ref hists
    prob = np.array(probs).sum(axis=0) / nRef
    pull = Sigmas(prob)

    ## Reference histogram weighted by per-bin probabilities
    R_prob_wgt_avg = np.zeros_like(D_raw)

    for iR in range(len(R_list_raw)):
        R_raw = R_list_raw[iR]
        ## Get reference hist normalized to 1
        R_prob_wgt = R_raw / np.sum(R_raw)
        ## Compute per-bin probabilities relative to sum of probabilites
        prob_rel = np.divide(probs[iR], np.array(probs).sum(axis=0))
        ## Scale normalized reference by per-bin relative probabilities
        R_prob_wgt = np.multiply(R_prob_wgt, prob_rel)
        ## Add into average probability-weighted distribution
        R_prob_wgt_avg = np.add(R_prob_wgt_avg, R_prob_wgt)

    ## Normalize to data
    R_prob_wgt_avg = R_prob_wgt_avg * np.sum(D_raw)

    return [pull, R_prob_wgt_avg]

def maxPullNorm(maxPull, nBinsUsed, cutoff=pow(10,-15)):
    sign = np.sign(maxPull)
    ## sf (survival function) better than 1-cdf for large pulls (no precision error)
    probGood = stats.chi2.sf(np.power(min(abs(maxPull), 37), 2), 1)

    ## Use binomial approximation for low probs (accurate within 1%)
    if nBinsUsed * probGood < 0.01:
        probGoodNorm = nBinsUsed * probGood
    else:
        probGoodNorm = 1 - np.power(1 - probGood, nBinsUsed)

    pullNorm = Sigmas(probGoodNorm) * sign

    return pullNorm


## Mean expectation for number of expected data events
def Mean(Data, Ref, func):
    nRef = Ref.sum()
    nData = Data.sum()
    if func == 'Gaus1' or func == 'Gaus2':
        return 1.0*nData*Ref/nRef
    ## https://en.wikipedia.org/wiki/Beta-binomial_distribution#Moments_and_properties
    ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
    if func == 'BetaB' or func == 'Gamma':
        return 1.0*nData*(Ref+1)/(nRef+2)

    print('\nInside Mean, no valid func = %s. Quitting.\n' % func)
    sys.exit()
## Standard deviation of gaussian and beta-binomial functions
def StdDev(Data, Ref, func):
    nData = Data.sum()
    nRef = Ref.sum()
    mask = Ref > 0.5*nRef
    if func == 'Gaus1':
        ## whole array is calculated using the (Ref <= 0.5*nRef) formula, then the ones where the
        ## conditions are actually failed is replaced using mask with the (Ref > 0.5*nRef) formula
        output = 1.0*nData*np.sqrt(np.clip(Ref, a_min=1, a_max=None))/nRef
        output[mask] = (1.0*nData*np.sqrt(np.clip(nRef-Ref, a_min=1, a_max=None)))[mask]/nRef
    elif func == 'Gaus2':
        ## instead of calculating max(Ref, 1), set the whole array to have a lower limit of 1
        clipped = np.clip(Ref, a_min=1, a_max=None)
        output = 1.0*nData*np.sqrt( clipped/np.square(nRef) + Mean(nData, Ref, nRef, func)/np.square(nData) )
        clipped = np.clip(nRef-Ref, a_min=1, a_max=None)
        output[mask] = (1.0*nData*np.sqrt( clipped/np.square(nRef) + (nData - Mean(nData, Ref, nRef, func))/np.square(nData) ))
    elif (func == 'BetaB') or (func == 'Gamma'):
        output = 1.0*np.sqrt( nData*(Ref+1)*(nRef-Ref+1)*(nRef+2+nData) / (np.power(nRef+2, 2)*(nRef+3)) )

    else:
        print('\nInside StdDev, no valid func = %s. Quitting.\n' % func)
        sys.exit()

    return output


## Number of standard devations from the mean in any function
def numStdDev(Data, Ref, func):
    nData = Data.sum()
    nRef = Ref.sum()
    return (Data - Mean(Data, Ref, func)) / StdDev(Data, Ref, func)


## Predicted probability of observing Data / nData given a reference of Ref / nRef
def Prob(Data, nData, Ref, nRef, func, tol=0.01):
    scaleTol = np.power(1 + np.power(Ref * tol**2, 2), -0.5)
    nRef_tol = np.round(scaleTol * nRef)
    Ref_tol = np.round(Ref * scaleTol)
    nData_arr = np.zeros_like(Data) + np.float64(nData)

    if func == 'Gaus1' or func == 'Gaus2':
        return stats.norm.pdf( numStdDev(Data, Ref_tol, func) )
    if func == 'BetaB':
        ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
        ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
        return stats.betabinom.pmf(Data, nData_arr, Ref_tol + 1, nRef_tol - Ref_tol + 1)
    ## Expression for beta-binomial using definition in terms of gamma functions
    ## https://en.wikipedia.org/wiki/Beta-binomial_distribution#As_a_compound_distribution
    if func == 'Gamma':
        ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
        n_  = nData_arr
        k_  = Data
        a_  = Ref_tol + 1
        b_  = nRef_tol - Ref_tol + 1
        ab_ = nRef_tol + 2
        logProb  = gammaln(n_+1) + gammaln(k_+a_) + gammaln(n_-k_+b_) + gammaln(ab_)
        logProb -= ( gammaln(k_+1) + gammaln(n_-k_+1) + gammaln(n_+ab_) + gammaln(a_) + gammaln(b_) )
        return np.exp(logProb)

    print('\nInside Prob, no valid func = %s. Quitting.\n' % func)
    sys.exit()


## Predicted probability relative to the maximum probability (i.e. at the mean)
def ProbRel(Data, Ref, func, tol=0.01):
    nData = Data.sum()
    nRef = Ref.sum()
    ## Find the most likely expected data value
    exp_up = np.clip(np.ceil(Mean(Data, Ref, 'Gaus1')), a_min=None, a_max=nData) # make sure nothing goes above nData
    exp_down = np.clip(np.floor(Mean(Data, Ref, 'Gaus1')), a_min=0, a_max=None) # make sure nothing goes below zero

    ## Find the maximum likelihood
    maxProb_up  = Prob(exp_up, nData, Ref, nRef, func, tol)
    maxProb_down = Prob(exp_down, nData, Ref, nRef, func, tol)
    maxProb = np.maximum(maxProb_up, maxProb_down)
    thisProb = Prob(Data, nData, Ref, nRef, func, tol)

    ## Sanity check to not have relative likelihood > 1
    ratio = np.divide(thisProb, maxProb, out=np.zeros_like(thisProb), where=maxProb!=0)
    cond = thisProb > maxProb
    ratio[cond] = 1

    return ratio


## Convert relative probability to number of standard deviations in normal distribution
def Sigmas(probRel):
    ## chi2.isf function fails for probRel < 10^-323, so cap at 10^-300 (37 sigma)
    probRel = np.maximum(probRel, pow(10, -300))
    return np.sqrt(stats.chi2.isf(probRel, 1))
    ## For very low prob, can use logarithmic approximation:
    ## chi2.isf(prob, 1) = 2 * (np.log(2) - np.log(prob) - 3)
