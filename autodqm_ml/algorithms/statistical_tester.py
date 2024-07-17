import numpy
from scipy.stats import ks_2samp
import scipy.stats as stats
import pandas as pd

from autodqm_ml.algorithms.anomaly_detection_algorithm import AnomalyDetectionAlgorithm
import plugins.beta_binomial as bb
import autodqm.histpair as hp

import logging
logger = logging.getLogger(__name__)

class StatisticalTester(AnomalyDetectionAlgorithm):
    """
    Class to perform statistical-based tests for judging anomalies.
    For 1d histograms, it will perform a ks-test.
    For 2d histograms, it will perform a pull-value test.
    """

    def __init__(self, **kwargs):
        super(StatisticalTester, self).__init__(**kwargs)
        self.reference = kwargs.get("reference", None)

    def predict(self):
        nRef = 8 ## Number of reference runs required
        sort_runs= pd.DataFrame({x:self.df[x] for x in ['run_number', 'label']})## List of all runs, with the label
        sort_runs = sort_runs.sort_values(by='run_number', ascending=False) ##sorted low to high
        ## can make a boolean df here with the conditions, then access the bollean array in the  if (xRun < self.df['run_number'][i]) and (not xRun in ref_runs) line
        ## the boolean array will either be for {good vs. bad} or { test-only vs. train}??


        ## Loop over histograms
        for histogram, info in self.histograms.items():
            ## Initialize values to -99
            score_chi2 = numpy.zeros(len(self.df)) - 99
            score_pull = numpy.zeros(len(self.df)) - 99
            score_nEntries = numpy.zeros(len(self.df)) - 99
            score_nDims = numpy.zeros(len(self.df)) - 99

            ## Loop over runs
            for i in range(len(score_chi2)):
                ref_runs  = []  ## List of reference runs for this data run
                ## Use nRef previous runs as the reference runs
                ## select only good runs, select only
                for _, sort_run in sort_runs.iterrows(): # loop over df rows
                    xRun = sort_run['run_number']
                    if (xRun < self.df['run_number'][i]) and (not xRun in ref_runs) \
                    and (sort_run['label'] == 0):
                        ref_runs.append(xRun)
                    if len(ref_runs) >= nRef:
                        break
                ## Only process if we have the right number of reference runs
                if len(ref_runs) < nRef: continue
                ## Append nRef histograms
                ref_hists = []
                for xRun in ref_runs:
                    ref_hists.append(self.df[self.df.run_number == xRun][0][histogram])

                print('\nhistogram : ', histogram)
                print('refs : ', ref_runs)
                print('data : ', self.df['run_number'][i])
                print('label : ', self.df['label'][i])
                hPair = hp.HistPair('dqmSource', {'comparators': ['beta_binomial']},
                                    'd_ser', 'd_samp', self.df['run_number'][i], 'hName', self.df[histogram][i],
                                    'r_ser', 'r_samp', ref_runs, 'hName', [rh for rh in ref_hists],
                                    False, False)

                chi2_value, pull_value, n_entries, nDims = bb.beta_binomial(hPair)
                score_chi2[i] = chi2_value
                score_pull[i] = abs(pull_value)
                score_nEntries[i] = n_entries
                score_nDims[i] = nDims
            self.add_prediction(histogram+'_chi2', score_chi2)
            self.add_prediction(histogram+'_pull', score_pull)
            self.add_prediction(histogram+'_nEntries', score_nEntries)
            self.add_prediction(histogram+'_nDims', score_nDims)

    def ks_test(self, target, reference):
        """
        Perform ks test on two 1d histograms.
        """

        score, p_value = ks_2samp(
                numpy.array(target),
                numpy.array(reference)
        )

        return score, p_value


    def pull_value_test(self, target, reference):
        """
        Perform pull value test on two 2d histograms.

        :param histogram: Histogram object
        :type histogram: autodqm_ml.data_formats.histogram.Histogram
        :param threshold: value for which to declare a histogram anomalous
        :type threshold: float
        """

        # TODO
        return 0