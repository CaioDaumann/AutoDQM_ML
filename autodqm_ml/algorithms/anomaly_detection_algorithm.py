import os
import pandas
import numpy
import awkward
import json

from autodqm_ml import utils
from autodqm_ml.data_formats.histogram import Histogram
from autodqm_ml.constants import kANOMALOUS, kGOOD

import logging
logger = logging.getLogger(__name__)



DEFAULT_COLUMNS = ["run_number", "train_label"] # columns which should always be read from input df

class AnomalyDetectionAlgorithm():
    """
    Abstract base class for any anomaly detection algorithm,
    including ks-test, pull-value test, pca, autoencoder, etc.
    :param name: name to identify this anomaly detection algorithm
    :type name: str
    """

    def __init__(self, name = "default", **kwargs):
        self.name = name

        self.data_is_loaded = False

        # These arguments will be overwritten if provided in kwargs
        self.output_dir = "output"
        self.tag = "test"
        self.histograms = {}
        self.input_file = None
        self.remove_low_stat = True

        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)


    def load_data(self, file = None, histograms = {}, train_frac = 0.5, remove_low_stat = True):
        """
        Loads data from pickle file into ML class.

        :param file: file containing data to be extracted. File output of fetch_data.py
        :type file: str
        :param histograms: names of histograms to be loaded. Must match histogram names used in fetch_data.py. Dictionary in the form {<histogram name> : {"normalize" : <bool>}}.
        :type histograms: dict. Default histograms = {}
        :param train_frac: fraction of dataset to be kept as training data. Must be between 0 and 1.
        :type train_frac: float. Default train_frac = 0.0
        :param remove_low_stat: removes runs containing histograms with low stats. Low stat threshold is 10000 events.
        :type remove_low_stat: bool. remove_low_stat = False
        """
        if self.data_is_loaded:
            return

        if file is not None:
            if self.input_file is not None:
                if not (file == self.input_file):
                    logger.warning("[AnomalyDetectionAlgorithm : load_data] Data file was previously set as '%s', but will be changed to '%s'." % (self.input_file, file))
                    self.input_file = file
            else:
                self.input_file = file

        if self.input_file is None:
            logger.exception("[AnomalyDetectionAlgorithm : load_data] No data file was provided to load_data and no data file was previously set for this instance, please specify the input data file.")
            raise ValueError()

        if not os.path.exists(self.input_file):
            self.input_file = utils.expand_path(self.input_file)

        if histograms:
            self.histograms = histograms
        self.histogram_name_map = {} # we replace "/" and spaces in input histogram names to play nicely with other packages, this map lets you go back and forth between them

        logger.debug("[AnomalyDetectionAlgorithm : load_data] Loading training data from file '%s'" % (self.input_file))

        # Load dataframe
        df = awkward.from_parquet(self.input_file)

        # Set helpful metadata
        for histogram, histogram_info in self.histograms.items():
            self.histograms[histogram]["name"] = histogram.replace("/", "").replace(" ","")
            self.histogram_name_map[self.histograms[histogram]["name"]] = histogram

            a = awkward.to_numpy(df[histogram][0])
            self.histograms[histogram]["shape"] = a.shape
            self.histograms[histogram]["n_dim"] = len(a.shape)
            self.histograms[histogram]["n_bins"] = 1
            for x in a.shape:
                self.histograms[histogram]["n_bins"] *= x

        if not "train_label" in df.fields: # don't overwrite if a train/test split was already determined
            if self.train_highest_only: #if desired, prioritize high-stat runs in train set
                histogram = next(iter(self.histograms.items()))[0]

		# Sum up all entires in histogram, order, then partition by training fraction
                logger.debug("[AnomalyDetectionAlgorithm : load_data] Assigning training/test set labels based on run stats, using histogram '%s'. For random train/test splitting, set train_highest_only to False (default) " % (histogram))
                df["train_label"] = awkward.sum(df[histogram], axis = -1)
                if self.histograms[histogram]["n_dim"] == 2:
                    df["train_label"] = awkward.sum(df["train_label"], axis = -1)
                df["train_label"] = awkward.argsort(df["train_label"])

                if train_frac > 0:
                    partition = int((1 - train_frac)*len(df))
                else:
                    logger.debug("[AnomalyDetectionAlgorithm : load_data] No Training fraction given, using 50/50 train/test split.")
                    partition = int(0.5*len(sorted_stats))

                df["train_label"] = df["train_label"] >= partition

            elif train_frac > 0:
                df["train_label"] = numpy.random.choice(2, size = len(df), p = [train_frac, 1 - train_frac]) # 0 = train, 1 = test, -1 = don't use in training or testing
                df["train_label"] = awkward.where(
                        df.label == kANOMALOUS,
                        awkward.ones_like(df.label) * -1, # set train label for anomalous events to -1 so they aren't used in training or testing sets
                        df.train_label # otherwise, keep the same test/train label as before
                )
            else:
                df["train_label"] = numpy.ones(len(df)) * 1

        # Keep only the necessary columns in dataframe
        #df = df[DEFAULT_COLUMNS + list(self.histograms.keys())]

        ## remove for now to prevent entire removal of df when run many hists
        if False:#self.remove_low_stat:
            logger.debug("[anomaly_detection_algorithm : load_data] Removing low stat runs.")
            cut = df.run_number > 0 # dummy all True cut
            for histogram, histogram_info in self.histograms.items():
                n_entries = awkward.sum(df[histogram], axis = -1)
                if histogram_info["n_dim"] == 2:
                    n_entries = awkward.sum(n_entries, axis = -1)

                if awkward.all((n_entries <= 1.000001) & (n_entries >= 0.999999)): # was already normalized in a previous train.py run which would have removed low stat bins as well, so continue
                    continue
                else:
                    cut = cut & (n_entries >= self.low_stat_threshold) # FIXME: hard-coded to 10k for now  THIS IS FIXED (Not hard coded any more)
            n_runs_pre = len(df)
            n_runs_post = awkward.sum(cut)
            logger.debug("[anomaly_detection_algorithm : load_data] Removing %d/%d runs in which one or more of the requested histograms had less than %d entries." % (n_runs_pre - n_runs_post, n_runs_pre, self.low_stat_threshold))
            df = df[cut]

        for histogram, histogram_info in self.histograms.items():
            # Normalize (if specified in histograms dict)
            if False:#"normalize" in histogram_info.keys():
                if histogram_info["normalize"]:
                    sum = awkward.sum(df[histogram], axis = -1)
                    if histogram_info["n_dim"] == 2:
                        sum = awkward.sum(sum, axis = -1)

                    logger.debug("[anomaly_detection_algorithm : load_data] Scaling all entries in histogram '%s' by the sum of total entries." % histogram)
                    df[histogram] = df[histogram] * (1. / sum)
        self.n_train = awkward.sum(df.train_label == 0)
        self.n_test = awkward.sum(df.train_label == 1)
        self.df = df
        self.n_histograms = len(list(self.histograms.keys()))

        logger.debug("[AnomalyDetectionAlgorithm : load_data] Loaded data for %d histograms with %d events in training set and %d events in testing set." % (self.n_histograms, self.n_train, self.n_test))

        self.data_is_loaded = True


    def add_prediction(self, histogram, score, reconstructed_hist = None):
        """
        Add fields to the df containing the score for this algorithm (p-value/pull-value for statistical tests, sse for ML algorithms)
        and the reconstructed histograms (for ML algorithms only).
        """
        self.df[histogram + "_score_" + self.tag] = score
        if reconstructed_hist is not None:
            self.df[histogram + "_reco_" + self.tag] = reconstructed_hist


    def save(self):
        """

        """
        os.system("mkdir -p %s" % self.output_dir)

        self.output_file = "%s/%s.parquet" % (self.output_dir, self.input_file.split("/")[-1].replace(".parquet", ""))
        logger.info("[AnomalyDetectionAlgorithm : save] Saving output with additional fields to file '%s'." % (self.output_file))
        awkward.to_parquet(self.df, self.output_file)

        self.config_file = "%s/%s_%s.json" % (self.output_dir, self.name, self.tag)
        config = {}
        for k,v in vars(self).items():
            if utils.is_json_serializable(v):
                config[k] = v

        logger.info("[AnomalyDetectionAlgorithm : save] Saving AnomalyDetectionAlgorithm config to file '%s'." % (self.config_file))
        with open(self.config_file, "w") as f_out:
            json.dump(config, f_out, sort_keys = True, indent = 4)