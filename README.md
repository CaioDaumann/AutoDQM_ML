# AutoDQM_ML
[![DOI](https://zenodo.org/badge/356313006.svg)](https://zenodo.org/badge/latestdoi/356313006)

## Description
This repository contains tools relevant for training and evaluating anomaly detection algorithms on CMS DQM data.
Core code is contained in `autodqm_ml`, core scripts are contained in `scripts` and some helpful examples are in `examples`.
See the README of each subdirectory for more information on each.

## Installation
**1. Clone repository**
```
git clone https://github.com/AutoDQM/AutoDQM_ML.git 
cd AutoDQM_ML
```
**2. Install dependencies**

Dependencies are listed in ```environment.yml``` and installed using `conda`. If you do not already have `conda` set up on your system, you can install (for linux) with:
```
curl -O -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
```
You can then set `conda` to be available upon login with
```
~/miniconda3/bin/conda init # adds conda setup to your ~/.bashrc, so relogin after executing this line
```

Once `conda` is installed and set up, install dependencies with (warning: this step may take a while)
```
conda env create -f environment.yml -p <path to install conda env>
```

Some packages cannot be installed via `conda` or take too long and need to be installed with `pip` (after activating your `conda` env above):
```
pip install yahist
pip install tensorflow==2.5
```

Note: if you are running on `lxplus`, you may run into permissions errors, which may be fixed with:
```
chmod 755 -R /afs/cern.ch/user/s/<your_user_name>/.conda
```
and then rerunning the command to create the `conda` env. The resulting `conda env` can also be several GB in size, so it may also be advisable to specify the installation location in your work area if running on `lxplus`, i.e. running the `conda env create` command with `-p /afs/cern.ch/work/...`.

**3. Install autodqm-ml**

Install with:
```
pip install -e .
```

Once your setup is installed, you can activate your python environment with
```
conda activate autodqm-ml
```

**Note**: `CMSSW` environments can interfere with `conda` environments. Recommended to unset your CMSSW environment (if any) by running
```
eval `scram unsetenv -sh`
```
before attempting installation and each time before activating the `conda` environment.

## Development Guidelines

### Documentation
Please comment code following [this convention](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) from `sphinx`.

In the future, `sphinx` can be used to automatically generate documentation pages for this project.

### Logging
Logging currently uses the Python [logging facility](https://docs.python.org/3/library/logging.html) together with [rich](https://github.com/willmcgugan/rich) (for pretty printing) to provide useful information printed both to the console and a log file (optional).

Two levels of information can be printed: `INFO` and `DEBUG`. `INFO` level displays a subset of the information printed by `DEBUG` level.

A logger can be created in your script with
```
from autodqm_ml.utils import setup_logger
logger = setup_logger(<level>, <log_file>)
```
And printouts can be added to the logger with:
```
logger.info(<message>) # printed out only in INFO level
logger.debug(<message>) # printed out in both INFO and DEBUG levels
```

It is only necessary to explicit create the logger with `setup_logger` once (likely in your main script). Submodules of `autodqm_ml` should initialize loggers as:
```
import logging
logger = logging.getLogger(__name__)
```
If a logger has been created in your main script with `setup_logger`, the line `logger = logging.getLogger(__name__)` will automatically detect the existing logger and inherit its settings (print-out level and log file).

Some good rules of thumb for logging:
```
logger.info # important & succint info that user should always see
logger.debug # less important info, or info that will have many lines of print-out
logger.warning # for something that may result in unintended behavior but isn't necessarily wrong
logger.exception # for something where the user definitely made a mistake
```

### Contributing
To contribute anything beyond a minor bug fix or modifying documentation/comments, first check out a new branch:
```
git checkout -b my_new_improvement
```
Add your changes to this branch and push:
```
git push origin my_new_improvement
```
Finally, when you think it's ready to be included in the main branch create a pull request (if you push your changes from the command line, Github should give you a link that you can click to automatically do this.) 

If you think the changes you are making might benefit from discussion, create an "Issue" under the [Issues](https://github.com/AutoDQM/AutoDQM_ML/issues) tab.

## Studies of Large Data using ML

In order to obtain large data sets of SSE scores for histograms across a large number of runs (e.g. all data recorded in 2022), write up a data set config selecting the data file(s) from which to read the eos Prompt or Re-Reco files, and the set of runs of interest (with runs that are a priori known bad runs marked as such). Then select the histograms of interest using a histogram config file. Common use config files are found in the metadata directory. To fetch the data, run the command
```
python scripts/fetch_data.py --output_dir "data_fetched/pretraining" --contents "metadata/histogram_lists/myHistList.json" --datasets "metadata/dataset_lists/myDataSetList.json"
```
This may need to be run multiple times if using more than one data set e.g. Muon and SingleMuon (necessary for 2022 data) or Muon and JetMET (HLTPhysics is often a suitable replacement for these however) with a large number of (primarily 2D) histograms. The output .parquet file (named for each single data set or "allCollections" for more than one) is then fed to the training module, which is run for each algorithm to obtain a .csv file of SSE scores for all histograms and runs. These scores are calculated following training the algorithm on all the non-bad runs (as marked in the data-fetching stage), and are a Chi2-like measure of the difference between the original histogram and the histogram reconstructed by the algorithm according to the trained NN. This is done as follows:
```
python scripts/train.py --input_file "data_fetched/pretraining/myOutputFile.parquet" --output_dir "data_fetched/ae" --algorithm "autoencoder" --tag "myAutoencoder" --histograms "CSV-list-of-histos" --debug
python scripts/train.py --input_file "data_fetched/pretraining/myOutputFile.parquet" --output_dir "data_fetched/pca" --algorithm "pca" --tag "myPCA" --histograms "CSV-list-of-histos" --debug
```
Here, the full set or subset of histograms as feature in your `myHistList.json` file is entered as an argument and the data set family (e.g. L1T, or a subdetector e.g. Muon). A quick way to obtain this list is to run the command
```
python scripts/json_to_string.py -i metadata/histogram_lists/myHistList.json -d DATA_SET_FAMILY
```
FOR SMALL ORIGINAL V RECO STUDIES: If interested in using the `scripts/assess.py` macro to generate plots comparing original and reconstructed histogram distributions (i.e. the original assessment version of the repo), add the argument `--reco_assess_plots True` to the `scripts/train.py` stage to output a parquet file containing the relevant histogram information to do this. This is recommended for a subset of the runs fetched, and a subset of the histograms fetched, due to the exhaustive nature of generating the plots. A typical plotting assessment command for this would be
```
python scripts/assess.py --output_dir "assess_data_trained" --input_file "data_fetched/ae/HLTPhysics.parquet" --histograms "CSV-list-of-histos" --algorithms "myAutoencoder" --runs "35XXXX,36XXXX" --debug
```
The output CSV files from the training step are then processed to produce ROC curves, which measure the Mean number of Histogram Flags (per each algorithm) per good/bad run (the HF-ROC curve), and the Fraction of Runs with N histogram Flags (RF-ROC), where N = 1, 3, and 5 (although this is simple enough to change in the script). This can be done with the following script:
```
python scripts/sse_scores_to_roc.py --input_file "data_fetched/ae/myOutputFile_test_ae_runs_and_sse_scores.csv" --output_dir "data_fetched/assessment/"
python scripts/sse_scores_to_roc.py --input_file "data_fetched/pca/myOutputFile_test_pca_runs_and_sse_scores.csv" --output_dir "data_fetched/assessment/"
```
The end result is two plots per algorithm, one with the HF-ROC curve, and the other with the RF-ROC curve. In cases where the scores are to be combined, there is a template combiner script `scripts/combine_scores.py` which can plot output using the template `scripts/plot_merged_df.py` script.
