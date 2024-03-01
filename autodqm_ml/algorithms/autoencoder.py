import os
import pandas
import numpy
import json
import awkward
import copy

import logging
logger = logging.getLogger(__name__)

import tensorflow as tf
import tensorflow.keras as keras

from autodqm_ml.algorithms.ml_algorithm import MLAlgorithm
from autodqm_ml import utils

DEFAULT_OPT = {
        "batch_size" : 128,
        "val_batch_size" : 1024,
        "learning_rate" : 0.0005, # 0.001
        "n_epochs" : 1000,
        "early_stopping" : True,
        "early_stopping_rounds" : 3,
        "n_hidden_layers" : 2,
        "n_nodes" : 50,
        "n_components" : 20, # 3
        "kernel_1d" : 3,
        "kernel_2d" : 3,
        "strides_1d" : 1,
        "strides_2d" : 1,
        "dropout" : 0.0,
        "batch_norm" : False,
        "n_filters" : 12
}

class AutoEncoder(MLAlgorithm):
    """
    Autoencoder base class.

    :param config: dictionary with hyperparameters for autoencoder training. Any hyperparameters not specified will be taken from the default values in `DEFAULT_OPT`
    :type config: dict
    :param mode: string to specify whether you want to train an autoencoder for each histogram ("individual") or a single autoencoder on all histograms ("simultaneous")
    :type mode: str
    """
    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)

        self.config = utils.update_dict(
                original = DEFAULT_OPT,
                new = kwargs.get('config', {})
        )

        self.mode = kwargs.get('autoencoder_mode', 'individual')
        if self.mode is None:
            self.mode = "individual"

        if not self.mode in ["individual", "simultaneous"]:
            logger.exception("[AutoEncoder : __init__] mode '%s' is not a recognized option for AutoEncoder. Currently available modes are 'individual' (default) and 'simultaneous'." % (self.mode))
            raise ValueError()
        self.models = {}

        logger.debug("[AutoEncoder : __init__] Constructing AutoEncoder with the following training options and hyperparameters:")
        for param, value in self.config.items():
            logger.debug("\t %s : %s" % (param, str(value)))


    def load_model(self, model_file):
        """

        """
        model = keras.models.load_model(model_file)
        return model


    def save_model(self, model, model_file):
        """

        """
        logger.debug("[AutoEncoder : save_model] Saving trained autoencoder to file '%s'." % (model_file))
        model.save(model_file)


    def train(self):
        """

        """
        if self.mode == "simultaneous":
            self.models = { None : None }
            logger.debug("[AutoEncoder : train] Mode selected as 'simultaneous', meaning a single autoencoder will be trained simultaneously on all histograms. Use 'individual' if you wish to train one autoencoder for each histogram.")
        elif self.mode == "individual":
            self.models = { k : None for k,v in self.histograms.items() } #copy.deepcopy(self.histograms)
            logger.debug("[AutoEncoder : train] Mode selected as 'individual', meaning one autoencoder will be trained for each histogram. Use 'simultaneous' if you wish to train a single autoencoder for all histograms.")

        for histogram, histogram_info in self.models.items():
            if histogram is None:
                model_file = "%s/autoencoder_%s.h5" % (self.output_dir, self.tag)
            else:
                model_file = "%s/autoencoder_%s_%s.h5" % (self.output_dir, histogram, self.tag)
            
            if os.path.exists(model_file):
                logger.warning("[AutoEncoder : train] A trained AutoEncoder already exists with tag '%s' at file '%s'. We will load the saved model from the file rather than retraining. If you wish to retrain please provide a new tag or delete the old outputs." % (self.tag, model_file))
                self.models[histogram] = self.load_model(model_file)
                return


            inputs, outputs = self.make_inputs(split = "train", histogram_name = histogram)
            inputs_val, outputs_val = self.make_inputs(split = "test", histogram_name = histogram)

            if histogram is None:
                hist_name = str(list(self.models.keys()))
            else:
                hist_name = histogram
            logger.debug("[AutoEncoder : train] Training autoencoder with %d dimensions in latent space for histogram(s) '%s' with %d training examples." % (self.config["n_components"], hist_name, len(list(inputs.values())[0]))) 

            #print(self.histograms)

            if self.mode == "simultaneous":
                histograms = self.histograms
            elif self.mode == "individual":
                histograms = { histogram : self.histograms[histogram] }

            model = AutoEncoder_DNN(histograms, **self.config).model()
           
            model.compile(
                    optimizer = keras.optimizers.Adam(learning_rate = self.config["learning_rate"]), 
                    loss = keras.losses.MeanSquaredError()
            )

            callbacks = []
            if self.config["early_stopping"]:
                callbacks.append(keras.callbacks.EarlyStopping(patience = self.config["early_stopping_rounds"]))
            #print(self.config["n_components"])
            #print(list(inputs.values()).shape)

            model.fit(
                    inputs,
                    outputs,
                    validation_data = (inputs_val, outputs_val),
                    callbacks = callbacks, 
                    epochs = self.config["n_epochs"],
                    batch_size = self.config["batch_size"]
            )
            
            self.save_model(model, model_file)
            self.models[histogram] = model

    
    def predict(self, batch_size = 1024):
        for histogram, model in self.models.items():
            inputs, outputs = self.make_inputs(split = "all", histogram_name = histogram)
            predictions = model.predict(inputs, batch_size = batch_size)

            if self.mode == "simultaneous" and self.n_histograms >= 2:
                predictions = { name : pred for name, pred in zip(model.output_names, predictions) }
            else:
                predictions = { model.output_names[0] : predictions }

            for name, pred in predictions.items():
                hist_name = self.histogram_name_map[name.replace("output_", "")] # shape [n_runs, histogram dimensions, 1]
                original_hist = self.df[hist_name] # shape [n_runs, histogram dimensions]
                reconstructed_hist = awkward.flatten( # change shape from [n_runs, histogram dimensions, 1] -> [n_runs, histogram dimensions]
                        awkward.from_numpy(pred),
                        axis = -1 
                )

                sse = awkward.sum( # perform sum along inner-most axis, i.e. first histogram dimension
                        (original_hist - reconstructed_hist) ** 2,
                        axis = -1
                )
                
                # For 2d histograms, we need to sum over one more axis to get a single SSE score for each run
                if self.histograms[hist_name]["n_dim"] == 2:
                    sse = awkward.sum(sse, axis = -1) # second histogram dimension

                self.add_prediction(hist_name, sse, reconstructed_hist) 
 

    def make_inputs(self, split = None, histogram_name = None):
        """

        """
        inputs = {}
        outputs = {}

        if split == "train":
            cut = self.df.label == 0
        elif split == "test":
            cut = self.df.label == 1
        else:
            cut = self.df.run_number >= 0 # dummy all True cut

        df = self.df[cut]

        for histogram, info in self.histograms.items():
            if histogram_name is not None: # self.mode == "individual", i.e. separate autoencoder for each histogram
                if not histogram == histogram_name: # only grab the relevant histogram for this autoencoder
                    continue

            data = tf.convert_to_tensor(df[histogram])
            inputs["input_" + info["name"]] = data
            outputs["output_" + info["name"]] = data

        

        return inputs, outputs


class AutoEncoder_DNN():
    """
    An AutoEncoder instance owns AutoEncoder_DNN(s), which is the actual implementation of the DNN.
    """
    def __init__(self, histograms, **kwargs): 
        self.n_histograms = len(histograms.keys())

        self.__dict__.update(kwargs)

        self.inputs = []
        self.outputs = []
        self.encoders = []
        self.decoders = []
        for histogram, info in histograms.items(): 
            input, encoder = self.build_encoder(histogram, info)
            self.inputs.append(input)
            self.encoders.append(encoder)

        encoder_outputs = [x for x in self.encoders]

        if self.n_histograms > 1:
            encoders_merged = keras.layers.concatenate(encoder_outputs)
        else:
            encoders_merged = encoder_outputs[0]

        layer = encoders_merged
        for i in range(self.n_hidden_layers):
            layer = keras.layers.Dense(
                    units = self.n_nodes,
                    activation = "relu",
                    name = "hidden_%d" % i,
            )(layer)

        latent_representation = keras.layers.Dense(
                units = self.n_components,
                activation = None,
                name = "latent_representation"
        )(layer)
        
        for histogram, info in histograms.items(): 
            output = self.build_decoder(histogram, info, latent_representation)
            self.outputs.append(output)


    def model(self):
        model = keras.models.Model(
                inputs = self.inputs,
                outputs = self.outputs,
                name = "autoencoder"
        )
        model.summary()
        return model


    def build_encoder(self, histogram, info):
        input = keras.layers.Input(
                shape = info["shape"] + (1,), 
                name = "input_%s" % info["name"]
        )

        layer = input
        for i in range(self.n_hidden_layers):
            name = "encoder_%d_%s" % (i, info["name"])
            if info["n_dim"] == 1:
                layer = keras.layers.Conv1D(
                        filters = self.n_filters,
                        kernel_size = self.kernel_1d,
                        strides = 1,
                        activation = "relu",
                        name = name 
                )(layer)
            elif info["n_dim"] == 2:
                layer = keras.layers.Conv2D(
                        filters = self.n_filters,
                        kernel_size = self.kernel_2d,
                        strides = 1,
                        activation = "relu",
                        name = name
                )(layer)
            if self.batch_norm:
                layer = keras.layers.BatchNormalization(name = name + "_batch_norm")
            if self.dropout > 0:
                layer = keras.layers.Dropout(self.dropout, name = name + "_dropout")


        encoder = keras.layers.Flatten()(layer)
        return input, encoder


    def build_decoder(self, histogram, info, input):
        n_output_units = info["n_bins"] * self.n_filters 
        layer = keras.layers.Dense(
                units = n_output_units,
                activation = "relu",
                name = "decoder_input_%s" % info["name"] 
        )(input)
        target_shape = info["shape"] + (self.n_filters,) 
        layer = keras.layers.Reshape(target_shape = target_shape)(layer)

        for i in range(self.n_hidden_layers):
            if i == (self.n_hidden_layers - 1):
                activation = "relu"
                n_filters = 1
                name = "output_%s" % (info["name"])
                batch_norm = False
                dropout = 0
            else:
                activation = "relu"
                n_filters = self.n_filters 
                name = "decoder_%d_%s" % (i, info["name"])
                batch_norm = self.batch_norm
                dropout = self.dropout

            if info["n_dim"] == 1:
                layer = keras.layers.Conv1DTranspose(
                        filters = n_filters,
                        kernel_size = self.kernel_1d,
                        strides = 1,
                        padding = "same",            
                        activation = activation,
                        name = name 
                )(layer)
            elif info["n_dim"] == 2:
                layer = keras.layers.Conv2DTranspose(
                        filters = n_filters, 
                        kernel_size = self.kernel_2d,
                        strides = 1,
                        padding = "same", 
                        activation = activation,
                        name = name
                )(layer)
            if batch_norm:
                layer = keras.layers.BatchNormalization(name = name + "_batch_norm")
            if dropout > 0:
                layer = keras.layers.Dropout(self.dropout, name = name + "_dropout") 

        output = layer
        return output
