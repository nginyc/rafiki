from __future__ import absolute_import, division, print_function

import os
import sys
import time

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.client import device_lib
from tensorflow.python.framework.ops import Tensor, Operation
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

import numpy as np
from datetime import datetime
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
import itertools
import tempfile
import base64

from ds_ctcdecoder import ctc_beam_search_decoder_batch, ctc_beam_search_decoder, Scorer

from rafiki.model import BaseModel, FixedKnob, FloatKnob, CategoricalKnob, dataset_utils, logger, test_model_class, InvalidModelParamsException
# IntegerKnob

from rafiki.constants import TaskType, ModelDependency
from rafiki.utils.text import Alphabet


class ConfigSingleton:
    _config = None

    def __getattr__(self, name):
        if not ConfigSingleton._config:
            raise RuntimeError("Global configuration not yet initialized.")
        if not hasattr(ConfigSingleton._config, name):
            raise RuntimeError("Configuration option {} not found in config.".format(name))
        return ConfigSingleton._config[name]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


FLAGS = tf.app.flags.FLAGS


class TfDeepSpeech(BaseModel):
    '''
    Implements a speech recognition neural network model developed by Baidu. It contains five hiddlen layers.
    Validation set not implemented
    '''
    @staticmethod
    def get_knob_config():
        return {
            'epochs': FixedKnob(1),
            'learning_rate': FloatKnob(1e-5, 1e-1, is_exp=True),
            'batch_size': CategoricalKnob([1]),
        }

    @staticmethod
    def create_flags():
        # Importer
        # ========

        f = tf.app.flags

        f.DEFINE_string('train_files', '',
                        'comma separated list of files specifying the dataset used for training. Multiple files will get merged. If empty, training will not be run.')
        f.DEFINE_string('dev_files', '',
                        'comma separated list of files specifying the dataset used for validation. Multiple files will get merged. If empty, validation will not be run.')
        f.DEFINE_string('test_files', '',
                        'comma separated list of files specifying the dataset used for testing. Multiple files will get merged. If empty, the model will not be tested.')

        f.DEFINE_string('feature_cache', '',
                        'path where cached features extracted from --train_files will be saved. If empty, caching will be done in memory and no files will be written.')

        f.DEFINE_integer('feature_win_len', 32, 'feature extraction audio window length in milliseconds')
        f.DEFINE_integer('feature_win_step', 20, 'feature extraction window step length in milliseconds')
        f.DEFINE_integer('audio_sample_rate', 16000, 'sample rate value expected by model')

        # Global Constants
        # ================

        f.DEFINE_integer('epochs', 75, 'how many epochs (complete runs through the train files) to train for')

        f.DEFINE_float('dropout_rate', 0.05, 'dropout rate for feedforward layers')
        f.DEFINE_float('dropout_rate2', -1.0, 'dropout rate for layer 2 - defaults to dropout_rate')
        f.DEFINE_float('dropout_rate3', -1.0, 'dropout rate for layer 3 - defaults to dropout_rate')
        f.DEFINE_float('dropout_rate4', 0.0, 'dropout rate for layer 4 - defaults to 0.0')
        f.DEFINE_float('dropout_rate5', 0.0, 'dropout rate for layer 5 - defaults to 0.0')
        f.DEFINE_float('dropout_rate6', -1.0, 'dropout rate for layer 6 - defaults to dropout_rate')

        f.DEFINE_float('relu_clip', 20.0, 'ReLU clipping value for non-recurrent layers')

        # Adam optimizer(http://arxiv.org/abs/1412.6980) parameters

        f.DEFINE_float('beta1', 0.9, 'beta 1 parameter of Adam optimizer')
        f.DEFINE_float('beta2', 0.999, 'beta 2 parameter of Adam optimizer')
        f.DEFINE_float('epsilon', 1e-8, 'epsilon parameter of Adam optimizer')
        f.DEFINE_float('learning_rate', 0.001, 'learning rate of Adam optimizer')

        # Batch sizes

        f.DEFINE_integer('train_batch_size', 1, 'number of elements in a training batch')
        f.DEFINE_integer('dev_batch_size', 1, 'number of elements in a validation batch')
        f.DEFINE_integer('test_batch_size', 1, 'number of elements in a test batch')

        f.DEFINE_integer('export_batch_size', 1, 'number of elements per batch on the exported graph')

        # Performance(UNSUPPORTED)
        f.DEFINE_integer('inter_op_parallelism_threads', 0,
                         'number of inter-op parallelism threads - see tf.ConfigProto for more details')
        f.DEFINE_integer('intra_op_parallelism_threads', 0,
                         'number of intra-op parallelism threads - see tf.ConfigProto for more details')

        # Sample limits

        f.DEFINE_integer('limit_train', 0, 'maximum number of elements to use from train set - 0 means no limit')
        f.DEFINE_integer('limit_dev', 0, 'maximum number of elements to use from validation set- 0 means no limit')
        f.DEFINE_integer('limit_test', 0, 'maximum number of elements to use from test set- 0 means no limit')

        # Checkpointing

        f.DEFINE_string('checkpoint_dir', '',
                        'directory in which checkpoints are stored - defaults to directory "/tmp/deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
        f.DEFINE_integer('checkpoint_secs', 600, 'checkpoint saving interval in seconds')
        f.DEFINE_integer('max_to_keep', 3, 'number of checkpoint files to keep - default value is 5')
        f.DEFINE_string('load', 'auto',
                        '"last" for loading most recent epoch checkpoint, "best" for loading best validated checkpoint, "init" for initializing a fresh model, "auto" for trying the other options in order last > best > init')

        # Exporting

        f.DEFINE_integer('export_version', 1, 'version number of the exported model')
        f.DEFINE_boolean('export_tflite', False, 'export a graph ready for TF Lite engine')
        f.DEFINE_boolean('use_seq_length', True,
                         'have sequence_length in the exported graph(will make tfcompile unhappy)')
        f.DEFINE_integer('n_steps', 16,
                         'how many timesteps to process at once by the export graph, higher values mean more latency')

        # Reporting

        f.DEFINE_integer('log_level', 1, 'log level for console logs - 0: INFO, 1: WARN, 2: ERROR, 3: FATAL')
        f.DEFINE_boolean('show_progressbar', True,
                         'Show progress for training, validation and testing processes. Log level should be > 0.')

        f.DEFINE_boolean('log_placement', False, 'whether to log device placement of the operators to the console')
        f.DEFINE_integer('report_count', 10,
                         'number of phrases with lowest WER(best matching) to print out during a WER report')

        f.DEFINE_string('summary_dir', '',
                        'target directory for TensorBoard summaries - defaults to directory "deepspeech/summaries" within user\'s data home specified by the XDG Base Directory Specification')

        # Geometry

        f.DEFINE_integer('n_hidden', 2048, 'layer width to use when initialising layers')

        # Initialization

        f.DEFINE_integer('random_seed', 4568, 'default random seed that is used to initialize variables')

        # Early Stopping

        f.DEFINE_boolean('early_stop', True,
                         'enable early stopping mechanism over validation dataset. If validation is not being run, early stopping is disabled.')
        f.DEFINE_integer('es_steps', 4,
                         'number of validations to consider for early stopping. Loss is not stored in the checkpoint so when checkpoint is revived it starts the loss calculation from start at that point')
        f.DEFINE_float('es_mean_th', 0.5,
                       'mean threshold for loss to determine the condition if early stopping is required')
        f.DEFINE_float('es_std_th', 0.5,
                       'standard deviation threshold for loss to determine the condition if early stopping is required')

        # Decoder

        f.DEFINE_string('alphabet_config_path', 'data/alphabet.txt',
                        'path to the configuration file specifying the alphabet used by the network. See the comment in data/alphabet.txt for a description of the format.')
        f.DEFINE_string('lm_binary_path', 'data/lm.binary',
                        'path to the language model binary file created with KenLM')
        f.DEFINE_string('lm_trie_path', 'data/trie',
                        'path to the language model trie file created with native_client/generate_trie')
        f.DEFINE_integer('beam_width', 1024,
                         'beam width used in the CTC decoder when building candidate transcriptions')
        f.DEFINE_float('lm_alpha', 0.75, 'the alpha hyperparameter of the CTC decoder. Language Model weight.')
        f.DEFINE_float('lm_beta', 1.85, 'the beta hyperparameter of the CTC decoder. Word insertion weight.')

        # Inference mode

        f.DEFINE_string('one_shot_infer', '',
                        'one-shot inference mode: specify a wav file and the script will load the checkpoint and perform inference on it.')

    @staticmethod
    def initialize_globals():

        def get_available_gpus():
            r"""
            Returns the number of GPUs available on this system.
            """
            local_device_protos = device_lib.list_local_devices()
            return [x.name for x in local_device_protos if x.device_type == 'GPU']

        c = AttrDict()

        # CPU device
        c.cpu_device = '/cpu:0'

        # Available GPU devices
        c.available_devices = get_available_gpus()

        # If there is no GPU available, we fall back to CPU based operation
        if not c.available_devices:
            c.available_devices = [c.cpu_device]

        # Set default dropout rates
        if FLAGS.dropout_rate2 < 0:
            FLAGS.dropout_rate2 = FLAGS.dropout_rate
        if FLAGS.dropout_rate3 < 0:
            FLAGS.dropout_rate3 = FLAGS.dropout_rate
        if FLAGS.dropout_rate6 < 0:
            FLAGS.dropout_rate6 = FLAGS.dropout_rate

        # Set default checkpoint dir
        if not FLAGS.checkpoint_dir:
            FLAGS.checkpoint_dir = '/tmp/deepspeech/checkpoints'

        if not os.path.isdir(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)

        if FLAGS.load not in ['last', 'best', 'init', 'auto']:
            FLAGS.load = 'auto'

        c.alphabet = Alphabet(os.path.abspath(os.path.abspath('examples/datasets/speech_recognition/alphabet.txt')))

        c.n_input = 26

        # The number of frames in the context
        c.n_context = 9

        # Number of units in hidden layers
        c.n_hidden = FLAGS.n_hidden

        c.n_hidden_1 = c.n_hidden

        c.n_hidden_2 = c.n_hidden

        c.n_hidden_5 = c.n_hidden

        # LSTM cell state dimension
        c.n_cell_dim = c.n_hidden

        # The number of units in the third layer, which feeds in to the LSTM
        c.n_hidden_3 = c.n_cell_dim

        # Units in the sixth layer = number of characters in the target language plus one
        c.n_hidden_6 = c.alphabet.size() + 1  # +1 for CTC blank label

        # Size of audio window in samples
        c.audio_window_samples = FLAGS.audio_sample_rate * (FLAGS.feature_win_len / 1000)

        # Stride for feature computations in samples
        c.audio_step_samples = FLAGS.audio_sample_rate * (FLAGS.feature_win_step / 1000)

        ConfigSingleton._config = c  # pylint: disable=protected-access

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        # self._graph = tf.Graph()
        self._sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                inter_op_parallelism_threads=0, intra_op_parallelism_threads=0)
        self._sess_config.gpu_options.allow_growth = True
        # self._sess = tf.Session(graph=self._graph, config=self._sess_config)
        # self._sess = tf.Session( config=self._sess_config)

        self.create_flags()
        self.f = tf.app.flags.FLAGS
        self.c = ConfigSingleton()
        self.initialize_globals()

    def samples_to_mfccs(self, samples, sample_rate):
        Config = self.c
        spectrogram = contrib_audio.audio_spectrogram(samples,
                                                      window_size=Config.audio_window_samples,
                                                      stride=Config.audio_step_samples,
                                                      magnitude_squared=True)
        mfccs = contrib_audio.mfcc(spectrogram, sample_rate, dct_coefficient_count=Config.n_input)
        mfccs = tf.reshape(mfccs, [-1, self.c.n_input])

        return mfccs, tf.shape(mfccs)[0]

    def audiofile_to_features(self, wav_filename):
        samples = tf.read_file(wav_filename)
        decoded = contrib_audio.decode_wav(samples, desired_channels=1)
        features, features_len = self.samples_to_mfccs(decoded.audio, decoded.sample_rate)

        return features, features_len

    def create_dataset(self, dataset_uri, dataset_dir, cache_path=''):
        def to_sparse_tuple(sequence):
            r"""Creates a sparse representention of ``sequence``.
                Returns a tuple with (indices, values, shape)
            """
            indices = np.asarray(list(zip([0] * len(sequence), range(len(sequence)))), dtype=np.int64)
            shape = np.asarray([1, len(sequence)], dtype=np.int64)
            return indices, sequence, shape

        def generate_values():
            for _, row in df.iterrows():
                yield os.path.join(dataset_dir.name, row.wav_filename), to_sparse_tuple(row.transcript)

        # Batching a dataset of 2D SparseTensors creates 3D batches, which fail
        # when passed to tf.nn.ctc_loss, so we reshape them to remove the extra
        # dimension here.
        def sparse_reshape(sparse):
            shape = sparse.dense_shape
            return tf.sparse.reshape(sparse, [shape[0], shape[2]])

        def batch_fn(features, features_len, transcripts):
            features = tf.data.Dataset.zip((features, features_len))
            features = features.padded_batch(batch_size,
                                             padded_shapes=([None, Config.n_input], []))
            transcripts = transcripts.batch(batch_size).map(sparse_reshape)
            return tf.data.Dataset.zip((features, transcripts))

        def entry_to_features(wav_filename, transcript):
            # https://bugs.python.org/issue32117
            features, features_len = self.audiofile_to_features(wav_filename)
            return features, features_len, tf.SparseTensor(*transcript)

        batch_size = self._knobs.get('batch_size')
        Config = self.c

        dataset = dataset_utils.load_dataset_of_audio_files(dataset_uri, dataset_dir)
        df = dataset.df

        num_gpus = len(Config.available_devices)

        dataset = (tf.data.Dataset.from_generator(generate_values,
                                                  output_types=(tf.string, (tf.int64, tf.int32, tf.int64)))
                   .map(entry_to_features)
                   .cache(cache_path)
                   .window(batch_size, drop_remainder=True).flat_map(batch_fn)
                   .prefetch(num_gpus))

        return dataset

    def create_optimizer(self):

        # Adam Optimization
        # =================

        # In contrast to 'Deep Speech: Scaling up end-to-end speech recognition'
        # (http://arxiv.org/abs/1412.5567),
        # in which 'Nesterov's Accelerated Gradient Descent'
        # (www.cs.toronto.edu/~fritz/absps/momentum.pdf) was used,
        # we will use the Adam method for optimization (http://arxiv.org/abs/1412.6980),
        # because, generally, it requires less fine-tuning.

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                           beta1=FLAGS.beta1,
                                           beta2=FLAGS.beta2,
                                           epsilon=FLAGS.epsilon)
        return optimizer

    def variable_on_cpu(self, name, shape, initializer):
        r"""
        used to create a variable in CPU memory.
        """
        # Use the /cpu:0 device for scoped operations
        with tf.device(self.c.cpu_device):
            # Create or get apropos variable
            var = tf.get_variable(name=name, shape=shape, initializer=initializer)
        return var

    def rnn_impl_lstmblockfusedcell(self, x, seq_length, previous_state, reuse):
        # Forward direction cell
        fw_cell = tf.contrib.rnn.LSTMBlockFusedCell(self.c.n_cell_dim, reuse=reuse)

        output, output_state = fw_cell(inputs=x,
                                       dtype=tf.float32,
                                       sequence_length=seq_length,
                                       initial_state=previous_state)

        return output, output_state

    def create_overlapping_windows(self, batch_x):
        Config = self.c
        batch_size = tf.shape(batch_x)[0]
        window_width = 2 * Config.n_context + 1
        num_channels = Config.n_input

        # Create a constant convolution filter using an identity matrix, so that the
        # convolution returns patches of the input tensor as is, and we can create
        # overlapping windows over the MFCCs.
        eye_filter = tf.constant(np.eye(window_width * num_channels)
                                 .reshape(window_width, num_channels, window_width * num_channels),
                                 tf.float32)  # pylint: disable=bad-continuation

        # Create overlapping windows
        batch_x = tf.nn.conv1d(batch_x, eye_filter, stride=1, padding='SAME')

        # Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
        batch_x = tf.reshape(batch_x, [batch_size, -1, window_width, num_channels])

        return batch_x

    def create_model(self, batch_x, seq_length, dropout, reuse=False, previous_state=None, overlap=True,
                     rnn_impl=None):

        def dense(name, x, units, dropout_rate=None, relu=True):
            with tf.variable_scope(name):
                bias = self.variable_on_cpu('bias', [units], tf.zeros_initializer())
                weights = self.variable_on_cpu('weights', [x.shape[-1], units], tf.contrib.layers.xavier_initializer())

            output = tf.nn.bias_add(tf.matmul(x, weights), bias)

            if relu:
                output = tf.minimum(tf.nn.relu(output), FLAGS.relu_clip)

            if dropout_rate is not None:
                output = tf.nn.dropout(output, keep_prob=1-dropout_rate)

            return output

        Config = self.c
        layers = {}

        # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
        batch_size = tf.shape(batch_x)[0]

        # Create overlapping feature windows if needed
        if overlap:
            batch_x = self.create_overlapping_windows(batch_x)

        # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
        # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

        # Permute n_steps and batch_size
        batch_x = tf.transpose(batch_x, [1, 0, 2, 3])
        # Reshape to prepare input for first layer
        batch_x = tf.reshape(batch_x, [-1,
                                       Config.n_input + 2 * Config.n_input * Config.n_context])  # (n_steps*batch_size, n_input + 2*n_input*n_context)
        layers['input_reshaped'] = batch_x

        # The next three blocks will pass `batch_x` through three hidden layers with
        # clipped RELU activation and dropout.
        layers['layer_1'] = layer_1 = dense('layer_1', batch_x, Config.n_hidden_1, dropout_rate=dropout[0])
        layers['layer_2'] = layer_2 = dense('layer_2', layer_1, Config.n_hidden_2, dropout_rate=dropout[1])
        layers['layer_3'] = layer_3 = dense('layer_3', layer_2, Config.n_hidden_3, dropout_rate=dropout[2])

        # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
        # as the LSTM RNN expects its input to be of shape `[max_time, batch_size, input_size]`.
        layer_3 = tf.reshape(layer_3, [-1, batch_size, Config.n_hidden_3])

        # Run through parametrized RNN implementation, as we use different RNNs
        # for training and inference
        rnn_impl = self.rnn_impl_lstmblockfusedcell
        output, output_state = rnn_impl(layer_3, seq_length, previous_state, reuse)

        # Reshape output from a tensor of shape [n_steps, batch_size, n_cell_dim]
        # to a tensor of shape [n_steps*batch_size, n_cell_dim]
        output = tf.reshape(output, [-1, Config.n_cell_dim])
        layers['rnn_output'] = output
        layers['rnn_output_state'] = output_state

        # Now we feed `output` to the fifth hidden layer with clipped RELU activation
        layers['layer_5'] = layer_5 = dense('layer_5', output, Config.n_hidden_5, dropout_rate=dropout[5])

        # Now we apply a final linear layer creating `n_classes` dimensional vectors, the logits.
        layers['layer_6'] = layer_6 = dense('layer_6', layer_5, Config.n_hidden_6, relu=False)

        # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
        # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
        # Note, that this differs from the input in that it is time-major.
        layer_6 = tf.reshape(layer_6, [-1, batch_size, Config.n_hidden_6], name='raw_logits')
        layers['raw_logits'] = layer_6

        # Output shape: [n_steps, batch_size, n_hidden_6]
        return layer_6, layers

    def calculate_mean_edit_distance_and_loss(self, iterator, dropout, reuse):

        # Accuracy and Loss
        # =================

        # In accord with 'Deep Speech: Scaling up end-to-end speech recognition'
        # (http://arxiv.org/abs/1412.5567),
        # the loss function used by our network should be the CTC loss function
        # (http://www.cs.toronto.edu/~graves/preprint.pdf).
        # Conveniently, this loss function is implemented in TensorFlow.
        # Thus, we can simply make use of this implementation to define our loss.

        r'''
        This routine beam search decodes a mini-batch and calculates the loss and mean edit distance.
        Next to total and average loss it returns the mean edit distance,
        the decoded result and the batch's original Y.
        '''
        # Obtain the next batch of data
        (batch_x, batch_seq_len), batch_y = iterator.get_next()

        # Calculate the logits of the batch
        logits, _ = self.create_model(batch_x, batch_seq_len, dropout, reuse=reuse)

        # Compute the CTC loss using TensorFlow's `ctc_loss`
        total_loss = tf.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)

        # Calculate the average loss across the batch
        avg_loss = tf.reduce_mean(total_loss)

        # Finally we return the average loss
        return avg_loss

    def get_tower_results(self, iterator, optimizer, dropout_rates):

        # Towers
        # ======

        # In order to properly make use of multiple GPU's, one must introduce new abstractions,
        # not present when using a single GPU, that facilitate the multi-GPU use case.
        # In particular, one must introduce a means to isolate the inference and gradient
        # calculations on the various GPU's.
        # The abstraction we intoduce for this purpose is called a 'tower'.
        # A tower is specified by two properties:
        # * **Scope** - A scope, as provided by `tf.name_scope()`,
        # is a means to isolate the operations within a tower.
        # For example, all operations within 'tower 0' could have their name prefixed with `tower_0/`.
        # * **Device** - A hardware device, as provided by `tf.device()`,
        # on which all operations within the tower execute.
        # For example, all operations of 'tower 0' could execute on the first GPU `tf.device('/gpu:0')`

        r'''
        With this preliminary step out of the way, we can for each GPU introduce a
        tower for which's batch we calculate and return the optimization gradients
        and the average loss across towers.
        '''

        Config = self.c

        # To calculate the mean of the losses
        tower_avg_losses = []

        # Tower gradients to return
        tower_gradients = []

        with tf.variable_scope(tf.get_variable_scope()):
            # Loop over available_devices
            for i in range(len(Config.available_devices)):
                # Execute operations of tower i on device i
                device = Config.available_devices[i]
                with tf.device(device):
                    # Create a scope for all operations of tower i
                    with tf.name_scope('tower_%d' % i):
                        # Calculate the avg_loss and mean_edit_distance and retrieve the decoded
                        # batch along with the original batch's labels (Y) of this tower
                        avg_loss = self.calculate_mean_edit_distance_and_loss(iterator, dropout_rates, reuse=i > 0)

                        # Allow for variables to be re-used by the next tower
                        tf.get_variable_scope().reuse_variables()

                        # Retain tower's avg losses
                        tower_avg_losses.append(avg_loss)

                        # Compute gradients for model parameters using tower's mini-batch
                        gradients = optimizer.compute_gradients(avg_loss)

                        # Retain tower's gradients
                        tower_gradients.append(gradients)

        avg_loss_across_towers = tf.reduce_mean(tower_avg_losses, 0)

        tf.summary.scalar(name='step_loss', tensor=avg_loss_across_towers, collections=['step_summaries'])

        # Return gradients and the average loss
        return tower_gradients, avg_loss_across_towers

    def average_gradients(self, tower_gradients):
        r'''
        A routine for computing each variable's average of the gradients obtained from the GPUs.
        Note also that this code acts as a synchronization point as it requires all
        GPUs to be finished with their mini-batch before it can run to completion.
        '''
        # List of average gradients to return to the caller
        Config = self.c
        average_grads = []

        # Run this on cpu_device to conserve GPU memory
        with tf.device(Config.cpu_device):
            # Loop over gradient/variable pairs from all towers
            for grad_and_vars in zip(*tower_gradients):
                # Introduce grads to store the gradients for the current variable
                grads = []

                # Loop over the gradients for the current variable
                for g, _ in grad_and_vars:
                    # Add 0 dimension to the gradients to represent the tower.
                    expanded_g = tf.expand_dims(g, 0)
                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)

                # Average over the 'tower' dimension
                grad = tf.concat(grads, 0)
                grad = tf.reduce_mean(grad, 0)

                # Create a gradient/variable tuple for the current variable with its average gradient
                grad_and_var = (grad, grad_and_vars[0][1])

                # Add the current tuple to average_grads
                average_grads.append(grad_and_var)

        # Return result to caller
        return average_grads

    def log_variable(self, variable, gradient=None):
        r'''
        We introduce a function for logging a tensor variable's current state.
        It logs scalar values for the mean, standard deviation, minimum and maximum.
        Furthermore it logs a histogram of its state and (if given) of an optimization gradient.
        '''
        name = variable.name.replace(':', '_')
        mean = tf.reduce_mean(variable)
        tf.summary.scalar(name='%s/mean' % name, tensor=mean)
        tf.summary.scalar(name='%s/sttdev' % name, tensor=tf.sqrt(tf.reduce_mean(tf.square(variable - mean))))
        tf.summary.scalar(name='%s/max' % name, tensor=tf.reduce_max(variable))
        tf.summary.scalar(name='%s/min' % name, tensor=tf.reduce_min(variable))
        tf.summary.histogram(name=name, values=variable)
        if gradient is not None:
            if isinstance(gradient, tf.IndexedSlices):
                grad_values = gradient.values
            else:
                grad_values = gradient
            if grad_values is not None:
                tf.summary.histogram(name='%s/gradients' % name, values=grad_values)

    def log_grads_and_vars(self, grads_and_vars):
        r'''
        Let's also introduce a helper function for logging collections of gradient/variable tuples.
        '''
        for gradient, variable in grads_and_vars:
            self.log_variable(variable, gradient=gradient)

    def try_loading(self, session, saver, checkpoint_filename, caption):
        try:
            checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir, checkpoint_filename)
            if not checkpoint:
                return False
            checkpoint_path = checkpoint.model_checkpoint_path
            saver.restore(session, checkpoint_path)
            restored_step = session.run(tf.train.get_global_step())
            logger.log('Restored variables from %s checkpoint at %s, step %d' % (caption, checkpoint_path, restored_step))
            return True
        except tf.errors.InvalidArgumentError as e:
            logger.log(str(e))
            logger.log('The checkpoint in {0} does not match the shapes of the model.'
                       ' Did you change alphabet.txt or the --n_hidden parameter'
                       ' between train runs using the same checkpoint dir? Try moving'
                       ' or removing the contents of {0}.'.format(checkpoint_path))
            sys.exit(1)

    def samples_to_mfccs(self, samples, sample_rate):
        Config = self.c
        spectrogram = contrib_audio.audio_spectrogram(samples,
                                                      window_size=Config.audio_window_samples,
                                                      stride=Config.audio_step_samples,
                                                      magnitude_squared=True)
        mfccs = contrib_audio.mfcc(spectrogram, sample_rate, dct_coefficient_count=Config.n_input)
        mfccs = tf.reshape(mfccs, [-1, Config.n_input])

        return mfccs, tf.shape(mfccs)[0]

    def create_inference_graph(self, batch_size=1, n_steps=16):
        Config = self.c

        batch_size = batch_size if batch_size > 0 else None

        # Create feature computation graph
        input_samples = tf.placeholder(tf.float32, [Config.audio_window_samples], 'input_samples')
        samples = tf.expand_dims(input_samples, -1)
        mfccs, _ = self.samples_to_mfccs(samples, FLAGS.audio_sample_rate)
        mfccs = tf.identity(mfccs, name='mfccs')

        input_tensor = tf.placeholder(tf.float32, [batch_size, n_steps if n_steps > 0 else None, 2 * Config.n_context + 1, Config.n_input], name='input_node')
        seq_length = tf.placeholder(tf.int32, [batch_size], name='input_lengths')

        if not batch_size or batch_size <= 0:
            # No statement management since n_step is expected to be dynamic too (see below)
            previous_state = previous_state_c = previous_state_h = None
        else:
            previous_state_c = self.variable_on_cpu('previous_state_c', [batch_size, Config.n_cell_dim], initializer=None)
            previous_state_h = self.variable_on_cpu('previous_state_h', [batch_size, Config.n_cell_dim], initializer=None)
            previous_state = tf.contrib.rnn.LSTMStateTuple(previous_state_c, previous_state_h)

        # One rate per layer
        no_dropout = [None] * 6

        rnn_impl = self.rnn_impl_lstmblockfusedcell

        logits, layers = self.create_model(batch_x=input_tensor,
                                           seq_length=seq_length if FLAGS.use_seq_length else None,
                                           dropout=no_dropout,
                                           previous_state=previous_state,
                                           overlap=False,
                                           rnn_impl=rnn_impl)

        # Apply softmax for CTC decoder
        logits = tf.nn.softmax(logits)

        if not batch_size or batch_size <= 0:
            if n_steps > 0:
                logger.log('Dynamic batch_size expect n_steps to be dynamic too')
            return (
                {
                    'input': input_tensor,
                    'input_lengths': seq_length,
                },
                {
                    'outputs': tf.identity(logits, name='logits'),
                },
                layers
            )

        new_state_c, new_state_h = layers['rnn_output_state']

        zero_state = tf.zeros([batch_size, Config.n_cell_dim], tf.float32)
        initialize_c = tf.assign(previous_state_c, zero_state)
        initialize_h = tf.assign(previous_state_h, zero_state)
        initialize_state = tf.group(initialize_c, initialize_h, name='initialize_state')
        with tf.control_dependencies([tf.assign(previous_state_c, new_state_c), tf.assign(previous_state_h, new_state_h)]):
            logits = tf.identity(logits, name='logits')

        inputs = {
            'input': input_tensor,
            'input_lengths': seq_length,
            'input_samples': input_samples,
        }
        outputs = {
            'outputs': logits,
            'initialize_state': initialize_state,
            'mfccs': mfccs,
        }

        return inputs, outputs, layers

    def train(self, dataset_uri):
        tf.reset_default_graph()
        tf.set_random_seed(FLAGS.random_seed)

        ep = self._knobs.get('epochs')
        Config = self.c

        logger.log('Available devices: {}'.format(str(device_lib.list_local_devices())))

        # Define 2 plots: Loss against time, loss against epochs
        logger.define_loss_plot()
        logger.define_plot('Loss Over Time', ['loss'])

        # Create a temp directory to place the train files
        dataset_dir = tempfile.TemporaryDirectory()
        logger.log('Train dataset will be extracted to {}'.format(dataset_dir.name))

        train_set = self.create_dataset(dataset_uri, dataset_dir)

        iterator = tf.data.Iterator.from_structure(train_set.output_types,
                                                   train_set.output_shapes,
                                                   output_classes=train_set.output_classes)

        # Make initialization ops for switching between the two sets
        train_init_op = iterator.make_initializer(train_set)

        dropout_rates = [tf.placeholder(tf.float32, name='dropout_{}'.format(i)) for i in range(6)]
        dropout_feed_dict = {
            dropout_rates[0]: FLAGS.dropout_rate,
            dropout_rates[1]: FLAGS.dropout_rate2,
            dropout_rates[2]: FLAGS.dropout_rate3,
            dropout_rates[3]: FLAGS.dropout_rate4,
            dropout_rates[4]: FLAGS.dropout_rate5,
            dropout_rates[5]: FLAGS.dropout_rate6,
        }
        no_dropout_feed_dict = {
            rate: 0. for rate in dropout_rates
        }

        # Building the graph
        # with self._graph.as_default():
        optimizer = self.create_optimizer()
        gradients, loss = self.get_tower_results(iterator, optimizer, dropout_rates)

        # Average tower gradients across GPUS
        avg_tower_gradients = self.average_gradients(gradients)
        self.log_grads_and_vars(avg_tower_gradients)

        # global step is automatically incremented by the optimizer
        global_step = tf.train.get_or_create_global_step()
        apply_gradient_op = optimizer.apply_gradients(avg_tower_gradients, global_step=global_step)

        # Summaries
        step_summaries_op = tf.summary.merge_all('step_summaries')

        # Checkpointing
        checkpoint_saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'train')
        checkpoint_filename = 'checkpoint'

        initializer = tf.global_variables_initializer()

        # with self._sess.as_default():
        # with tf.Session(config=self._sess_config) as session:
        session = tf.Session(config=self._sess_config)
        # session = tf.get_default_session()
        self._sess = session
        tf.get_default_graph().finalize()

        # Initializing
        logger.log('Initializing variables...')
        session.run(initializer)

        def run_set(set_name, epoch, init_op, dataset=None):
            is_train = set_name == 'train'
            train_op = apply_gradient_op if is_train else []
            feed_dict = dropout_feed_dict if is_train else no_dropout_feed_dict

            total_loss = 0.0
            step_count = 0
            checkpoint_time = time.time()

            # Initialize iterator to the appropriate dataset
            session.run(init_op)

            # Batch loop
            while True:
                try:
                    _, current_step, batch_loss, step_summary = \
                        session.run([train_op, global_step, loss, step_summaries_op], feed_dict=feed_dict)
                except tf.errors.OutOfRangeError:
                    break

                total_loss += batch_loss
                step_count += 1

                if is_train and FLAGS.checkpoint_secs > 0 and time.time() - checkpoint_time > FLAGS.checkpoint_secs:
                    checkpoint_saver.save(session, checkpoint_path, global_step=current_step)
                    checkpoint_time = time.time()

            mean_loss = total_loss / step_count if step_count > 0 else 0.0
            return mean_loss, step_count

        logger.log('STARTING OPTIMIZATION')
        train_start_time = datetime.utcnow()
        try:
            for epoch in range(ep):
                # Training
                logger.log('Training epoch %d...' % epoch)
                train_loss, _ = run_set('train', epoch, train_init_op)
                logger.log('Finished training epoch %d - loss: %f' % (epoch, train_loss))
                checkpoint_saver.save(session, checkpoint_path, global_step=global_step)

        except KeyboardInterrupt:
            pass
        logger.log('FINISHED optimization in {}'.format(datetime.utcnow() - train_start_time))

    def evaluate(self, dataset_uri):
        Config = self.c
        tf.reset_default_graph()

        scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                        FLAGS.lm_binary_path, FLAGS.lm_trie_path,
                        Config.alphabet)

        dataset_dir = tempfile.TemporaryDirectory()
        logger.log('Test dataset will be extracted to {}'.format(dataset_dir.name))

        test_set = self.create_dataset(dataset_uri, dataset_dir)

        iterator = tf.data.Iterator.from_structure(test_set.output_types,
                                                   test_set.output_shapes,
                                                   output_classes=test_set.output_classes)

        test_init_op = iterator.make_initializer(test_set)

        (batch_x, batch_x_len), batch_y = iterator.get_next()

        # One rate per layer
        no_dropout = [None] * 6
        logits, _ = self.create_model(batch_x=batch_x,
                                      seq_length=batch_x_len,
                                      dropout=no_dropout)

        # Transpose to batch major and apply softmax for decoder
        transposed = tf.nn.softmax(tf.transpose(logits, [1, 0, 2]))

        loss = tf.nn.ctc_loss(labels=batch_y,
                              inputs=logits,
                              sequence_length=batch_x_len)

        tf.train.get_or_create_global_step()

        # Get number of accessible CPU cores for this process
        try:
            num_processes = cpu_count()
        except NotImplementedError:
            num_processes = 1

        # Create a saver using variables from the above newly created graph
        saver = tf.train.Saver()

        # with tf.Session(config=self._sess_config) as session:
        session = tf.Session(config=self._sess_config)
        self._sess = session

        # Restore variables from training checkpoint
        loaded = self.try_loading(session, saver, 'checkpoint', 'most recent')
        if not loaded:
            logger.log('Checkpoint directory ({}) does not contain a valid checkpoint state.'
                       .format(FLAGS.checkpoint_dir))
            sys.exit(1)

        def sparse_tuple_to_texts(sp_tuple, alphabet):
            indices = sp_tuple[0]
            values = sp_tuple[1]
            results = [''] * sp_tuple[2][0]
            for i, index in enumerate(indices):
                results[index[0]] += alphabet.string_from_label(values[i])
            # List of strings
            return results

        def sparse_tensor_value_to_texts(value, alphabet):
            r"""
            Given a :class:`tf.SparseTensor` ``value``, return an array of Python strings
            representing its values, converting tokens to strings using ``alphabet``.
            """
            return sparse_tuple_to_texts((value.indices, value.values, value.dense_shape), alphabet)

        def calculate_report(labels, decodings, losses):
            r'''
            This routine will calculate a WER report.
            It'll compute the `mean` WER and create ``Sample`` objects of the ``report_count`` top lowest
            loss items from the provided WER results tuple (only items with WER!=0 and ordered by their WER).
            '''

            def pmap(fun, iterable):
                pool = Pool()
                results = pool.map(fun, iterable)
                pool.close()
                return results

            def wer_cer_batch(samples):
                r"""
                The WER is defined as the edit/Levenshtein distance on word level divided by
                the amount of words in the original text.
                In case of the original having more words (N) than the result and both
                being totally different (all N words resulting in 1 edit operation each),
                the WER will always be 1 (N / N = 1).
                """
                wer = sum(s.word_distance for s in samples) / sum(s.word_length for s in samples)
                cer = sum(s.char_distance for s in samples) / sum(s.char_length for s in samples)

                wer = min(wer, 1.0)
                cer = min(cer, 1.0)

                return wer, cer

            def levenshtein(a, b):
                "Calculates the Levenshtein distance between a and b."
                n, m = len(a), len(b)
                if n > m:
                    # Make sure n <= m, to use O(min(n,m)) space
                    a, b = b, a
                    n, m = m, n

                current = list(range(n + 1))
                for i in range(1, m + 1):
                    previous, current = current, [i] + [0] * n
                    for j in range(1, n + 1):
                        add, delete = previous[j] + 1, current[j - 1] + 1
                        change = previous[j - 1]
                        if a[j - 1] != b[i - 1]:
                            change = change + 1
                        current[j] = min(add, delete, change)

                return current[n]

            def process_decode_result(item):
                ground_truth, prediction, loss = item
                char_distance = levenshtein(ground_truth, prediction)
                char_length = len(ground_truth)
                word_distance = levenshtein(ground_truth.split(), prediction.split())
                word_length = len(ground_truth.split())
                return AttrDict({
                    'src': ground_truth,
                    'res': prediction,
                    'loss': loss,
                    'char_distance': char_distance,
                    'char_length': char_length,
                    'word_distance': word_distance,
                    'word_length': word_length,
                    'cer': char_distance / char_length,
                    'wer': word_distance / word_length,
                })

            samples = pmap(process_decode_result, zip(labels, decodings, losses))

            # Getting the WER and CER from the accumulated edit distances and lengths
            samples_wer, samples_cer = wer_cer_batch(samples)

            # Order the remaining items by their loss (lowest loss on top)
            samples.sort(key=lambda s: s.loss)

            # Then order by WER (highest WER on top)
            samples.sort(key=lambda s: s.wer, reverse=False)

            return samples_wer, samples_cer, samples

        def run_test(init_op, dataset):
            losses = []
            predictions = []
            ground_truths = []

            logger.log('Running the test set...')

            # Initialize iterator to the appropriate dataset
            session.run(init_op)

            # First pass, compute losses and transposed logits for decoding
            while True:
                try:
                    batch_logits, batch_loss, batch_lengths, batch_transcripts = \
                        session.run([transposed, loss, batch_x_len, batch_y])
                except tf.errors.OutOfRangeError:
                    break

                decoded = ctc_beam_search_decoder_batch(batch_logits, batch_lengths,
                                                        Config.alphabet, FLAGS.beam_width,
                                                        num_processes=num_processes, scorer=scorer)

                predictions.extend(d[0][1] for d in decoded)
                ground_truths.extend(sparse_tensor_value_to_texts(batch_transcripts, Config.alphabet))
                losses.extend(batch_loss)

            wer, cer, samples = calculate_report(ground_truths, predictions, losses)
            mean_loss = np.mean(losses)

            # Take only the first report_count items
            report_samples = itertools.islice(samples, FLAGS.report_count)

            logger.log('Test on %s - WER: %f, CER: %f, loss: %f' % (dataset, wer, cer, mean_loss))
            logger.log('-' * 80)
            for sample in report_samples:
                print('WER: %f, CER: %f, loss: %f' %\
                      (sample.wer, sample.cer, sample.loss))
                print(' - src: "%s"' % sample.src)
                print(' - res: "%s"' % sample.res)
                print('-' * 80)

            return samples, mean_loss

        samples, mean_loss = run_test(test_init_op, dataset=dataset_uri)

        return float(mean_loss)

    def predict(self, queries, n_steps=16):
        # Load from graph_def saved in the class attribute
        Config = self.c

        new_input_tensor = tf.placeholder(tf.float32, [None, None, 2 * Config.n_context + 1, Config.n_input], name='input_node')
        new_seq_length = tf.placeholder(tf.int32, [None], name='input_lengths')

        tf.import_graph_def(self._loaded_graph_def,
                            {'input_node:0': new_input_tensor, 'input_lengths:0': new_seq_length},
                            name='')

        input = self.graph.get_tensor_by_name('input_node:0')
        input_lengths = self.graph.get_tensor_by_name('input_lengths:0')
        output = self.graph.get_tensor_by_name('logits:0')

        session = self._sess

        predictions = []
        for input_file_path in queries:
            features, features_len = self.audiofile_to_features(input_file_path)

            # Add batch dimension
            features = tf.expand_dims(features, 0)
            features_len = tf.expand_dims(features_len, 0)

            # Evaluate
            features = self.create_overlapping_windows(features).eval(session=session)
            features_len = features_len.eval(session=session)

            logits = output.eval(feed_dict={
                input: features,
                input_lengths: features_len,
            }, session=session)

            logits = np.squeeze(logits)

            scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                            FLAGS.lm_binary_path, FLAGS.lm_trie_path,
                            Config.alphabet)
            decoded = ctc_beam_search_decoder(logits, Config.alphabet, FLAGS.beam_width, scorer=scorer)

            predictions.append(decoded[0][1])

        return predictions

    def destroy(self):
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            FLAGS.__delattr__(keys)
        pass

    def dump_parameters(self):
        r'''
        Export the trained variables into a Protocol Buffers (.pb) file and dump into the DB
        Use a structure optimal for inference
        '''

        tf.reset_default_graph()
        batch_size = self._knobs.get('batch_size')
        input, outputs, _ = self.create_inference_graph(batch_size=-1, n_steps=-1)
        output_names_tensor = [tensor.op.name for tensor in outputs.values() if isinstance(tensor, Tensor)]
        output_names_ops = [op.name for op in outputs.values() if isinstance(op, Operation)]
        output_names = ','.join(output_names_tensor + output_names_ops)

        mapping = {v.op.name: v for v in tf.global_variables() if not v.op.name.startswith('previous_state_')}
        saver = tf.train.Saver(mapping)

        # Restore variables from training checkpoint
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        checkpoint_path = checkpoint.model_checkpoint_path

        output_filename = 'output_graph.pb'
        export_temp_dir = tempfile.TemporaryDirectory()
        export_dir = export_temp_dir.name

        try:
            output_graph_path = os.path.join(export_dir, output_filename)

            def do_graph_freeze(output_file=None, output_node_names=None, variables_blacklist=None):
                return freeze_graph.freeze_graph_with_def_protos(
                    input_graph_def=tf.get_default_graph().as_graph_def(),
                    input_saver_def=saver.as_saver_def(),
                    input_checkpoint=checkpoint_path,
                    output_node_names=output_node_names,
                    restore_op_name=None,
                    filename_tensor_name=None,
                    output_graph=output_file,
                    clear_devices=False,
                    variable_names_blacklist=variables_blacklist,
                    initializer_nodes='')

            frozen_graph = do_graph_freeze(output_node_names=output_names, variables_blacklist='previous_state_c,previous_state_h')
            frozen_graph.version = 1

            with tf.gfile.GFile(output_graph_path, 'wb') as fout:
                fout.write(frozen_graph.SerializeToString())

            params = {}
            # Read from temp pb file & encode it to base64 string
            with open(output_graph_path, 'rb') as f:
                pb_model_bytes = f.read()

            params['pb_model_base64'] = base64.b64encode(pb_model_bytes).decode('utf-8')
            return params

        except RuntimeError as e:
            logger.log('Error occured! {}'.format(e))

    def load_parameters(self, params):
        # Load the Protocol Buffers into graph def
        tf.reset_default_graph()
        self.graph = tf.Graph()
        self._sess = tf.InteractiveSession(graph=self.graph)

        # Load model parameters
        pb_model_base64 = params.get('pb_model_base64', None)
        if pb_model_base64 is None:
            raise InvalidModelParamsException()
        pb_model_bytes = base64.b64decode(pb_model_base64.encode('utf-8'))

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(pb_model_bytes)
        self._loaded_graph_def = graph_def


if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='TfDeepSpeech',
        task=TaskType.SPEECH_RECOGNITION,
        dependencies={
            # ModelDependency.TENSORFLOW: '1.12.0',
            ModelDependency.DS_CTCDECODER: os.path.abspath('examples/models/speech_recognition/utils/taskcluster.py')
        },
        # Demonstrative only, this dataset only contains one sample, we use batch_size = 1 to run
        # Replace with larger test data and larger batch_size in practice
        train_dataset_uri=os.path.abspath('data/ldc93s1/ldc93s1.zip'),
        test_dataset_uri=os.path.abspath('data/ldc93s1/ldc93s1.zip'),
        queries=[os.path.abspath('data/ldc93s1/ldc93s1/LDC93S1.wav')]
    )