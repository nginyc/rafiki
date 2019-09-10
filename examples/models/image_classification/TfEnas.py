#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.training import moving_averages
import os
import math
import json
import random
from datetime import datetime
from collections import namedtuple
import numpy as np
import argparse

from rafiki.constants import ModelDependency
from rafiki.model import utils, BaseModel, IntegerKnob, CategoricalKnob, FloatKnob, \
                            FixedKnob, ArchKnob, KnobValue, PolicyKnob
from rafiki.model.dev import test_model_class

_Model = namedtuple('_Model', 
                    ['train_dataset_init_op', 'pred_dataset_init_op', 
                    'train_op', 'summary_op', 'pred_probs', 'pred_corrects', 
                    'train_corrects', 'step', 'vars_assign_op', 'ph', 'var_phs'])
_ModelMemo = namedtuple('_ModelMemo', 
                    ['train_params', 'use_dynamic_arch', 'knobs', 
                    'model', 'graph', 'sess', 'saver', 'monitored_values'])
_ModelPlaceholder = namedtuple('_ModelPlaceholder', 
                                ['train_images', 'train_classes', 'pred_images', 
                                'pred_classes', 'normal_arch', 'reduction_arch'])

OPS = [0, 1, 2, 3, 4] 
CELL_NUM_BLOCKS = 5 # No. of blocks in a cell
TF_COLLECTION_MONITORED = 'MONITORED'

class TfEnas(BaseModel):
    '''
        Implements the child model of cell-based "Efficient Neural Architecture Search via Parameter Sharing" (ENAS) 
        for IMAGE_CLASSIFICATION, configured for *architecture tuning with ENAS* on Rafiki. 
        
        Original paper: https://arxiv.org/abs/1802.03268
        Implementation is with credits to https://github.com/melodyguan/enas
    '''
    # Memoise across trials to speed up training
    _datasets_memo = {}                 # { <dataset_path> -> <dataset> }
    _model_memo = None                  # of class `_MemoModel`
    _loaded_tf_vars_id_memo = None      # ID of TF vars loaded
    _loaded_train_dataset_memo = None   # Train dataset <dataset_path> loaded into the graph
    _loaded_pred_dataset_memo = None    # Predict dataset <dataset_path> loaded into the graph

    @staticmethod
    def get_knob_config():
        return {
            'cell_archs': TfEnas.make_arch_knob(),
            'max_image_size': FixedKnob(32),
            'epochs': FixedKnob(310), # Total no. of epochs during a standard train
            'batch_size': FixedKnob(128),
            'learning_rate': FixedKnob(0.05), 
            'initial_block_ch': FixedKnob(36),
            'stem_ch_mul': FixedKnob(3),
            'reg_decay': FixedKnob(4e-4),
            'dropout_keep_prob': FixedKnob(0.8),
            'opt_momentum': FixedKnob(0.9),
            'use_sgdr': FixedKnob(True),
            'sgdr_alpha': FixedKnob(0.002),
            'sgdr_decay_epochs': FixedKnob(10),
            'sgdr_t_mul': FixedKnob(2),  
            'num_layers': FixedKnob(15), 
            'aux_loss_mul': FixedKnob(0.4),
            'drop_path_keep_prob': FixedKnob(0.6),
            'drop_path_decay_epochs': FixedKnob(310),
            'cutout_size': FixedKnob(0),
            'grad_clip_norm': FixedKnob(0),
            'use_aux_head': FixedKnob(False),
            'share_params': PolicyKnob('SHARE_PARAMS'),

            # Affects whether model constructed is a scaled-down version with fewer layers
            'downscale': PolicyKnob('DOWNSCALE'), 
            'enas_num_layers': FixedKnob(6), 
            'enas_initial_block_ch': FixedKnob(20), 
            'enas_dropout_keep_prob': FixedKnob(0.9),
            'enas_sgdr_alpha': FixedKnob(0.01),
            'enas_drop_path_keep_prob': FixedKnob(0.9),
            'enas_drop_path_decay_epochs': FixedKnob(150),

            # Affects whether training is shortened using a reduced no. of epochs
            'quick_train': PolicyKnob('EARLY_STOP'), 
            'enas_epochs': FixedKnob(1),

            # Affects whether training is skipped
            'skip_train': PolicyKnob('SKIP_TRAIN'),

            # Affects whether evaluation is done on only a batch of the validation dataset
            'quick_eval': PolicyKnob('QUICK_EVAL')
        }

    @staticmethod
    def make_arch_knob():
        # Make knob values for ops
        # Operations across blocks are considered identical for the purposes of architecture search
        # E.g. operation "conv3x3" with op code 0 has the same meaning across blocks 
        ops = [KnobValue(i) for i in OPS]

        # Build list of knobs for ``cell_archs``
        cell_archs = []
        for c in range(2): # 1 for normal cell, 1 for reduction cell
            
            # Make knob values for inputs
            # Input indices across blocks in the same cell are considered identical for the purposes of architecture search
            # E.g. input from block 0 with index 2 has the same meaning across blocks in the same cell 
            input_knob_values = [KnobValue(i) for i in range(CELL_NUM_BLOCKS + 2)]

            # For each block
            for b in range(CELL_NUM_BLOCKS): 
                # Input 1 & input 2 can only can take input from prev prev cell, prev cell, or one of prev blocks
                inputs = input_knob_values[:(b + 2)]
                cell_archs.extend([inputs, ops, inputs, ops]) 
            
        return ArchKnob(cell_archs)
    
    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._model = None
        self._graph = None
        self._sess = None
        self._saver = None
        self._monitored_values = None
        self._train_params = None
        self._knobs = self._process_knobs(knobs)

    def train(self, dataset_path, shared_params=None):
        knobs = self._knobs

        # Load dataset
        (images, classes, self._train_params) = self._maybe_load_dataset(dataset_path, **knobs)
        
        # Build model
        (self._model, self._graph, self._sess, self._saver, 
            self._monitored_values) = self._maybe_build_model(**knobs)
        
        if not knobs['skip_train']:
            # Maybe load shared variables, then train model
            with self._graph.as_default():
                if knobs['share_params'] and shared_params is not None:
                    self._maybe_load_tf_vars(shared_params)
                self._train_model(images, classes, dataset_path=dataset_path, **knobs)

    def evaluate(self, dataset_path):
        (images, classes, _) = self._maybe_load_dataset(dataset_path, train_params=self._train_params, **self._knobs)
        
        with self._graph.as_default():
            acc = self._evaluate_model(images, classes, dataset_path=dataset_path, **self._knobs)

        return acc

    def predict(self, queries):
        image_size = self._train_params['image_size']
        images = utils.dataset.transform_images(queries, image_size=image_size, mode='RGB')
        with self._graph.as_default():
            probs = self._predict_with_model(images, **self._knobs)
        return probs.tolist()

    def dump_parameters(self):
        params = {}

        # Add train params
        params['train_params'] = json.dumps(self._train_params)

        # Add model parameters
        with self._graph.as_default():
            tf_vars = tf.global_variables()
            values = self._sess.run(tf_vars)

            for (tf_var, value) in zip(tf_vars, values):
                params[tf_var.name] = np.asarray(value)

        # Add an ID for diffing
        vars_id = np.random.rand()
        params['vars_id'] = vars_id

        # Memo ID
        TfEnas._loaded_tf_vars_id_memo = vars_id

        return params

    def load_parameters(self, params):
        # Add train params
        self._train_params = json.loads(params['train_params'])

        # Build model
        (self._model, self._graph, self._sess, 
            self._saver, self._monitored_values) = self._maybe_build_model(**self._knobs)

        # Add model parameters
        with self._graph.as_default():
            self._maybe_load_tf_vars(params)

    @staticmethod
    def teardown():
        if TfEnas._model_memo is not None:
            TfEnas._model_memo.sess.close()
            TfEnas._model_memo = None

    ####################################
    # Memoized methods
    ####################################

    def _maybe_load_dataset(self, dataset_path, train_params=None, **knobs):
        # Try to use memoized dataset
        if dataset_path in TfEnas._datasets_memo:
            dataset = TfEnas._datasets_memo[dataset_path]
            return dataset

        dataset = self._load_dataset(dataset_path, train_params, **knobs)
        TfEnas._datasets_memo[dataset_path] = dataset
        return dataset

    def _maybe_load_tf_vars(self, params):
        # If same TF vars has been loaded in previous trial, don't bother loading again
        vars_id = params['vars_id']
        
        if TfEnas._loaded_tf_vars_id_memo == vars_id:
            return  # Skipping loading of vars

        self._load_tf_vars(params)

        # Memo ID
        TfEnas._loaded_tf_vars_id_memo = vars_id

    def _maybe_feed_dataset_to_model(self, images, classes=None, dataset_path=None, is_train=False):
        memo = TfEnas._loaded_train_dataset_memo if is_train else TfEnas._loaded_pred_dataset_memo
        if dataset_path is None or memo != dataset_path:
            # To load new dataset
            self._feed_dataset_to_model(images, classes, is_train=is_train)
            if is_train:
                TfEnas._loaded_train_dataset_memo = dataset_path
            else:
                TfEnas._loaded_pred_dataset_memo = dataset_path
        else:
            # Otherwise, dataset has previously been loaded, so do nothing
            pass
        
    def _maybe_build_model(self, **knobs):
        train_params = self._train_params
        use_dynamic_arch = knobs['downscale']

        # Use memoized model when possible
        if not self._if_model_same(TfEnas._model_memo, knobs, train_params, use_dynamic_arch):

            (model, graph, sess, saver, monitored_values) = \
                self._build_model(**knobs)

            TfEnas._model_memo = _ModelMemo(
                train_params, use_dynamic_arch, knobs, 
                model, graph, sess, saver, monitored_values 
            )

        model_memo = TfEnas._model_memo
        return (model_memo.model, model_memo.graph, model_memo.sess, model_memo.saver, 
                model_memo.monitored_values)
                
    def _if_model_same(self, model_memo, knobs, train_params, use_dynamic_arch):
        if model_memo is None:
            return False

        # Must have the same `train_params` & `use_dynamic_arch`
        if (train_params, use_dynamic_arch) != (model_memo.train_params, model_memo.use_dynamic_arch):
            return False

        # Knobs must be the same except for some that doesn't affect model construction
        # If arch is dynamic, knobs can differ by `cell_archs`
        ignored_knobs = ['skip_train', 'quick_train', 'quick_eval', 'downscale', 'epochs']
        if use_dynamic_arch:
            ignored_knobs.append('cell_archs')
        
        for (name, value) in knobs.items():
            if name not in ignored_knobs and value != model_memo.knobs.get(name):
                utils.logger.log('Detected that knob "{}" is different!'.format(name))
                return False
        
        return True

    ####################################
    # Private methods
    ####################################

    def _process_knobs(self, knobs):
        # Activates dynamic architecture with fewer layers 
        if knobs['downscale']:
            knobs = {
                **knobs,
                'num_layers': knobs['enas_num_layers'],
                'initial_block_ch': knobs['enas_initial_block_ch'],
                'dropout_keep_prob': knobs['enas_dropout_keep_prob'],
                'sgdr_alpha': knobs['enas_sgdr_alpha'],
                'drop_path_keep_prob': knobs['enas_drop_path_keep_prob'],
                'drop_path_decay_epochs': knobs['enas_drop_path_decay_epochs'],
            }

        # Activates mode where training finishes with fewer epochs
        if knobs['quick_train']:
            knobs = {
                **knobs,
                'epochs': knobs['enas_epochs'] 
            }

        return knobs

    def _load_dataset(self, dataset_path, train_params=None, **knobs):
        max_image_size = knobs['max_image_size']
        image_size = train_params['image_size'] if train_params is not None else max_image_size

        utils.logger.log('Loading dataset...')    
        dataset = utils.dataset.load_dataset_of_image_files(dataset_path, max_image_size=image_size, 
                                                            mode='RGB')
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        norm_mean = np.mean(images, axis=(0, 1, 2)).tolist() 
        norm_std = np.std(images, axis=(0, 1, 2)).tolist()  
          
        train_params = {
            'N': len(images),
            'image_size': dataset.image_size,
            'K': dataset.classes,
            'norm_mean': norm_mean,
            'norm_std': norm_std
        }
        
        return (images, classes, train_params)

    def _build_model(self, **knobs):
        use_dynamic_arch = knobs['downscale']

        # Create graph
        graph = tf.Graph()
        
        with graph.as_default():
            # Define input placeholders to graph
            ph = self._make_placeholders()

            # Use fixed archs if specified, otherwise use placeholders'
            (normal_arch, reduction_arch) = self._get_fixed_cell_archs(**knobs)
            normal_arch = normal_arch if not use_dynamic_arch else ph.normal_arch
            reduction_arch = reduction_arch if not use_dynamic_arch else ph.reduction_arch

            # Initialize steps variable
            step = self._make_var('step', (), dtype=tf.int32, trainable=False, initializer=tf.initializers.constant(0))

            # For train dataset, preprocess & do inference
            utils.logger.log('Building model for training...')
            (train_X, train_classes, train_dataset_init_op) = \
                self._preprocess(ph.train_images, ph.train_classes, is_train=True, **knobs)
            (train_logits, train_aux_logits_list) = self._forward(train_X, step, normal_arch, reduction_arch, is_train=True, **knobs)
            
            # Compute training loss 
            total_loss = self._compute_loss(train_logits, train_aux_logits_list, train_classes, **knobs)

            # Optimize training loss 
            train_op = self._optimize(total_loss, step, **knobs)

            # Compute predictions 
            (_, train_corrects) = self._compute_predictions(train_logits, train_classes)

            # For pred dataset, preprocess & do inference
            utils.logger.log('Building model for predictions...')
            (pred_X, pred_classes, pred_dataset_init_op) = \
                self._preprocess(ph.pred_images, ph.pred_classes, is_train=False, **knobs)
            (pred_logits, _) = self._forward(pred_X, step, normal_arch, reduction_arch, is_train=False, 
                                            **knobs)

            # Compute predictions 
            (pred_probs, pred_corrects) = self._compute_predictions(pred_logits, pred_classes)

            # Count model parameters 
            model_params_count = self._count_model_parameters()

            # Monitor values
            (summary_op, monitored_values) = self._add_monitoring_of_values()

            # Add saver
            tf_vars = tf.global_variables()
            saver = tf.train.Saver(tf_vars)

            # Allow loading of model variables
            (var_phs, vars_assign_op) = self._add_vars_assign_op(tf_vars)

            model = _Model(train_dataset_init_op, pred_dataset_init_op, train_op, summary_op, 
                            pred_probs, pred_corrects, train_corrects, step, vars_assign_op, ph, var_phs)

            # Make session
            sess = self._make_session()
            self._init_session(sess, model)

        return (model, graph, sess, saver, monitored_values)

    def _load_tf_vars(self, params):
        m = self._model

        utils.logger.log('Loading TF vars...')

        tf_vars = tf.global_variables()
        values = self._sess.run(tf_vars) # Get current values for vars

        # Build feed dict for op for loading vars
        # For each var, use current value of param in session if not in params
        var_feeddict = {
            m.var_phs[tf_var.name]: params[tf_var.name] 
            if tf_var.name in params else values[i]
            for (i, tf_var) in enumerate(tf_vars)
        }

        self._sess.run(m.vars_assign_op, feed_dict=var_feeddict)

    def _make_placeholders(self):
        w = self._train_params['image_size']
        h = self._train_params['image_size']
        in_ch = 3 # Num channels of input images

        train_images_ph = tf.placeholder(tf.int32, name='train_images_ph', shape=(None, w, h, in_ch)) # Train images
        pred_images_ph = tf.placeholder(tf.int32, name='pred_images_ph', shape=(None, w, h, in_ch)) # Predict images
        train_classes_ph = tf.placeholder(tf.int32, name='train_classes_ph', shape=(None,)) # Train classes
        pred_classes_ph = tf.placeholder(tf.int32, name='pred_classes_ph', shape=(None,)) # Predict classes
        normal_arch_ph = tf.placeholder(tf.int32, name='normal_arch_ph', shape=(CELL_NUM_BLOCKS, 4))
        reduction_arch_ph = tf.placeholder(tf.int32, name='reduction_arch_ph', shape=(CELL_NUM_BLOCKS, 4))

        return _ModelPlaceholder(train_images_ph, train_classes_ph, pred_images_ph, pred_classes_ph, 
                                normal_arch_ph, reduction_arch_ph)

    def _forward(self, X, step, normal_arch, reduction_arch, is_train=False, **knobs):
        K = self._train_params['K'] # No. of classes
        in_ch = 3 # Num channels of input images
        w = self._train_params['image_size'] # Initial input width
        h = self._train_params['image_size'] # Initial input height
        dropout_keep_prob = knobs['dropout_keep_prob']
        use_dynamic_arch = knobs['downscale']
        L = knobs['num_layers'] # Total number of layers
        initial_block_ch = knobs['initial_block_ch'] # Initial no. of channels for operations in block
        stem_ch_mul = knobs['stem_ch_mul'] # No. of channels for stem convolution as multiple of initial block channels
        use_aux_head = knobs['use_aux_head'] # Whether to use auxiliary head
        stem_ch = initial_block_ch * stem_ch_mul
        
        # Layers with reduction cells (otherwise, normal cells)
        reduction_layers = [L // 3, L // 3 * 2 + 1] 

        # Layers with auxiliary heads
        # Aux heads speed up training of good feature repsentations early in the network
        # Add aux heads only if enabled and downsampling width can happen 3 times
        aux_head_layers = []
        if use_aux_head and w % (2 << 3) == 0:
            aux_head_layers.append(reduction_layers[-1] + 1)

        with tf.variable_scope('model', reuse=(not is_train)):
            
            # "Stem" convolution layer (layer -1)
            with tf.variable_scope('layer_stem'):
                X = self._do_conv(X, w, h, in_ch, stem_ch, filter_size=3, no_relu=True, is_train=is_train) # 3x3 convolution
                stem = (X, w, h, stem_ch)

            # Core layers of cells
            block_ch = initial_block_ch
            aux_logits_list = [] # Stores list of logits from aux heads
            layers = [stem, stem] # Stores previous layers. layers[i] = (<layer (i + 1)>, <width>, <height>, <channels>)
            for l in range(L + 2):
                utils.logger.log('Building layer {}...'.format(l))
                
                with tf.variable_scope('layer_{}'.format(l)):
                    layers_ratio = (l + 1) / (L + 2)
                    drop_path_keep_prob = self._get_drop_path_keep_prob(layers_ratio, step, is_train, **knobs)
                    
                    # Either add a reduction cell or normal cell
                    if l in reduction_layers:
                        block_ch *= 2
                        w >>= 1
                        h >>= 1

                        with tf.variable_scope('reduction_cell'):
                            if use_dynamic_arch:
                                self._add_dynamic_cell(reduction_arch, layers, w, h, block_ch, drop_path_keep_prob, is_train)
                            else:
                                self._add_static_cell(reduction_arch, layers, w, h, block_ch, drop_path_keep_prob, is_train,
                                                    is_reduction=True)
                    else:
                        with tf.variable_scope('normal_cell'):
                            if use_dynamic_arch:
                                self._add_dynamic_cell(normal_arch, layers, w, h, block_ch, drop_path_keep_prob, is_train)
                            else:
                                self._add_static_cell(normal_arch, layers, w, h, block_ch, drop_path_keep_prob, is_train)

                    # Maybe add auxiliary heads 
                    if l in aux_head_layers:
                        with tf.variable_scope('aux_head'):
                            aux_logits = self._add_aux_head(*layers[-1], K, is_train)
                        aux_logits_list.append(aux_logits)

            # Global average pooling
            (X, w, h, ch) = layers[-1]
            X = self._add_global_avg_pool(X, w, h, ch)

            # Add dropout if training
            if is_train:
                X = tf.nn.dropout(X,  dropout_keep_prob)

            # Compute logits from X
            with tf.variable_scope('fully_connected'):
                logits = self._add_fully_connected(X, (ch,), K)
        
        return (logits, aux_logits_list)

    def _optimize(self, loss, step, **knobs):
        opt_momentum = knobs['opt_momentum'] # Momentum optimizer momentum
        grad_clip_norm = knobs['grad_clip_norm'] # L2 norm to clip gradients by

        # Compute learning rate, gradients
        tf_trainable_vars = tf.trainable_variables()
        lr = self._get_learning_rate(step, **knobs)
        grads = tf.gradients(loss, tf_trainable_vars)
        self._mark_for_monitoring('lr', lr)

        # Clip gradients
        if grad_clip_norm > 0:
            grads = [tf.clip_by_norm(x, grad_clip_norm) for x in grads]

        # Init optimizer
        opt = tf.train.MomentumOptimizer(lr, opt_momentum, use_locking=True, use_nesterov=True)
        train_op = opt.apply_gradients(zip(grads, tf_trainable_vars), global_step=step)

        return train_op

    def _preprocess(self, images, classes, is_train=False, **knobs):
        batch_size = knobs['batch_size']
        cutout_size = knobs['cutout_size']
        image_norm_mean = self._train_params['norm_mean']
        image_norm_std = self._train_params['norm_std']
        w = self._train_params['image_size']
        h = self._train_params['image_size']
        in_ch = 3 # Num channels of input images
        
        def _prepare(images, classes):
            # Bulk preprocessing of images
            images = tf.cast(images, tf.float32)
            images = (images - image_norm_mean) / image_norm_std # Normalize
            images = images / 255 # Convert to [0, 1]
            return (images, classes)

        # Prepare train dataset
        def _preprocess_train(image, clazz):
            # Do random crop + horizontal flip for each train image
            image = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
            image = tf.image.random_crop(image, (w, h, in_ch))
            image = tf.image.random_flip_left_right(image)

            if cutout_size > 0:
                image = self._do_cutout(image, w, h, cutout_size)
            
            return (image, clazz)
        
        (images, classes) = _prepare(images, classes) 
        dataset = tf.data.Dataset.from_tensor_slices((images, classes)).repeat()
        if is_train:
            dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=_preprocess_train, batch_size=batch_size))
        else:
            dataset = dataset.batch(batch_size)
        dataset_itr = dataset.make_initializable_iterator()
        (images_batch, classes_batch) = dataset_itr.get_next()
        dataset_init_op = dataset_itr.initializer

        return (images_batch, classes_batch, dataset_init_op)
    
    def _get_drop_path_keep_prob(self, layers_ratio, step, is_train=False, **knobs):
        batch_size = knobs['batch_size'] 
        drop_path_keep_prob = knobs['drop_path_keep_prob'] # Base keep prob for drop path
        drop_path_decay_epochs = knobs['drop_path_decay_epochs']
        N = self._train_params['N']

        # Only drop path during training
        keep_prob = tf.constant(1, dtype=tf.float32)
        if is_train:
            # Decrease keep prob deeper into network
            keep_prob = 1 - layers_ratio * (1 - drop_path_keep_prob)
            
            # Decrease keep prob with increasing steps
            steps_per_epoch = math.ceil(N / batch_size)
            steps_ratio = tf.minimum(((step + 1) / steps_per_epoch) / drop_path_decay_epochs, 1)
            keep_prob = 1 - steps_ratio * (1 - keep_prob)
            keep_prob = tf.cast(keep_prob, tf.float32)

            # Monitor last layer's keep prob
            if layers_ratio == 1:
                self._mark_for_monitoring('drop_path_keep_prob', keep_prob)

        return keep_prob

    def _get_learning_rate(self, step, **knobs):
        N = self._train_params['N']
        batch_size = knobs['batch_size'] 
        lr = knobs['learning_rate'] # Learning rate
        use_sgdr = knobs['use_sgdr']
        sgdr_decay_epochs = knobs['sgdr_decay_epochs']
        sgdr_alpha = knobs['sgdr_alpha'] 
        sgdr_t_mul = knobs['sgdr_t_mul']

        # Compute epoch from step
        steps_per_epoch = math.ceil(N / batch_size)
        epoch = step // steps_per_epoch

        if use_sgdr is True:
            # Apply Stoachastic Gradient Descent with Warm Restarts (SGDR)
            lr = tf.train.cosine_decay_restarts(lr, epoch, sgdr_decay_epochs, t_mul=sgdr_t_mul, alpha=sgdr_alpha)

        return lr

    def _init_session(self, sess, model):
        w = self._train_params['image_size']
        h = self._train_params['image_size']
        in_ch = 3 
        m = model

        # Do initialization of all variables
        sess.run(tf.global_variables_initializer())

        # Load datasets with defaults
        sess.run([m.train_dataset_init_op, m.pred_dataset_init_op], feed_dict={
            m.ph.train_images: np.zeros((1, w, h, in_ch)),
            m.ph.train_classes: np.zeros((1,)),
            m.ph.pred_images: np.zeros((1, w, h, in_ch)),
            m.ph.pred_classes: np.zeros((1,))
        })

    def _make_session(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        return sess

    def _feed_dataset_to_model(self, images, classes=None, is_train=False):
        m = self._model
        utils.logger.log('Feeding dataset to model...')

        # Mock classes if required
        classes = classes or [0 for _ in range(len(images))]
        
        if is_train:
            self._sess.run(m.train_dataset_init_op, feed_dict={
                m.ph.train_images: images,
                m.ph.train_classes: classes
            })
        else:
            self._sess.run(m.pred_dataset_init_op, feed_dict={
                m.ph.pred_images: images,
                m.ph.pred_classes: classes,
            })

    def _train_model(self, images, classes, dataset_path=None, **knobs):
        num_epochs = knobs['epochs']
        m = self._model
        N = len(images)
        
        self._maybe_feed_dataset_to_model(images, classes, dataset_path=dataset_path, is_train=True)

        # Define plots
        # TODO: Investigate bug where plots for acc and loss are always 1 and 0
        utils.logger.define_plot('Train accuracy over Epochs', ['mean_acc'], 'epoch')
        for (name, _) in self._monitored_values.items():
            utils.logger.define_plot('"{}" Over Time'.format(name), [name])

        log_condition = TimedRepeatCondition()
        for epoch in range(num_epochs):
            utils.logger.log('Running epoch {}...'.format(epoch))

            corrects = []
            itr = self._get_dataset_iterator(N,[m.train_op, m.train_corrects, m.step, m.pred_probs, 
                                            *self._monitored_values.values()], **knobs)
            for  (_, batch_corrects, batch_steps, batch_probs, *values) in itr:
                # To track mean batch accuracy
                corrects.extend(batch_corrects)

                # Periodically, log monitored values
                if log_condition.check():
                    utils.logger.log(step=batch_steps, 
                        **{ name: v for (name, v) in zip(self._monitored_values.keys(), values) })

            # Log mean batch accuracy and epoch
            mean_acc = np.mean(corrects)
            utils.logger.log(epoch=epoch, mean_acc=mean_acc)

    def _evaluate_model(self, images, classes, dataset_path=None, **knobs):
        batch_size = self._knobs['batch_size']
        m = self._model
        N = batch_size if self._knobs['quick_eval'] else len(images)

        self._maybe_feed_dataset_to_model(images, classes, dataset_path=dataset_path)

        corrects = []
        itr = self._get_dataset_iterator(N, [m.pred_corrects], **knobs)
        for (batch_corrects,) in itr:
            corrects.extend(batch_corrects)

        acc = np.mean(corrects)

        return acc

    def _predict_with_model(self, images, **knobs):
        m = self._model
        N = len(images)

        self._maybe_feed_dataset_to_model(images)

        all_probs = []
        itr = self._get_dataset_iterator(N, [m.pred_probs], **knobs)
        for (batch_probs,) in itr:
            all_probs.extend(batch_probs)

        all_probs = np.asarray(all_probs)

        return all_probs

    def _get_dataset_iterator(self, N, run_ops, **knobs):
        batch_size = knobs['batch_size']
        steps_per_epoch = math.ceil(N / batch_size)
        m = self._model

        (normal_arch, reduction_arch) = self._get_fixed_cell_archs(**knobs)
        feed_dict = {
            m.ph.normal_arch: normal_arch,
            m.ph.reduction_arch: reduction_arch
        }

        for itr_step in range(steps_per_epoch):
            results = self._sess.run(run_ops, feed_dict=feed_dict)
            yield results

    def _get_fixed_cell_archs(self, **knobs):
        cell_archs = knobs['cell_archs']
        b = CELL_NUM_BLOCKS
        normal_arch = [cell_archs[(4 * i):(4 * i + 4)] for i in range(b)]
        reduction_arch = [cell_archs[(4 * i):(4 * i + 4)] for i in range(b, b + b)]
        return (normal_arch, reduction_arch)
 
    def _add_aux_head(self, X, in_w, in_h, in_ch, K, is_train):
        pool_ksize = 5
        pool_stride = 3
        conv_ch = 128
        global_conv_ch = 768

        w = in_w
        h = in_h
        ch = in_ch

        # Pool
        with tf.variable_scope('pool'):
            X = tf.nn.relu(X)
            X = tf.nn.avg_pool(X, ksize=(1, pool_ksize, pool_ksize, 1), strides=(1, pool_stride, pool_stride, 1), 
                            padding='VALID')
        w //= pool_stride
        h //= pool_stride

        # Conv 1x1
        with tf.variable_scope('conv_0'):
            X = self._do_conv(X, w, h, ch, conv_ch, filter_size=1, no_reg=True, is_train=is_train)
        ch = conv_ch

        # Global conv
        with tf.variable_scope('conv_1'):
            X = self._do_conv(X, w, h, ch, global_conv_ch, filter_size=w, no_reg=True, is_train=is_train)
        ch = global_conv_ch
        
        # Global pooling
        X = self._add_global_avg_pool(X, w, h, ch)

        # Fully connected
        with tf.variable_scope('fully_connected'):
            aux_logits = self._add_fully_connected(X, (ch,), K, no_reg=True)

        return aux_logits

    def _compute_predictions(self, logits, classes):
        probs = tf.nn.softmax(logits)
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        corrects = tf.equal(preds, classes)
        return (probs, corrects)

    def _compute_loss(self, logits, aux_logits_list, classes, **knobs):
        reg_decay = knobs['reg_decay']
        aux_loss_mul = knobs['aux_loss_mul'] # Multiplier for auxiliary loss

        # Compute sparse softmax cross entropy loss from logits & labels
        log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=classes)
        loss = tf.reduce_mean(log_probs)
        self._mark_for_monitoring('loss', loss)

        # Add regularization loss
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = reg_decay * tf.add_n(reg_losses)
        self._mark_for_monitoring('reg_loss', reg_loss)

        # Add loss from auxiliary logits
        aux_loss = tf.constant(0, dtype=tf.float32)
        for aux_logits in aux_logits_list:
            log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=aux_logits, labels=classes)
            aux_loss += aux_loss_mul * tf.reduce_mean(log_probs)

        total_loss = loss + reg_loss + aux_loss      

        return total_loss

    def _add_global_avg_pool(self, X, in_w, in_h, in_ch):
        X = tf.nn.relu(X)
        X = tf.reduce_mean(X, (1, 2))
        X = tf.reshape(X, (-1, in_ch)) # Sanity shape check
        return X

    def _count_model_parameters(self):
        tf_trainable_vars = tf.trainable_variables()
        num_params = 0
        # utils.logger.log('Model parameters:')
        for var in tf_trainable_vars:
            # utils.logger.log(str(var))
            num_params += np.prod([dim.value for dim in var.get_shape()])

        utils.logger.log('Model has {} parameters'.format(num_params))
        return num_params

    def _add_vars_assign_op(self, vars):
        var_phs = {
            tf_var.name: tf.placeholder(dtype=tf_var.dtype, shape=tf_var.shape)
            for tf_var in vars
        }
        vars_assign_op = tf.group([
            tf.assign(tf_var, ph) 
            for (tf_var, ph) in zip(vars, var_phs.values())
        ], name='vars_assign_op')

        return (var_phs, vars_assign_op)

    ####################################
    # Cells
    ####################################

    def _add_dynamic_cell(self, cell_arch, layers, w, h, block_ch, drop_path_keep_prob, is_train=False):
        b = CELL_NUM_BLOCKS

        # Downsample inputs to have same dimensions as blocks
        with tf.variable_scope('layer_-1_calibrate'):
            layers[-1] = (self._calibrate(*layers[-1], w, h, block_ch, is_train=is_train), w, h, block_ch)
        
        with tf.variable_scope('layer_-2_calibrate'):
            layers[-2] = (self._calibrate(*layers[-2], w, h, block_ch, is_train=is_train), w, h, block_ch)

        cell_inputs = [layers[-2][0] if len(layers) > 1 else layers[-1][0], layers[-1][0]]
        blocks = []
        for bi in range(b):
            with tf.variable_scope('block_{}'.format(bi)):
                idx1 = cell_arch[bi][0]
                op1 = cell_arch[bi][1]
                idx2 = cell_arch[bi][2]
                op2 = cell_arch[bi][3]

                with tf.variable_scope('X1'):
                    X1 = self._add_op_dynamic(cell_inputs, blocks, idx1, op1, w, h, block_ch, is_train=is_train)
                    X1 = self._add_drop_path(X1, drop_path_keep_prob)

                with tf.variable_scope('X2'):
                    X2 = self._add_op_dynamic(cell_inputs, blocks, idx2, op2, w, h, block_ch, is_train=is_train)
                    X2 = self._add_drop_path(X2, drop_path_keep_prob)
                    
                X = tf.add_n([X1, X2])

            blocks.append(X)

        (X, comb_ch) = self._combine_cell_blocks_dynamic(cell_inputs, blocks, cell_arch, w, h, block_ch, is_train)

        X = tf.reshape(X, (-1, w, h, comb_ch)) # Sanity shape check

        layers.append((X, w, h, comb_ch))

    def _add_static_cell(self, cell_arch, layers, w, h, block_ch, drop_path_keep_prob, is_train=False, is_reduction=False):
        b = CELL_NUM_BLOCKS

        # Calibrate inputs as necessary to last input layer's dimensions and add them to hidden states
        cell_inputs = [layers[-2] if len(layers) > 1 else layers[-1], layers[-1]]
        (_, w_inp_last, h_inp_last, _) = cell_inputs[-1]
        for (i, (inp, w_inp, h_inp, ch_inp)) in enumerate(cell_inputs):
            with tf.variable_scope('input_{}_calibrate'.format(i)):
                inp = self._calibrate(inp, w_inp, h_inp, ch_inp, w_inp_last, h_inp_last, block_ch, is_train=is_train)

                # Apply conv 1x1 on last input
                if i == len(cell_inputs) - 1:
                    with tf.variable_scope('input_{}_conv'.format(i)):
                        inp = self._do_conv(inp, w_inp_last, h_inp_last, block_ch, block_ch, is_train=is_train)

            cell_inputs[i] = inp

        blocks = []
        for bi in range(b):
            with tf.variable_scope('block_{}'.format(bi)):
                idx1 = cell_arch[bi][0]
                op1 = cell_arch[bi][1]
                idx2 = cell_arch[bi][2]
                op2 = cell_arch[bi][3]

                with tf.variable_scope('X1'):
                    X1 = self._add_op(cell_inputs, blocks, idx1, op1, w, h, block_ch, 
                                    is_reduction=is_reduction, is_train=is_train)
                    X1 = self._add_drop_path(X1, drop_path_keep_prob)

                with tf.variable_scope('X2'):
                    X2 = self._add_op(cell_inputs, blocks, idx2, op2, w, h, block_ch,
                                    is_reduction=is_reduction, is_train=is_train)
                    X2 = self._add_drop_path(X2, drop_path_keep_prob)
                    
                X = tf.add_n([X1, X2])

            blocks.append(X)

        (X, comb_ch) = self._combine_cell_blocks(cell_inputs, blocks, cell_arch, w, h, block_ch, is_train)

        X = tf.reshape(X, (-1, w, h, comb_ch)) # Sanity shape check

        layers.append((X, w, h, comb_ch))
    
    def _combine_cell_blocks(self, cell_inputs, blocks, cell_arch, w, h, block_ch, is_train=False):
        # Count usage of inputs
        input_use_counts = [0] * len(cell_inputs + blocks)
        for (idx1, _, idx2, _) in cell_arch:
            input_use_counts[idx1] += 1
            input_use_counts[idx2] += 1

        # Concat only unused blocks
        with tf.variable_scope('combine'):
            block_use_counts = input_use_counts[len(cell_inputs):]
            out_blocks = [block for (block, use_count) in zip(blocks, block_use_counts) if use_count == 0]
            comb_ch = len(out_blocks) * block_ch
            X = tf.concat(out_blocks, axis=3)

        return (X, comb_ch)

    def _combine_cell_blocks_dynamic(self, cell_inputs, blocks, cell_arch, w, h, block_ch, is_train=False):
        ni = len(cell_inputs + blocks)
        b = len(blocks)

        # Count usage of inputs
        block_uses = []
        for bi in range(b):
            idx1 = cell_arch[bi][0]
            idx2 = cell_arch[bi][2]
            block_use = tf.one_hot(idx1, ni, dtype=tf.int32) + tf.one_hot(idx2, ni, dtype=tf.int32)
            block_uses.append(block_use)
        block_uses = tf.add_n(block_uses)
        unused_indices = tf.reshape(tf.cast(tf.where(tf.equal(block_uses, 0)), tf.int32), [-1])
        num_out_blocks = tf.size(unused_indices)

        # Select only unused blocks
        with tf.variable_scope('select'):
            stacked_blocks = tf.stack(cell_inputs + blocks)
            out_blocks = tf.gather(stacked_blocks, unused_indices, axis=0)
            out_blocks = tf.transpose(out_blocks, (1, 2, 3, 0, 4))

        # Combine to constant channels
        with tf.variable_scope('combine'):
            W = self._make_var('W', (ni, block_ch * block_ch))
            W = tf.gather(W, unused_indices, axis=0)
            W = tf.reshape(W, (1, 1, num_out_blocks * block_ch, block_ch))
            X = tf.reshape(out_blocks, (-1, w, h, num_out_blocks * block_ch))
            X = tf.nn.relu(X)
            X = tf.nn.conv2d(X, W, (1, 1, 1, 1), padding='SAME')
            X = self._add_batch_norm(X, block_ch, is_train=is_train)

        return (X, block_ch)

    def _add_op(self, cell_inputs, blocks, input_idx, op, w, h, ch, is_reduction=False, is_train=False):
        ni = len(cell_inputs + blocks)
        inputs = cell_inputs + blocks
        op_map = self._get_op_map()
    
        # Just build output for select operation
        X = inputs[input_idx]
        op_no = OPS[op]
        op_method = op_map[op_no]

        # If we were to account for reduction
        if is_reduction and input_idx < len(cell_inputs):
            X = op_method(X, input_idx, ni, w << 1, h << 1, ch, is_reduction=True, is_dynamic=False, is_train=is_train) 
        else:
            X = op_method(X, input_idx, ni, w, h, ch, is_reduction=False, is_dynamic=False, is_train=is_train) 

        return X

    def _add_op_dynamic(self, cell_inputs, blocks, input_idx, op, w, h, ch, is_train=False):
        ni = len(cell_inputs + blocks)
        inputs = tf.stack(cell_inputs + blocks, axis=0)
        op_map = self._get_op_map()

        # Build output for each available operation 
        X = inputs[input_idx]
        op_Xs = []
        for op_no in OPS:
            op_method = op_map[op_no]
            op_X = op_method(X, input_idx, ni, w, h, ch, is_reduction=False, is_dynamic=True, is_train=is_train)
            op_Xs.append(op_X)

        # Stack operation outputs and index by op
        op_Xs = tf.stack(op_Xs)
        X = op_Xs[op]

        return X

    ####################################
    # Block Ops
    ####################################

    def _get_op_map(self):
        # List of all possible operations and their associated numbers
        return {
            0: self._add_separable_conv_3x3_op,
            1: self._add_separable_conv_5x5_op,
            2: self._add_avg_pool_3x3_op,
            3: self._add_max_pool_3x3_op,
            4: self._add_identity_op, 
            5: self._add_separable_conv_7x7_op
        }

    def _add_avg_pool_3x3_op(self, X, input_idx, ni, w, h, ch, is_reduction, is_dynamic, is_train):
        filter_size = 3
        stride = 2 if is_reduction else 1
        with tf.variable_scope('avg_pool_3x3_op'):
            X = tf.nn.avg_pool(X, ksize=(1, filter_size, filter_size, 1), strides=[1, stride, stride, 1], padding='SAME')
        X = tf.reshape(X, (-1, w // stride, h // stride, ch)) # Sanity shape check
        return X
    
    def _add_identity_op(self, X, input_idx, ni, w, h, ch, is_reduction, is_dynamic, is_train):
        stride = 2 if is_reduction else 1
        with tf.variable_scope('identity_op'):
            # If stride > 1, calibrate, else, just return itself
            if stride > 1:
                X = self._calibrate(X, w, h, ch, w // stride, h // stride, ch, is_train=is_train)
        X = tf.reshape(X, (-1, w // stride, h // stride, ch)) # Sanity shape check
        return X
    
    def _add_max_pool_3x3_op(self, X, input_idx, ni, w, h, ch, is_reduction, is_dynamic, is_train):
        filter_size = 3
        stride = 2 if is_reduction else 1
        with tf.variable_scope('max_pool_3x3_op'):
            X = tf.nn.max_pool(X, ksize=(1, filter_size, filter_size, 1), strides=[1, stride, stride, 1], padding='SAME')
        X = tf.reshape(X, (-1, w // stride, h // stride, ch)) # Sanity shape check
        return X
    
    def _add_separable_conv_3x3_op(self, *args, **kwargs):
        return self._add_separable_conv_op(*args, **kwargs, filter_size=3)

    def _add_separable_conv_5x5_op(self, *args, **kwargs):
        return self._add_separable_conv_op(*args, **kwargs, filter_size=5)

    def _add_separable_conv_7x7_op(self, *args, **kwargs):
        return self._add_separable_conv_op(*args, **kwargs, filter_size=7)

    def _add_separable_conv_op(self, X, input_idx, ni, w, h, ch, is_reduction, is_dynamic, is_train, filter_size=3):
        num_stacks = 2
        stride = 2 if is_reduction else 1

        with tf.variable_scope('separable_conv_{}x{}_op'.format(filter_size, filter_size)):
            # For each stack of separable convolution (default of 2)
            for stack_no in range(num_stacks):
                # Only have > 1 stride for first stack 
                stack_stride = stride if stack_no == 0 else 1 
                with tf.variable_scope('stack_{}'.format(stack_no)):
                    W_d = None
                    W_p = None
                    batch_norm_offset = None
                    batch_norm_scale = None
                    if is_dynamic:
                        # Select weights corresponding to input index
                        W_d = self._make_var('W_d', (ni, filter_size, filter_size, ch, 1))
                        W_d = W_d[input_idx] 
                        W_p = self._make_var('W_p', (ni, 1, 1, ch, ch))
                        W_p = W_p[input_idx]
                        batch_norm_offset = self._make_var('batch_norm_offset', (ni, ch), init_constant=0)
                        batch_norm_offset = batch_norm_offset[input_idx]
                        batch_norm_scale = self._make_var('batch_norm_scale', (ni, ch), init_constant=1)
                        batch_norm_scale = batch_norm_scale[input_idx]
                    
                    X = self._do_separable_conv(X, w, h, ch, filter_size=filter_size, stride=stack_stride, 
                                                W_d=W_d, W_p=W_p, no_batch_norm=True)
                    X = self._add_batch_norm(X, ch, offset=batch_norm_offset, scale=batch_norm_scale,
                                            no_moving_average=is_dynamic, is_train=is_train)

        X = tf.reshape(X, (-1, w // stride, h // stride, ch)) # Sanity shape check
        return X

    ####################################
    # Utils
    ####################################

    def _do_cutout(self, image, im_width, im_height, cutout_size):
        mask = tf.ones([cutout_size, cutout_size], dtype=tf.int32)
        start_x = tf.random.uniform(shape=(1,), minval=0, maxval=im_width, dtype=tf.int32)
        start_y = tf.random.uniform(shape=(1,), minval=0, maxval=im_height, dtype=tf.int32)
        mask = tf.pad(mask, [[cutout_size + start_y[0], im_height - start_y[0]],
                            [cutout_size + start_x[0], im_width - start_x[0]]])
        mask = mask[cutout_size: cutout_size + im_height,
                    cutout_size: cutout_size + im_width]
        mask = tf.tile(tf.reshape(mask, (im_height, im_width, 1)), (1, 1, 3))
        image = tf.where(tf.equal(mask, 0), x=image, y=tf.zeros_like(image))
        return image

    def _add_drop_path(self, X, keep_prob):
        with tf.variable_scope('drop_path'):
            batch_size = tf.shape(X)[0]
            noise_shape = (batch_size, 1, 1, 1)
            random_tensor = keep_prob + tf.random_uniform(noise_shape, dtype=tf.float32)
            binary_tensor = tf.floor(random_tensor)
            X = (X / keep_prob) * binary_tensor
        return X

    def _do_conv(self, X, w, h, in_ch, out_ch, filter_size=1, no_relu=False, no_reg=False, is_train=False):
        W = self._make_var('W', (filter_size, filter_size, in_ch, out_ch), no_reg=no_reg)
        if not no_relu:
            X = tf.nn.relu(X)
        X = tf.nn.conv2d(X, W, (1, 1, 1, 1), padding='SAME')
        X = self._add_batch_norm(X, out_ch, is_train=is_train)
        X = tf.reshape(X, (-1, w, h, out_ch)) # Sanity shape check
        return X

    def _do_separable_conv(self, X, w, h, ch, filter_size=3, stride=1, ch_mul=1,
                            no_batch_norm=False, W_d=None, W_p=None,  is_train=False):
        if W_d is None:
            W_d = self._make_var('W_d', (filter_size, filter_size, ch, ch_mul))
        if W_p is None:
            W_p = self._make_var('W_p', (1, 1, ch_mul * ch, ch))
        X = tf.nn.relu(X)
        X = tf.nn.separable_conv2d(X, W_d, W_p, strides=(1, stride, stride, 1), padding='SAME')
        if not no_batch_norm:
            X = self._add_batch_norm(X, ch, is_train=is_train)
        return X

    def _calibrate(self, X, w, h, ch, w_out, h_out, ch_out, is_train=False):
        '''
        Calibrate input of shape (-1, w, h, ch) to (-1, w_out, h_out, ch_out), assuming (w, h) / (w_out, h_out) is power of 2
        '''
        # Downsample with factorized reduction
        downsample_no = 0
        while w > w_out or h > h_out:
            downsample_no += 1
            with tf.variable_scope('downsample_{}x'.format(downsample_no)):
                X = tf.nn.relu(X)
                X = self._add_factorized_reduction(X, w, h, ch, ch_out, is_train=is_train)
                ch = ch_out
                w >>= 1
                h >>= 1

        # If channel counts finally don't match, convert channel counts with 1x1 conv
        if ch != ch_out:
            with tf.variable_scope('convert_conv'):
                X = self._do_conv(X, w, h, ch, ch_out, filter_size=1, is_train=is_train)

        X = tf.reshape(X, (-1, w_out, h_out, ch_out)) # Sanity shape check
        return X

    def _add_fully_connected(self, X, in_shape, out_ch, no_reg=False):
        ch = np.prod(in_shape)
        X = tf.reshape(X, (-1, ch))
        W = self._make_var('W', (ch, out_ch), no_reg=no_reg)
        X = tf.matmul(X, W)
        X = tf.reshape(X, (-1, out_ch)) # Sanity shape check
        return X

    def _add_factorized_reduction(self, X, in_w, in_h, in_ch, out_ch, is_train=False):
        '''
        Output is of shape (in_w // 2, in_h // 2, out_ch)
        '''
        assert in_w % 2 == 0 and in_h % 2 == 0, 'Width & height ({} & {}) must both be even!'.format(in_w, in_h)

        with tf.variable_scope('fac_reduc'):
            # Split area into 2 halves 
            half_1 = tf.nn.avg_pool(X, ksize=(1, 1, 1, 1), strides=(1, 2, 2, 1), padding='VALID')
            shifted_X = tf.pad(X, ((0, 0), (0, 1), (0, 1), (0, 0)))[:, 1:, 1:, :]
            half_2 = tf.nn.avg_pool(shifted_X, ksize=(1, 1, 1, 1), strides=(1, 2, 2, 1), padding='VALID')

            # Apply 1 x 1 convolution to each half separately
            W_half_1 = self._make_var('W_half_1', (1, 1, in_ch, out_ch >> 1))
            X_half_1 = tf.nn.conv2d(half_1, W_half_1, (1, 1, 1, 1), padding='VALID')
            W_half_2 = self._make_var('W_half_2', (1, 1, in_ch, out_ch >> 1))
            X_half_2 = tf.nn.conv2d(half_2, W_half_2, (1, 1, 1, 1), padding='VALID')
            
            # Concat both halves across channels
            X = tf.concat([X_half_1, X_half_2], axis=3)

            # Apply batch normalization
            X = self._add_batch_norm(X, out_ch, is_train=is_train)

        X = tf.reshape(X, (-1, in_w // 2, in_h // 2, out_ch)) # Sanity shape check

        return X

    def _add_batch_norm(self, X, in_ch, decay=0.9, epsilon=1e-5, offset=None, scale=None, is_train=False, 
                        no_moving_average=False):
        with tf.variable_scope('batch_norm'):
            if offset is None:
                offset = self._make_var('offset', (in_ch,), init_constant=0)
            if scale is None:
                scale = self._make_var('scale', (in_ch,), init_constant=1)

            if not no_moving_average:
                moving_mean = self._make_var('moving_mean', (in_ch,), trainable=False, init_constant=0)
                moving_variance = self._make_var('moving_variance', (in_ch,), trainable=False, init_constant=1)

                if is_train:
                    # For training, do batch norm with batch mean & variance
                    # Update moving averages if training
                    (X, mean, variance) = tf.nn.fused_batch_norm(X, scale, offset, epsilon=epsilon, is_training=True)
                    update_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
                    update_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
                    with tf.control_dependencies([update_mean, update_variance]):
                        X = tf.identity(X)
                else:
                    # For prediction, do batch norm with computed moving mean & variance from training
                    # Don't update moving averages if predicting
                    (X, _, _) =  tf.nn.fused_batch_norm(X, scale, offset, mean=moving_mean, variance=moving_variance,
                                                            epsilon=epsilon, is_training=False)
            else:
                (X, _, _) =  tf.nn.fused_batch_norm(X, scale, offset, epsilon=epsilon, is_training=True)

            return X
    
    def _mark_for_monitoring(self, name, value):
        tf.add_to_collection(TF_COLLECTION_MONITORED, tf.identity(value, name))

    def _add_monitoring_of_values(self):
        monitored_values = tf.get_collection(TF_COLLECTION_MONITORED)
        monitored_values = { 
            value.name.split(':')[0]: value # Get rid of ':0' from name
            for value in monitored_values
        }

        for (name, value) in monitored_values.items():
            tf.summary.scalar(name, value)
            
        summary_op = tf.summary.merge_all()

        return (summary_op, monitored_values)

    def _make_var(self, name, shape, dtype=None, no_reg=False, initializer=None, init_constant=None, trainable=True):
        if initializer is None:
            if init_constant is not None:
                initializer = tf.constant_initializer(init_constant, dtype=tf.float32)
            else:
                initializer = tf.contrib.keras.initializers.he_normal()

        # Ensure that name is unique by shape too
        name += '-shape-{}'.format('x'.join([str(x) for x in shape]))

        var = tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable)

        # Add L2 regularization node for trainable var
        if trainable and not no_reg:
            l2_loss = tf.nn.l2_loss(var)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2_loss)
        
        return var

class TimedRepeatCondition():
    def __init__(self, every_secs=60):
        self._every_secs = every_secs
        self._last_trigger_time = datetime.now()
            
    def check(self) -> bool:
        if (datetime.now() - self._last_trigger_time).total_seconds() >= self._every_secs:
            self._last_trigger_time = datetime.now()
            return True
        else:
            return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/cifar10_train.zip', help='Path to train dataset')
    parser.add_argument('--val_path', type=str, default='data/cifar10_val.zip', help='Path to validation dataset')
    parser.add_argument('--test_path', type=str, default='data/cifar10_test.zip', help='Path to test dataset')
    parser.add_argument('--query_path', type=str, default='examples/data/image_classification/cifar10_test_1.png', 
                        help='Path(s) to query image(s), delimited by commas')
    (args, _) = parser.parse_known_args()

    queries = utils.dataset.load_images(args.query_path.split(',')).tolist()
    test_model_class(
        model_file_path=__file__,
        model_class='TfEnas',
        task='IMAGE_CLASSIFICATION',
        dependencies={
            ModelDependency.TENSORFLOW: '1.12.0'
        },
        train_dataset_path=args.train_path,
        val_dataset_path=args.val_path,
        test_dataset_path=args.test_path,
        queries=queries
    )
