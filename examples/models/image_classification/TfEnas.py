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

from rafiki.advisor import Advisor, tune_model
from rafiki.model import utils, BaseModel, IntegerKnob, CategoricalKnob, FloatKnob, \
                            FixedKnob, ListKnob, TrialConfig, SharedParams

_Model = namedtuple('_Model', ['dataset_init_op', 'train_op', 'summary_op', 'images_ph', 'classes_ph', 'is_train_ph', 
        'probs', 'corrects', 'step', 'normal_arch_ph', 'reduction_arch_ph', 'shared_var_phs', 'shared_vars_assign_op'])

ENAS_SEARCH_TRAIN_EVERY_NUM_TRIALS = 300

class TfEnasTrain(BaseModel):
    '''
    Implements the child model of "Efficient Neural Architecture Search via Parameter Sharing" (ENAS) for image classification.
    
    Paper: https://arxiv.org/abs/1802.03268
    '''

    TF_COLLECTION_MONITORED = 'MONITORED'
    CELL_NUM_BLOCKS = 5
    OPS = [0, 1, 2, 3, 4]

    @staticmethod
    def get_knob_config():
        cell_num_blocks = TfEnasTrain.CELL_NUM_BLOCKS
        ops = TfEnasTrain.OPS

        def cell_arch_item(i):
            b = i // 4 # block no
            idx = i % 4 # item index within block
        
            # First half of blocks are for normal cell
            if b < cell_num_blocks:
                if idx in [0, 2]:
                    return CategoricalKnob(list(range(b + 2))) # input index 1/2
                elif idx in [1, 3]:
                    return CategoricalKnob(ops) # op for input 1/2
            
            # Last half of blocks are for reduction cell
            else:
                b -= cell_num_blocks # block no
                if idx in [0, 2]:
                    return CategoricalKnob(list(range(b + 2))) # input index 1/2
                elif idx in [1, 3]:
                    return CategoricalKnob(ops) # op for input 1/2
                    
        return {
            'max_image_size': FixedKnob(32),
            'trial_epochs': FixedKnob(630), # No. of epochs to run trial over
            'ops': FixedKnob(ops),
            'batch_size': FixedKnob(64),
            'learning_rate': FixedKnob(0.05), 
            'initial_block_ch': FixedKnob(36),
            'cell_num_blocks': FixedKnob(cell_num_blocks),
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
            'drop_path_decay_epochs': FixedKnob(630),
            'cutout_size': FixedKnob(0),
            'grad_clip_norm': FixedKnob(5.0),
            'use_aux_head': FixedKnob(False),
            'cell_archs': ListKnob(2 * cell_num_blocks * 4, lambda i: cell_arch_item(i)),
            'use_cell_arch_type': FixedKnob(''), # '' | 'ENAS' | 'NASNET-A',
            'init_params_dir': FixedKnob('') # Params directory to resume training from
        }

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs

    def train(self, dataset_uri):
        num_epochs = self._knobs['trial_epochs']
        init_params_dir = self._knobs['init_params_dir']
        (images, classes, self._train_params) = self._load_dataset(dataset_uri)
          
        if not init_params_dir:
            (self._model, self._graph, self._sess, self._saver, 
                self._monitored_values) = self._build_model()
        else:
            utils.logger.log('Loading parameters from "{}"...'.format(init_params_dir))
            self.load_parameters(init_params_dir)
        
        with self._graph.as_default():
            self._train_summaries = self._train_model(images, classes, num_epochs)
            utils.logger.log('Evaluating model on train dataset...')
            acc = self._evaluate_model(images, classes)
            utils.logger.log('Train accuracy: {}'.format(acc))

    def evaluate(self, dataset_uri):
        (images, classes, _) = self._load_dataset(dataset_uri, train_params=self._train_params)
        with self._graph.as_default():
            utils.logger.log('Evaluating model on validation dataset...')
            acc = self._evaluate_model(images, classes, dataset_uri=dataset_uri)
            utils.logger.log('Validation accuracy: {}'.format(acc))
        return acc

    def predict(self, queries):
        image_size = self._train_params['image_size']
        images = utils.dataset.transform_images(queries, image_size=image_size, mode='RGB')
        with self._graph.as_default():
            probs = self._predict_with_model(images)
        return probs.tolist()

    def save_parameters(self, params_dir):
        # Save model parameters
        model_file_path = os.path.join(params_dir, 'model')
        with self._graph.as_default():
            self._saver.save(self._sess, model_file_path)

        # Save pre-processing params
        train_params_file_path = os.path.join(params_dir, 'train_params.json')
        with open(train_params_file_path, 'w') as f:
            f.write(json.dumps(self._train_params))

        # Dump train summaries
        summaries_dir_path = os.path.join(params_dir, 'summaries')
        os.mkdir(summaries_dir_path)
        writer = tf.summary.FileWriter(summaries_dir_path, self._graph)
        if self._train_summaries is not None:
            for (steps, summary) in self._train_summaries:
                writer.add_summary(summary, steps)

    def load_parameters(self, params_dir):
        # Load pre-processing params
        train_params_file_path = os.path.join(params_dir, 'train_params.json')
        with open(train_params_file_path, 'r') as f:
            json_str = f.read()
            self._train_params = json.loads(json_str)

        # Build model
        (self._model, self._graph, self._sess, 
            self._saver, self._monitored_values) = self._build_model()

        # Load model parameters
        with self._graph.as_default():
            model_file_path = os.path.join(params_dir, 'model')
            self._saver.restore(self._sess, model_file_path)

    ####################################
    # Private methods
    ####################################

    def _load_dataset(self, dataset_uri, train_params=None):
        max_image_size = self._knobs['max_image_size']
        image_size = train_params['image_size'] if train_params is not None else max_image_size

        utils.logger.log('Loading dataset...')    
        dataset = utils.dataset.load_dataset_of_image_files(dataset_uri, max_image_size=image_size, 
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

    def _build_model(self):
        w = self._train_params['image_size']
        h = self._train_params['image_size']
        in_ch = 3 # Num channels of input images
        (normal_arch, reduction_arch) = self._get_arch() # Fixed architecture

        utils.logger.log('Building model...')

        # Create graph
        graph = tf.Graph()
        
        with graph.as_default():
            # Define input placeholders to graph
            images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, w, h, in_ch)) # Images
            classes_ph = tf.placeholder(tf.int32, name='classes_ph', shape=(None,)) # Classes
            is_train_ph = tf.placeholder(tf.bool, name='is_train_ph', shape=()) # Are we training or predicting?

            # Initialize steps variable
            step = self._make_var('step', (), dtype=tf.int32, trainable=False, initializer=tf.initializers.constant(0))

            # Preprocess & do inference
            (X, classes, dataset_init_op) = \
                self._preprocess(images_ph, classes_ph, is_train_ph, w, h, in_ch)
            (logits, aux_logits_list) = self._forward(X, step, normal_arch, reduction_arch, is_train_ph)
            
            # Compute predictions
            (probs, corrects) = self._compute_predictions(logits, classes)

            # Compute training loss
            total_loss = self._compute_loss(logits, aux_logits_list, classes)

            # Optimize training loss
            train_op = self._optimize(total_loss, step)

            # Count model parameters
            model_params_count = self._count_model_parameters()

            # Monitor values
            (summary_op, monitored_values) = self._add_monitoring_of_values()

            # Add saver
            tf_vars = tf.global_variables()
            saver = tf.train.Saver(tf_vars)

            # Make session
            sess = self._make_session()

        model = _Model(dataset_init_op, train_op, summary_op, images_ph, classes_ph, is_train_ph, 
                        probs, corrects, step, None, None, None, None)

        return (model, graph, sess, saver, monitored_values)

    def _forward(self, X, step, normal_arch, reduction_arch, is_train):
        K = self._train_params['K'] # No. of classes
        in_ch = 3 # Num channels of input images
        w = self._train_params['image_size'] # Initial input width
        h = self._train_params['image_size'] # Initial input height
        dropout_keep_prob = self._knobs['dropout_keep_prob']
        L = self._knobs['num_layers'] # Total number of layers
        initial_block_ch = self._knobs['initial_block_ch'] # Initial no. of channels for operations in block
        stem_ch_mul = self._knobs['stem_ch_mul'] # No. of channels for stem convolution as multiple of initial block channels
        use_aux_head = self._knobs['use_aux_head'] # Whether to use auxiliary head
        stem_ch = initial_block_ch * stem_ch_mul
        
        # Layers with reduction cells (otherwise, normal cells)
        reduction_layers = [L // 3, L // 3 * 2 + 1] 

        # Layers with auxiliary heads
        # Aux heads speed up training of good feature repsentations early in the network
        # Add aux heads only if enabled and downsampling width can happen 3 times
        aux_head_layers = []
        if use_aux_head and w % (2 << 3) == 0:
            aux_head_layers.append(reduction_layers[-1] + 1)

        # Stores previous layers. layers[i] = (<previous layer (i - 1) as input to layer i>, <width>, <height>, <channels (tensor)>)
        layers = []
        aux_logits_list = [] # Stores list of logits from aux heads
        block_ch = initial_block_ch

        # "Stem" convolution layer (layer -1)
        with tf.variable_scope('layer_stem'):
            X = self._do_conv(X, w, h, in_ch, stem_ch, is_train, filter_size=3, no_relu=True) # 3x3 convolution
            layers.append((X, w, h, stem_ch))

        # Core layers of cells
        for l in range(L + 2):
            utils.logger.log('Building layer {} of model...'.format(l))
            
            with tf.variable_scope('layer_{}'.format(l)):
                layers_ratio = (l + 1) / (L + 2)
                prev_layers = [layers[-2] if len(layers) > 1 else layers[-1], layers[-1]]
                drop_path_keep_prob = self._get_drop_path_keep_prob(layers_ratio, step, is_train)
                
                # Either add a reduction cell or normal cell
                if l in reduction_layers:
                    with tf.variable_scope('reduction_cell'):
                        block_ch *= 2
                        (X, w, h, ch) = self._add_reduction_cell(reduction_arch, prev_layers, block_ch, is_train,
                                                                drop_path_keep_prob)
                else:
                    with tf.variable_scope('normal_cell'):
                        (X, w, h, ch) = self._add_normal_cell(normal_arch, prev_layers, block_ch, is_train,
                                                            drop_path_keep_prob)

                # Maybe add auxiliary heads 
                if l in aux_head_layers:
                    with tf.variable_scope('aux_head'):
                        aux_logits = self._add_aux_head(X, w, h, ch, K, is_train)
                    aux_logits_list.append(aux_logits)

            layers.append((X, w, h, ch))
    
        # Global average pooling
        (X, w, h, ch) = layers[-1] # Get final layer
        X = self._add_global_avg_pool(X, w, h, ch)

        # Add dropout
        X = tf.cond(is_train, lambda: tf.nn.dropout(X, dropout_keep_prob), lambda: X)

        # Compute logits from X
        with tf.variable_scope('fully_connected'):
            logits = self._add_fully_connected(X, (ch,), K)
        
        return (logits, aux_logits_list)

    def _optimize(self, loss, step):
        opt_momentum = self._knobs['opt_momentum'] # Momentum optimizer momentum
        grad_clip_norm = self._knobs['grad_clip_norm'] # L2 norm to clip gradients by

        # Compute learning rate, gradients
        tf_trainable_vars = tf.trainable_variables()
        lr = self._get_learning_rate(step)
        grads = tf.gradients(loss, tf_trainable_vars)
        self._mark_for_monitoring('lr', lr)

        # Clip gradients
        if grad_clip_norm > 0:
            grads = [tf.clip_by_norm(x, grad_clip_norm) for x in grads]

        # Compute global norm of gradients
        grads_global_norm = tf.global_norm(grads)
        self._mark_for_monitoring('grads_global_norm', grads_global_norm)

        # Init optimizer
        opt = tf.train.MomentumOptimizer(lr, opt_momentum, use_locking=True, use_nesterov=True)
        train_op = opt.apply_gradients(zip(grads, tf_trainable_vars), global_step=step)

        return train_op

    def _preprocess(self, images, classes, is_train, w, h, in_ch):
        batch_size = self._knobs['batch_size']
        cutout_size = self._knobs['cutout_size']
        image_norm_mean = self._train_params['norm_mean']
        image_norm_std = self._train_params['norm_std']

        # Do random crop + horizontal flip for each train image
        def _preprocess_train_image(image):
            image = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
            image = tf.image.random_crop(image, (w, h, in_ch))
            image = tf.image.random_flip_left_right(image)

            if cutout_size > 0:
                image = self._do_cutout(image, w, h, cutout_size)
            
            return image
        
        def _preprocess(image, clazz):
            image = tf.cond(is_train, lambda: _preprocess_train_image(image), lambda: image)
            return (image, clazz)

        # Bulk preprocessing of images
        images = tf.cast(images, tf.float32)
        images = (images - image_norm_mean) / image_norm_std # Normalize
        images = images / 255 # Convert to [0, 1]

        # Create TF dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, classes))
        dataset = dataset.repeat()
        dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=_preprocess, batch_size=batch_size))
        dataset_itr = dataset.make_initializable_iterator()
        (X, classes) = dataset_itr.get_next()
        dataset_init_op = dataset_itr.initializer

        return (X, classes, dataset_init_op)
    
    def _get_drop_path_keep_prob(self, layers_ratio, step, is_train):
        batch_size = self._knobs['batch_size'] 
        N = self._train_params['N']
        drop_path_keep_prob = self._knobs['drop_path_keep_prob'] # Base keep prob for drop path
        drop_path_decay_epochs = self._knobs['drop_path_decay_epochs']
        
        # Decrease keep prob deeper into network
        keep_prob = 1 - layers_ratio * (1 - drop_path_keep_prob)
        
        # Decrease keep prob with increasing steps
        steps_per_epoch = math.ceil(N / batch_size)
        steps_ratio = tf.minimum(((step + 1) / steps_per_epoch) / drop_path_decay_epochs, 1)
        keep_prob = 1 - steps_ratio * (1 - keep_prob)

        # Drop path only during training 
        keep_prob = tf.cond(is_train, 
                    lambda: tf.cast(keep_prob, tf.float32), 
                    lambda: tf.constant(1, dtype=tf.float32))

        # Monitor last layer's keep prob
        if layers_ratio == 1:
            self._mark_for_monitoring('drop_path_keep_prob', keep_prob)

        return keep_prob

    def _get_learning_rate(self, step):
        batch_size = self._knobs['batch_size'] 
        N = self._train_params['N']
        lr = self._knobs['learning_rate'] # Learning rate
        use_sgdr = self._knobs['use_sgdr']
        sgdr_decay_epochs = self._knobs['sgdr_decay_epochs']
        sgdr_alpha = self._knobs['sgdr_alpha'] 
        sgdr_t_mul = self._knobs['sgdr_t_mul']

        # Compute epoch from step
        steps_per_epoch = math.ceil(N / batch_size)
        epoch = step // steps_per_epoch

        if use_sgdr is True:
            # Apply Stoachastic Gradient Descent with Warm Restarts (SGDR)
            lr = tf.train.cosine_decay_restarts(lr, epoch, sgdr_decay_epochs, t_mul=sgdr_t_mul, alpha=sgdr_alpha)

        return lr

    def _make_session(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        return sess

    def _feed_dataset_to_model(self, images, classes=None, dataset_uri=None, is_train=False):
        m = self._model
        
        # Initialize dataset (mock classes if required)
        utils.logger.log('Feeding dataset to model...')
        self._sess.run(m.dataset_init_op, feed_dict={
            m.images_ph: images, 
            m.classes_ph: classes if classes is not None else np.zeros((len(images),)),
            m.is_train_ph: is_train
        })

    def _train_model(self, images, classes, num_epochs, dataset_uri=None):
        m = self._model
        N = len(images)

        self._feed_dataset_to_model(images, classes, dataset_uri=dataset_uri, is_train=True)

        # Define plots for monitored values
        for (name, _) in self._monitored_values.items():
            utils.logger.define_plot('"{}" Over Time'.format(name), [name])

        train_summaries = [] # List of (<steps>, <summary>) collected during training

        log_condition = TimedRepeatCondition()
        for epoch in range(num_epochs):
            utils.logger.log('Running epoch {}...'.format(epoch))
            stepper = self._get_dataset_iterator(N, [m.train_op, m.summary_op, m.corrects, m.step, *self._monitored_values.values()], 
                                                is_train=True)

            # To track mean batch accuracy
            corrects = []
            for (_, summary, batch_corrects, batch_steps, *values) in stepper:
                train_summaries.append((batch_steps, summary))
                corrects.extend(batch_corrects)

                # Periodically, log monitored values
                if log_condition.check():
                    utils.logger.log(step=batch_steps, 
                        **{ name: v for (name, v) in zip(self._monitored_values.keys(), values) })

            # Log mean batch accuracy and epoch
            mean_acc = np.mean(corrects)
            utils.logger.log(epoch=epoch, mean_acc=mean_acc)

        return train_summaries

    def _evaluate_model(self, images, classes, dataset_uri=None):
        m = self._model
        N = len(images)

        self._feed_dataset_to_model(images, classes, dataset_uri=dataset_uri)

        corrects = []
        stepper = self._get_dataset_iterator(N, [m.corrects])
        for (batch_corrects,) in stepper:
            corrects.extend(batch_corrects)

        acc = np.mean(corrects)
        return acc

    def _predict_with_model(self, images):
        m = self._model
        N = len(images)

        self._feed_dataset_to_model(images)

        all_probs = []
        stepper = self._get_dataset_iterator(N, [m.probs])
        for (batch_probs,) in stepper:
            all_probs.extend(batch_probs)
        all_probs = np.asarray(all_probs)

        return all_probs

    def _get_dataset_iterator(self, N, run_ops, is_train=False):
        batch_size = self._knobs['batch_size']
        steps_per_epoch = math.ceil(N / batch_size)
        m = self._model
 
        # Feed architectures if placeholders are present
        if m.normal_arch_ph is not None and m.reduction_arch_ph is not None:
            (normal_arch, reduction_arch) = self._get_arch()
            feed_dict = {
                m.normal_arch_ph: normal_arch,
                m.reduction_arch_ph: reduction_arch,
                m.is_train_ph: is_train
            }
        else:
            feed_dict = {
                m.is_train_ph: is_train
            }

        for _ in range(steps_per_epoch):
            results = self._sess.run(run_ops, feed_dict=feed_dict)
            yield results

    def _get_arch(self):
        use_cell_arch_type = self._knobs['use_cell_arch_type']
        cell_archs = self._knobs['cell_archs']
        num_blocks = self._knobs['cell_num_blocks']
        
        # Use fixed architectures if specified
        if use_cell_arch_type:
            if use_cell_arch_type == 'ENAS':
                cell_archs = [
                    # Normal
                    0, 2, 0, 0, 
                    0, 4, 0, 1, 
                    0, 4, 1, 1, 
                    1, 0, 0, 1, 
                    0, 2, 1, 1,
                    # Reduction
                    1, 0, 1, 0,
                    0, 3, 0, 2,
                    1, 1, 3, 1,
                    1, 0, 0, 4,
                    0, 3, 1, 1
                ]
            elif use_cell_arch_type == 'NASNET-A':
                cell_archs = [
                     # Normal
                    1, 0, 1, 4,
                    0, 0, 1, 1,
                    1, 2, 0, 4,
                    0, 2, 0, 2,
                    0, 1, 0, 0,
                    # Reduction
                    0, 5, 1, 1,
                    1, 3, 0, 5,
                    1, 2, 0, 1,
                    1, 3, 2, 1,
                    2, 2, 3, 4
                ]
            else:
                raise Exception('Invalid `use_cell_arch_type`: "{}"'.format(use_cell_arch_type))

        normal_arch = [cell_archs[(4 * i):(4 * i + 4)] for i in range(num_blocks)]
        reduction_arch = [cell_archs[(4 * i):(4 * i + 4)] for i in range(num_blocks, num_blocks + num_blocks)]
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
            X = self._do_conv(X, w, h, ch, conv_ch, is_train, filter_size=1, no_reg=True)
        ch = conv_ch

        # Global conv
        with tf.variable_scope('conv_1'):
            X = self._do_conv(X, w, h, ch, global_conv_ch, is_train, filter_size=w, no_reg=True)
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

    def _compute_loss(self, logits, aux_logits_list, classes):
        reg_decay = self._knobs['reg_decay']
        aux_loss_mul = self._knobs['aux_loss_mul'] # Multiplier for auxiliary loss

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
        self._mark_for_monitoring('aux_loss', aux_loss)

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

    def _add_reduction_cell(self, cell_arch, inputs, block_ch, is_train, drop_path_keep_prob):
        b = self._knobs['cell_num_blocks'] # no. of blocks
        cell_inputs = []
        blocks = []

        # Initial width & height for this cell as the final input
        (_, w, h, _) = inputs[-1]

        # Calibrate inputs as necessary and add them to hidden states
        for (i, (inp, w_inp, h_inp, ch_inp)) in enumerate(inputs):
            with tf.variable_scope('input_{}_calibrate'.format(i)):
                inp = self._calibrate(inp, w_inp, h_inp, ch_inp, w, h, block_ch, is_train)
                        
                # Apply conv 1x1 on last input
                if i == len(inputs) - 1:
                    with tf.variable_scope('input_{}_conv'.format(i)):
                        inp = self._do_conv(inp, w, h, block_ch, block_ch, is_train)

                cell_inputs.append(inp)

        for bi in range(b):
            with tf.variable_scope('block_{}'.format(bi)):
                idx1 = cell_arch[bi][0]
                op1 = cell_arch[bi][1]
                idx2 = cell_arch[bi][2]
                op2 = cell_arch[bi][3]

                with tf.variable_scope('X1'):
                    X1 = self._apply_reduction_cell_op(idx1, op1, w, h, block_ch, cell_inputs, blocks, is_train)
                    X1 = self._add_drop_path(X1, drop_path_keep_prob)

                with tf.variable_scope('X2'):
                    X2 = self._apply_reduction_cell_op(idx2, op2, w, h, block_ch, cell_inputs, blocks, is_train)
                    X2 = self._add_drop_path(X2, drop_path_keep_prob)
                    
                X = tf.add_n([X1, X2])

            blocks.append(X)

        (X, comb_ch) = self._combine_cell_blocks(cell_inputs, blocks, cell_arch, block_ch)

        X = tf.reshape(X, (-1, w >> 1, h >> 1, comb_ch)) # Sanity shape check

        return (X, w >> 1, h >> 1, comb_ch)

    def _apply_reduction_cell_op(self, idx, op, w, h, ch, cell_inputs, blocks, is_train):
        # Just build output for select input index
        ni = len(cell_inputs)
        if idx < len(cell_inputs):
            X = self._add_op(cell_inputs[idx], op, w, h, ch, is_train, stride=2)
        else:
            X = self._add_op(blocks[idx - ni], op, w >> 1, h >> 1, ch, is_train)
        
        return X

    def _add_normal_cell(self, cell_arch, inputs, block_ch, is_train, drop_path_keep_prob):
        b = self._knobs['cell_num_blocks'] # no. of blocks
        cell_inputs = []
        blocks = [] 

        # Initial width & height for this cell as the final input
        (_, w, h, _) = inputs[-1]

        # Calibrate inputs as necessary and add them to hidden states
        for (i, (inp, w_inp, h_inp, ch_inp)) in enumerate(inputs):
            with tf.variable_scope('input_{}_calibrate'.format(i)):
                inp = self._calibrate(inp, w_inp, h_inp, ch_inp, w, h, block_ch, is_train)
                        
                # Apply conv 1x1 on last input
                if i == len(inputs) - 1:
                    with tf.variable_scope('input_{}_conv'.format(i)):
                        inp = self._do_conv(inp, w, h, block_ch, block_ch, is_train)

                cell_inputs.append(inp)

        for bi in range(b):
            with tf.variable_scope('block_{}'.format(bi)):
                idx1 = cell_arch[bi][0]
                op1 = cell_arch[bi][1]
                idx2 = cell_arch[bi][2]
                op2 = cell_arch[bi][3]

                with tf.variable_scope('X1'):
                    X1 = self._apply_normal_cell_op(idx1, op1, w, h, block_ch, cell_inputs, blocks, is_train)
                    X1 = self._add_drop_path(X1, drop_path_keep_prob)

                with tf.variable_scope('X2'):
                    X2 = self._apply_normal_cell_op(idx2, op2, w, h, block_ch, cell_inputs, blocks, is_train)
                    X2 = self._add_drop_path(X2, drop_path_keep_prob)

                X = tf.add_n([X1, X2])

            blocks.append(X)

        (X, comb_ch) = self._combine_cell_blocks(cell_inputs, blocks, cell_arch, block_ch)

        X = tf.reshape(X, (-1, w, h, comb_ch)) # Sanity shape check

        return (X, w, h, comb_ch)

    def _apply_normal_cell_op(self, idx, op, w, h, ch, cell_inputs, blocks, is_train):
        # Just build output for select input index
        X = (cell_inputs + blocks)[idx]
        X = self._add_op(X, op, w, h, ch, is_train)
        return X

    def _combine_cell_blocks(self, cell_inputs, blocks, cell_arch, block_ch):
        input_use_counts = [0] * len(cell_inputs + blocks)
        for (idx1, op1, idx2, op2) in cell_arch:
            input_use_counts[idx1] += 1
            input_use_counts[idx2] += 1

        # Concats only unused blocks
        block_use_counts = input_use_counts[len(cell_inputs):]
        out_blocks = [block for (block, use_count) in zip(blocks, block_use_counts) if use_count == 0]
        comb_ch = len(out_blocks) * block_ch
        with tf.variable_scope('combine'):
            X = tf.concat(out_blocks, axis=3)

        return (X, comb_ch)

    def _add_op(self, X, op, w, h, ch, is_train, stride=1):
        ops = self._knobs['ops']
        op_map = self._get_op_map()

        # Just build output for select operation
        op_no = ops[op]
        op_method = op_map[op_no]
        X = op_method(X, w, h, ch, is_train, stride) 
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

    def _add_avg_pool_3x3_op(self, X, w, h, ch, is_train, stride):
        filter_size = 3
        with tf.variable_scope('avg_pool_3x3_op'):
            X = tf.nn.avg_pool(X, ksize=(1, filter_size, filter_size, 1), strides=[1, stride, stride, 1], padding='SAME')
        X = tf.reshape(X, (-1, w // stride, h // stride, ch)) # Sanity shape check
        return X
    
    def _add_identity_op(self, X, w, h, ch, is_train, stride):
        # If stride > 1, calibrate, else, just return itself
        with tf.variable_scope('identity_op'):
            if stride > 1:
                X = self._calibrate(X, w, h, ch, w // stride, h // stride, ch, is_train)
        X = tf.reshape(X, (-1, w // stride, h // stride, ch)) # Sanity shape check
        return X
    
    def _add_max_pool_3x3_op(self, X, w, h, ch, is_train, stride):
        filter_size = 3
        with tf.variable_scope('max_pool_3x3_op'):
            X = tf.nn.max_pool(X, ksize=(1, filter_size, filter_size, 1), strides=[1, stride, stride, 1], padding='SAME')
        X = tf.reshape(X, (-1, w // stride, h // stride, ch)) # Sanity shape check
        return X
    
    def _add_separable_conv_3x3_op(self, X, w, h, ch, is_train, stride):
        filter_size = 3
        with tf.variable_scope('separable_conv_3x3_op'):
            X = self._do_separable_conv(X, w, h, ch, is_train, filter_size, stride)
        return X

    def _add_separable_conv_5x5_op(self, X, w, h, ch, is_train, stride):
        filter_size = 5
        with tf.variable_scope('separable_conv_5x5_op'):
            X = self._do_separable_conv(X, w, h, ch, is_train, filter_size, stride)
        return X

    def _add_separable_conv_7x7_op(self, X, w, h, ch, is_train, stride):
        filter_size = 7
        with tf.variable_scope('separable_conv_7x7_op'):
            X = self._do_separable_conv(X, w, h, ch, is_train, filter_size, stride)
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
            X = tf.div(X, keep_prob) * binary_tensor
        return X

    def _do_conv(self, X, w, h, in_ch, out_ch, is_train, filter_size=1, no_relu=False, no_reg=False):
        W = self._make_var('W', (filter_size, filter_size, in_ch, out_ch), no_reg=no_reg)
        if not no_relu:
            X = tf.nn.relu(X)
        X = tf.nn.conv2d(X, W, (1, 1, 1, 1), padding='SAME')
        X = self._add_batch_norm(X, out_ch, is_train)
        X = tf.reshape(X, (-1, w, h, out_ch)) # Sanity shape check
        return X

    def _do_separable_conv(self, X, w, h, ch, is_train, filter_size=3, stride=1, ch_mul=1, num_stacks=2):
        # For each stack of separable convolution (default of 2)
        for stack_no in range(num_stacks):
            # Only have > 1 stride for first stack 
            stack_stride = stride if stack_no == 0 else 1 

            with tf.variable_scope('stack_{}'.format(stack_no)):
                W_d = self._make_var('W_d', (filter_size, filter_size, ch, ch_mul))
                W_p = self._make_var('W_p', (1, 1, ch_mul * ch, ch))
                X = tf.nn.relu(X)
                X = tf.nn.separable_conv2d(X, W_d, W_p, strides=(1, stack_stride, stack_stride, 1), padding='SAME')
                X = self._add_batch_norm(X, ch, is_train)

        X = tf.reshape(X, (-1, w // stride, h // stride, ch)) # Sanity shape check
        return X

    def _calibrate(self, X, w, h, ch, w_out, h_out, ch_out, is_train):
        '''
        Calibrate input of shape (-1, w, h, ch) to (-1, w_out, h_out, ch_out), assuming (w, h) / (w_out, h_out) is power of 2
        '''
        # Downsample with factorized reduction
        downsample_no = 0
        while w > w_out or h > h_out:
            downsample_no += 1
            with tf.variable_scope('downsample_{}x'.format(downsample_no)):
                X = tf.nn.relu(X)
                X = self._add_factorized_reduction(X, w, h, ch, ch_out, is_train)
                ch = ch_out
                w >>= 1
                h >>= 1

        # If channel counts finally don't match, convert channel counts with 1x1 conv
        if ch != ch_out:
            with tf.variable_scope('convert_conv'):
                X = self._do_conv(X, w, h, ch, ch_out, is_train, filter_size=1)

        X = tf.reshape(X, (-1, w_out, h_out, ch_out)) # Sanity shape check
        return X

    def _add_fully_connected(self, X, in_shape, out_ch, no_reg=False):
        ch = np.prod(in_shape)
        X = tf.reshape(X, (-1, ch))
        W = self._make_var('W', (ch, out_ch), no_reg=no_reg)
        X = tf.matmul(X, W)
        X = tf.reshape(X, (-1, out_ch)) # Sanity shape check
        return X

    def _add_factorized_reduction(self, X, in_w, in_h, in_ch, out_ch, is_train):
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
            X = self._add_batch_norm(X, out_ch, is_train)

        X = tf.reshape(X, (-1, in_w // 2, in_h // 2, out_ch)) # Sanity shape check

        return X

    def _add_batch_norm(self, X, in_ch, is_train, decay=0.9, epsilon=1e-5):
        with tf.variable_scope('batch_norm'):
            offset = self._make_var('offset', (in_ch,),
                                    initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            scale = self._make_var('scale', (in_ch,), 
                                    initializer=tf.constant_initializer(1.0, dtype=tf.float32))
            moving_mean = self._make_var('moving_mean', (in_ch,), trainable=False, 
                                        initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            moving_variance = self._make_var('moving_variance', (in_ch,), trainable=False, 
                                            initializer=tf.constant_initializer(1.0, dtype=tf.float32))

            # For training, do batch norm with batch mean & variance
            # For prediction, do batch norm with computed moving mean & variance from training
            (X_pred, _, _) =  tf.nn.fused_batch_norm(X, scale, offset, mean=moving_mean, variance=moving_variance,
                                                    epsilon=epsilon, is_training=False)
            (X_train, mean, variance) = tf.nn.fused_batch_norm(X, scale, offset, epsilon=epsilon, is_training=True)

            # Update moving averages if training
            # Don't update moving averages if predicting
            mean = tf.cond(is_train, lambda: mean, lambda: moving_mean)
            variance = tf.cond(is_train, lambda: variance, lambda: moving_variance)
            update_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
            update_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
            with tf.control_dependencies([update_mean, update_variance]):
                X_train = tf.identity(X_train)

            X = tf.cond(is_train, lambda: X_train, lambda: X_pred) 
            return X
    
    def _mark_for_monitoring(self, name, value):
        tf.add_to_collection(self.TF_COLLECTION_MONITORED, tf.identity(value, name))

    def _add_monitoring_of_values(self):
        monitored_values = tf.get_collection(self.TF_COLLECTION_MONITORED)
        monitored_values = { 
            value.name.split(':')[0]: value # Get rid of ':0' from name
            for value in monitored_values
        }

        for (name, value) in monitored_values.items():
            tf.summary.scalar(name, value)
            
        summary_op = tf.summary.merge_all()

        return (summary_op, monitored_values)

    def _make_var(self, name, shape, dtype=None, no_reg=False, initializer=None, trainable=True):
        if initializer is None:
            initializer = tf.contrib.keras.initializers.he_normal()

        # Ensure that name is unique by shape too
        name += '-shape-{}'.format('x'.join([str(x) for x in shape]))

        var = tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable)

        # Add L2 regularization node for trainable var
        if trainable and not no_reg:
            l2_loss = tf.nn.l2_loss(var)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2_loss)
        
        return var

_ModelMemo = namedtuple('_ModelMemo', ['train_params', 'knobs', 'graph', 'sess', 'saver', 
            'monitored_values', 'model'])

class TfEnasSearch(TfEnasTrain):
    # Memoise across trials to speed up training
    _datasets_memo = {} # { <dataset_uri> -> <dataset> }
    _model_memo = None # of class `_MemoModel`
    _loaded_vars_id_memo = None # ID of vars loaded, if no training has happened
    _loaded_dataset_memo = None # Dataset (<dataset_uri>, <is_train>) loaded into the graph

    @staticmethod
    def get_trial_config(trial_no, total_trials, concurrent_trial_nos):
        train_every_num_trials = ENAS_SEARCH_TRAIN_EVERY_NUM_TRIALS
        T = train_every_num_trials + 1 

        # Every (X + 1) trials, only train 1 epoch for the first trial
        # The other X trials is for training the controller
        is_train_trial = ((trial_no - 1) % T == 0)

        override_knobs = {
            'batch_size': 64,
            'initial_block_ch': 20,
            'reg_decay': 2e-4,
            'num_layers': 6,
            'sgdr_alpha': 0.01,
            'dropout_keep_prob': 0.9,
            'drop_path_decay_epochs': 150
        }
        shared_params = SharedParams.GLOBAL_RECENT 

        if is_train_trial:
            # For a "train" trial, wait until previous trials to finish
            if len(concurrent_trial_nos) > 0:
                return TrialConfig(is_valid=False)
        
            # Set trial epoch to 1 & don't evaluate
            override_knobs.update({ 'trial_epochs': 1 })
            return TrialConfig(override_knobs=override_knobs, 
                                shared_params=shared_params,
                                should_evaluate=False,
                                should_save=False)
        else:
            # For a "eval" trial, just wait until corresponding "train" trial is finished
            train_trial_no = ((trial_no - 1) // T) * T + 1
            if train_trial_no in concurrent_trial_nos:
                return TrialConfig(is_valid=False)

            # Set trial epoch to 0
            override_knobs.update({ 'trial_epochs': 0 })
            return TrialConfig(override_knobs=override_knobs,
                                shared_params=shared_params,
                                should_save=False)

    @staticmethod
    def setup():
        TfEnasSearch._datasets_memo = {}
        TfEnasSearch._model_memo = None
        TfEnasSearch._loaded_vars_id_memo = None

    @staticmethod
    def teardown():
        if TfEnasSearch._model_memo is not None:
            TfEnasSearch._model_memo.sess.close()

        TfEnasSearch._datasets_memo = {}
        TfEnasSearch._model_memo = None
        TfEnasSearch._loaded_vars_id_memo = None

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._shared_params = {}

    def train(self, dataset_uri):
        num_epochs = self._knobs['trial_epochs']
        (images, classes, self._train_params) = self._load_dataset(dataset_uri)
        (self._model, self._graph, self._sess, self._saver, 
            self._monitored_values) = self._build_model()
        
        with self._graph.as_default():
            self._maybe_load_shared_vars(self._shared_params, num_epochs)
            self._train_summaries = []
            if num_epochs == 0:
                # Skipping training
                return

            self._train_summaries = self._train_model(images, classes, num_epochs, dataset_uri=dataset_uri)

    def evaluate(self, dataset_uri):
        (images, classes, _) = self._load_dataset(dataset_uri, train_params=self._train_params)
        with self._graph.as_default():
            utils.logger.log('Evaluating model on validation dataset...')
            acc = self._evaluate_model(images, classes, dataset_uri=dataset_uri)
            utils.logger.log('Validation accuracy: {}'.format(acc))
        return acc

    def set_shared_parameters(self, shared_params):
        self._shared_params = shared_params

    def get_shared_parameters(self):
        num_epochs = self._knobs['trial_epochs']
        if num_epochs > 0:
            with self._graph.as_default():
                shared_tf_vars = self._get_shared_tf_vars()
                values = self._sess.run(shared_tf_vars)

            shared_vars = {
                tf_var.name: np.asarray(value)
                for (tf_var, value)
                in zip(shared_tf_vars, values)
            }

            # Add an ID for diffing
            shared_vars_id = np.random.rand()
            shared_vars['id'] = np.array(shared_vars_id)

            # Memo ID
            TfEnasSearch._loaded_vars_id_memo = shared_vars_id

            return shared_vars
        else:
            return None # No new trained parameters to share

    ####################################
    # Private methods
    ####################################

    def _feed_dataset_to_model(self, images, classes, dataset_uri=None, is_train=False):
        if dataset_uri is None or TfEnasSearch._loaded_dataset_memo != (dataset_uri, is_train):
            # To load new dataset
            super()._feed_dataset_to_model(images, classes, dataset_uri=dataset_uri, is_train=is_train)
            TfEnasSearch._loaded_dataset_memo = (dataset_uri, is_train)
        else:
            # Otherwise, dataset has previously been loaded, so do nothing
            pass

    def _load_dataset(self, dataset_uri, train_params=None):
        # Try to use memoized dataset
        if dataset_uri in TfEnasSearch._datasets_memo:
            dataset = TfEnasSearch._datasets_memo[dataset_uri]
            return dataset

        dataset = super()._load_dataset(dataset_uri, train_params)
        self._datasets_memo[dataset_uri] = dataset
        return dataset

    def _maybe_load_shared_vars(self, shared_vars, num_epochs):
        if len(shared_vars) == 0:
            return

        # If shared vars has been loaded in previous trial, don't bother loading again
        shared_vars_id = shared_vars['id']
        if TfEnasSearch._loaded_vars_id_memo == shared_vars_id:
            pass # Skipping loading of shared variables
        else:
            self._load_shared_vars(shared_vars)

    def _load_shared_vars(self, shared_vars):
        m = self._model

        # Get current values for vars
        shared_tf_vars = self._get_shared_tf_vars()
        values = self._sess.run(shared_tf_vars)
        utils.logger.log('Loading {} / {} shared variables...'.format(len(shared_vars), len(shared_tf_vars)))

        # Build feed dict for op for loading shared params
        # For each param, use current value of param in session if not in shared vars
        var_feeddict = {
            m.shared_var_phs[tf_var.name]: shared_vars[tf_var.name] 
            if tf_var.name in shared_vars else values[i]
            for (i, tf_var) in enumerate(shared_tf_vars)
        }
        
        self._sess.run(m.shared_vars_assign_op, feed_dict=var_feeddict)

    def _get_shared_tf_vars(self):
        return tf.global_variables()

    def _build_model(self):
        # Use memoized graph when possible
        if self._if_model_same(TfEnasSearch._model_memo):
            model_memo = TfEnasSearch._model_memo
            return (model_memo.model, model_memo.graph, model_memo.sess, model_memo.saver, 
                    model_memo.monitored_values)
        
        w = self._train_params['image_size']
        h = self._train_params['image_size']
        cell_num_blocks = self._knobs['cell_num_blocks']
        in_ch = 3 # Num channels of input images

        utils.logger.log('Building model...')

        # Create graph
        graph = tf.Graph()
        
        with graph.as_default():
            # Define input placeholders to graph
            images_ph = tf.placeholder(tf.int8, name='images_ph', shape=(None, w, h, in_ch)) # Images
            classes_ph = tf.placeholder(tf.int32, name='classes_ph', shape=(None,)) # Classes
            is_train_ph = tf.placeholder(tf.bool, name='is_train_ph', shape=()) # Are we training or predicting?
            normal_arch_ph = tf.placeholder(tf.int32, name='normal_arch_ph', shape=(cell_num_blocks, 4))
            reduction_arch_ph = tf.placeholder(tf.int32, name='reduction_arch_ph', shape=(cell_num_blocks, 4))

            # Initialize steps variable
            step = self._make_var('step', (), dtype=tf.int32, trainable=False, initializer=tf.initializers.constant(0))

            # Preprocess & do inference
            (X, classes, dataset_init_op) = \
                self._preprocess(images_ph, classes_ph, is_train_ph, w, h, in_ch)
            (logits, aux_logits_list) = self._forward(X, step, normal_arch_ph, reduction_arch_ph, is_train_ph)
            
            # Compute predictions
            (probs, corrects) = self._compute_predictions(logits, classes)

            # Compute training loss
            total_loss = self._compute_loss(logits, aux_logits_list, classes)

            # Optimize training loss
            train_op = self._optimize(total_loss, step)

            # Count model parameters
            model_params_count = self._count_model_parameters()

            # Monitor values
            (summary_op, monitored_values) = self._add_monitoring_of_values()

            # Add saver
            tf_vars = tf.global_variables()
            saver = tf.train.Saver(tf_vars)

            # Allow loading of shared parameters
            shared_tf_vars = self._get_shared_tf_vars()
            (shared_var_phs, shared_vars_assign_op) = self._add_vars_assign_op(shared_tf_vars)

            # Make session
            sess = self._make_session()

        model = _Model(dataset_init_op, train_op, summary_op, images_ph, classes_ph, is_train_ph, 
                        probs, corrects, step, normal_arch_ph, reduction_arch_ph, shared_var_phs, 
                        shared_vars_assign_op)

        TfEnasSearch._model_memo = _ModelMemo(
            self._train_params, self._knobs, graph, sess,
            saver, monitored_values, model
        )

        return (model, graph, sess, saver, monitored_values)

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

    def _if_model_same(self, model_memo):
        if model_memo is None:
            return False

        # Must have the same train params
        if self._train_params != model_memo.train_params:
            return False

        # Must have the same knobs, except for certain knobs that don't affect model
        ignored_knobs = ['cell_archs', 'trial_count', 'trial_epochs']
        for (name, value) in self._knobs.items():
            if name not in ignored_knobs and value != model_memo.knobs.get(name):
                utils.logger.log('Detected that knob "{}" is different!'.format(name))
                return False
        
        return True

    def _apply_reduction_cell_op(self, idx, op, w, h, ch, cell_inputs, blocks, is_train):
        # Build output for each possible input
        ni = len(cell_inputs)

        # From cell input
        X_ops = []
        for i in range(ni):
            with tf.variable_scope('from_cell_input_{}'.format(i)):
                X_ops.append(self._add_op(cell_inputs[i], op, w, h, ch, is_train, stride=2))

        # From fellow block output as input
        for i in range(len(blocks)):
            with tf.variable_scope('from_block_{}'.format(i)):
                X_ops.append(self._add_op(blocks[i], op, w >> 1, h >> 1, ch, is_train))

        # Condition on input index
        X = tf.case({
            tf.equal(idx, i): lambda: X_op
            for (i, X_op) in enumerate(X_ops)
        }, exclusive=True)

        return X

    def _apply_normal_cell_op(self, idx, op, w, h, ch, cell_inputs, blocks, is_train):
        # Build output for each possible input
        ni = len(cell_inputs)

        # From cell input
        X_ops = []
        for i in range(ni):
            with tf.variable_scope('from_cell_input_{}'.format(i)):
                X_ops.append(self._add_op(cell_inputs[i], op, w, h, ch, is_train))

        # From fellow block output as input
        for i in range(len(blocks)):
            with tf.variable_scope('from_block_{}'.format(i)):
                X_ops.append(self._add_op(blocks[i], op, w, h, ch, is_train))

        # Condition on input index
        X = tf.case({
            tf.equal(idx, i): lambda: X_op
            for (i, X_op) in enumerate(X_ops)
        }, exclusive=True)

        return X

    def _combine_cell_blocks(self, cell_inputs, blocks, cell_arch, block_ch):
        # Concats all blocks
        comb_ch = len(blocks) * block_ch
        with tf.variable_scope('combine'):
            X = tf.concat(blocks, axis=3)

        return (X, comb_ch)

    def _add_op(self, X, op, w, h, ch, is_train, stride=1):
        ops = self._knobs['ops']
        op_map = self._get_op_map()

        # Build output for each available operation 
        op_Xs = []
        for op_no in ops:
            op_method = op_map[op_no]
            op_X = op_method(X, w, h, ch, is_train, stride)
            op_Xs.append(op_X)

        # Stack operation outputs and index by op
        op_Xs = tf.stack(op_Xs)
        X = op_Xs[op]

        return X

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
    parser.add_argument('--mode', type=str, choices=['TRAIN', 'SEARCH'], default='SEARCH')
    parser.add_argument('--total_trials', type=int, default=0) # No. of trials
    parser.add_argument('--num_samples', type=str, default=10) # How many models to sample after training advisor
    (args, _) = parser.parse_known_args()

    if args.mode == 'SEARCH':

        print('Training advisor...')
        knob_config = TfEnasTrain.get_knob_config()
        period = (ENAS_SEARCH_TRAIN_EVERY_NUM_TRIALS + 1)
        total_trials = args.total_trials if args.total_trials > 0 else period * 150
        advisor = Advisor(knob_config) 
        tune_model(
            TfEnasSearch, 
            train_dataset_uri='data/cifar_10_for_image_classification_train.zip',
            val_dataset_uri='data/cifar_10_for_image_classification_val.zip',
            total_trials=total_trials,
            advisor=advisor
        )

        print('Sampling {} models from trained advisor...'.format(args.num_samples))
        for i in range(args.num_samples):
            knobs = advisor.propose()
            print('Knobs {}: {}'.format(i, knobs))

    elif args.mode == 'TRAIN':

        print('Training models...')
        total_trials = args.total_trials if args.total_trials > 0 else 1
        best_knobs = tune_model(
            TfEnasTrain,
            train_dataset_uri='data/cifar_10_for_image_classification_train.zip',
            val_dataset_uri='data/cifar_10_for_image_classification_test.zip',
            total_trials=total_trials
        )

