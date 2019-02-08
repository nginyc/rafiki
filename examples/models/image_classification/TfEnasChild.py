import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
from tensorflow.python.training import moving_averages
import json
import os
import tempfile
import numpy as np
import base64

from rafiki.config import APP_MODE
from rafiki.model import BaseModel, InvalidModelParamsException, tune_model, \
                        IntegerKnob, CategoricalKnob, FloatKnob, FixedKnob, \
                        ListKnob, DynamicListKnob, utils
from rafiki.constants import TaskType, ModelDependency, AdvisorType

class TfEnasChild(BaseModel):
    '''
    Implements the child model of "Efficient Neural Architecture Search via Parameter Sharing" (ENAS) for image classification.
    
    Paper: https://arxiv.org/abs/1802.03268
    '''
    @staticmethod
    def get_knob_config():
        def cell_arch_block(i):
            return ListKnob(4, items=[
                CategoricalKnob(list(range(i + 2))), # index 1
                CategoricalKnob([0, 1, 2, 3, 4]), # op 1
                CategoricalKnob(list(range(i + 2))), # index 2
                CategoricalKnob([0, 1, 2, 3, 4]) # op 2
            ])

        return {
            'max_image_size': FixedKnob(32),
            'max_epochs': FixedKnob(126),
            'batch_size': FixedKnob(64),
            'learning_rate': FixedKnob(0.05), 
            'start_ch': FixedKnob(36),
            'l2_reg': FixedKnob(2e-4),
            'dropout_keep_prob': FixedKnob(0.8),
            'opt_momentum': FixedKnob(0.9),
            'use_sgdr': FixedKnob(True),
            'sgdr_alpha': FixedKnob(0.02),
            'sgdr_decay_epochs': FixedKnob(1),
            'sgdr_t_mul': FixedKnob(2),  
            'num_layers': FixedKnob(15), 
            'aux_loss_mul': FixedKnob(0.4),
            'drop_path_keep_prob': FixedKnob(0.6),
            'cutout_size': FixedKnob(16)
            # 'normal_cell_arch': DynamicListKnob(1, 12, cell_arch_block),
            # 'reduction_cell_arch': DynamicListKnob(1, 12, cell_arch_block)
        }

    @staticmethod
    def get_train_config():
        return {
            'advisor_type': AdvisorType.SKOPT
        }

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        self._graph = tf.Graph()
        self._sess = None
        
    def train(self, dataset_uri):
        max_image_size = self._knobs['max_image_size']

        dataset = utils.dataset.load_dataset_of_image_files(dataset_uri, max_image_size=max_image_size, 
                                                            mode='RGB')
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        (images, norm_mean, norm_std) = utils.dataset.normalize_images(images)
        self._train_params = {
            'image_size': dataset.image_size,
            'K': dataset.classes,
            'norm_mean': norm_mean,
            'norm_std': norm_std
        }

        with self._graph.as_default():
            self._build_model()
            self._init_session()
            self._train_model(images, classes)
            utils.logger.log('Evaluating model on train dataset...')
            acc = self._evaluate_model(images, classes)
            utils.logger.log('Train accuracy: {}'.format(acc))

    def evaluate(self, dataset_uri):
        max_image_size = self._knobs['max_image_size']
        norm_mean = self._train_params['norm_mean']
        norm_std = self._train_params['norm_std']

        dataset = utils.dataset.load_dataset_of_image_files(dataset_uri, max_image_size=max_image_size,
                                                            mode='RGB')
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        (images, _, _) = utils.dataset.normalize_images(images, norm_mean, norm_std)
        with self._graph.as_default():
            utils.logger.log('Evaluating model on validation dataset...')
            acc = self._evaluate_model(images, classes)
            utils.logger.log('Validation accuracy: {}'.format(acc))

        return acc

    def predict(self, queries):
        image_size = self._train_params['image_size']
        norm_mean = self._train_params['norm_mean']
        norm_std = self._train_params['norm_std']

        images = utils.dataset.transform_images(queries, image_size=image_size, mode='RGB')
        (images, _, _) = utils.dataset.normalize_images(images, norm_mean, norm_std)
        with self._graph.as_default():
            probs = self._predict_with_model(images)
                
        return probs.tolist()

    def destroy(self):
        if self._sess is not None:
            self._sess.close()

    def save_parameters(self, params_dir):
        # Save model
        model_file_path = os.path.join(params_dir, 'model')
        saver = tf.train.Saver(self._tf_vars)
        saver.save(self._sess, model_file_path) 

        # Save pre-processing params
        train_params_file_path = os.path.join(params_dir, 'train_params.json')
        with open(train_params_file_path, 'w') as f:
            f.write(json.dumps(self._train_params))

    def load_parameters(self, params_dir):
        # Load pre-processing params
        train_params_file_path = os.path.join(params_dir, 'train_params.json')
        with open(train_params_file_path, 'r') as f:
            json_str = f.read()
            self._train_params = json.loads(json_str)

        # Build model
        self._build_model()

        # Load model parameters
        self._init_session()
        model_file_path = os.path.join(params_dir, 'model')
        saver = tf.train.Saver(self._tf_vars)
        saver.restore(self._sess, model_file_path)

    def _build_model(self):
        N = self._knobs['batch_size'] 
        num_epochs = self._knobs['max_epochs']
        w = self._train_params['image_size']
        h = self._train_params['image_size']
        in_ch = 3 # Num channels of input images
        
        images_ph = tf.placeholder(tf.int8, name='images_ph', shape=(None, w, h, in_ch))
        classes_ph = tf.placeholder(tf.int64, name='classes_ph', shape=(None,))
        is_train = tf.placeholder(tf.bool, name='is_train_ph')
        epoch = tf.placeholder(tf.int64, name='epoch_ph')

        epochs_ratio = epoch / num_epochs
        
        dataset = tf.data.Dataset.from_tensor_slices((images_ph, classes_ph)).batch(N)
        dataset_itr = dataset.make_initializable_iterator()
        (images, classes) = dataset_itr.get_next()

        # Preprocess images
        X = self._preprocess(images, is_train, w, h, in_ch)

        # Do inference
        (probs, preds, logits, aux_logits_list) = self._inference(X, epochs_ratio, is_train)

        # Determine all model parameters and count them
        tf_vars = self._get_all_variables()
        model_params_count = self._count_model_parameters(tf_vars)
        utils.logger.log('Model has {} parameters'.format(model_params_count))

        # Compute training loss & accuracy
        loss = self._compute_loss(logits, aux_logits_list, tf_vars, classes)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, classes), tf.float32))

        # Optimize training loss
        (train_op, steps) = self._optimize(loss, tf_vars, epoch)

        self._loss = loss
        self._acc = acc
        self._probs = probs
        self._init_op = dataset_itr.initializer
        self._train_op = train_op
        self._steps = steps
        self._images_ph = images_ph
        self._classes_ph = classes_ph
        self._tf_vars = tf_vars
        self._is_train_ph = is_train
        self._epoch_ph = epoch

    def _inference(self, X, epochs_ratio, is_train):
        K = self._train_params['K'] # No. of classes
        in_ch = 3 # Num channels of input images
        w = self._train_params['image_size'] # Initial input width
        h = self._train_params['image_size'] # Initial input height
        ch = self._knobs['start_ch'] # Initial no. of channels
        dropout_keep_prob = self._knobs['dropout_keep_prob']
        L = self._knobs['num_layers'] # Total number of layers

        (normal_arch, reduction_arch) = self._get_arch()
        
        reduction_layers = [L // 3, L // 3 * 2 + 1] # Layers with reduction cells (otherwise, normal cells)
        aux_head_layers = [reduction_layers[-1] + 1] # Layers with auxiliary heads

        # Stores previous layers. layers[i] = (<previous layer (i - 1) as input to layer i>, <# of downsamples>)
        layers = []
        ds = 0 # Current no. of downsamples
        aux_logits_list = [] # Stores list of logits from aux heads

        # "Stem" convolution layer (layer 0)
        X = self._add_stem_conv(X, w, h, in_ch, ch) 
        layers.append((X, ds))

        # Core layers of cells
        for l in range(1, L + 1):
            with tf.variable_scope('layer_{}'.format(l)):
                layers_ratio = l / (L + 1)
                prev_layers = [layers[-1], layers[-2] if len(layers) > 1 else layers[-1]]

                # Either add a reduction cell or normal cell
                if l in reduction_layers:
                    with tf.variable_scope('reduction_cell'):
                        X = self._add_reduction_cell(reduction_arch, prev_layers, w, h, ch, ds, 
                                                    epochs_ratio, layers_ratio, is_train)
                        ds += 1
                else:
                    with tf.variable_scope('normal_cell'):
                        X = self._add_normal_cell(normal_arch, prev_layers, w, h, ch, ds, 
                                                epochs_ratio, layers_ratio, is_train)

                # Maybe add auxiliary heads 
                if l in aux_head_layers:
                    with tf.variable_scope('aux_head'):
                        aux_logits = self._add_aux_head(X, w >> ds, h >> ds, ch << ds, K)
                    aux_logits_list.append(aux_logits)

            layers.append((X, ds))

        # Global average pooling
        (X, _) = layers[-1] # Get final layer
        X = self._add_global_pooling(X, w >> ds, h >> ds, ch << ds)

        # Add dropout
        X = tf.case(
            { is_train: lambda: tf.nn.dropout(X, dropout_keep_prob) }, 
            default=lambda: X,
            exclusive=True
        )

        # Compute logits from X
        X = self._add_fully_connected(X, (ch << ds,), K)
        logits = tf.nn.softmax(X)

        # Compute probabilities and predictions
        probs = tf.nn.softmax(logits)
        preds = tf.argmax(logits, axis=1, output_type=tf.int64)
        
        return (probs, preds, logits, aux_logits_list) 

    def _preprocess(self, images, is_train, w, h, in_ch):
        cutout_size = self._knobs['cutout_size']

        def preprocess(image):
            # Do random crop + horizontal flip
            image = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
            image = tf.image.random_crop(image, [w, h, in_ch])
            image = tf.image.random_flip_left_right(image)

            if cutout_size > 0:
                image = self._do_cutout(image, w, h, cutout_size)

            return image

        # Only preprocess images during train
        images = tf.case(
            { is_train: (lambda: tf.map_fn(preprocess, images, back_prop=False)) }, 
            default=lambda: images,
            exclusive=True
        )

        X = tf.cast(images, tf.float32)
        return X

    def _optimize(self, loss, tf_vars, epoch):
        opt_momentum = self._knobs['opt_momentum'] # Momentum optimizer momentum

        # Initialize steps variable
        steps = tf.Variable(0, name='steps', dtype=tf.int32, trainable=False)

        # Compute learning rate, gradients
        lr = self._get_learning_rate(epoch)
        grads = tf.gradients(loss, tf_vars)

        # Init optimizer
        opt = tf.train.MomentumOptimizer(lr, opt_momentum, use_locking=True, use_nesterov=True)
        train_op = opt.apply_gradients(zip(grads, tf_vars), global_step=steps)

        return (train_op, steps)

    def _get_learning_rate(self, epoch):
        lr = self._knobs['learning_rate'] # Learning rate
        use_sgdr = self._knobs['use_sgdr']
        sgdr_decay_epochs = self._knobs['sgdr_decay_epochs']
        sgdr_alpha = self._knobs['sgdr_alpha'] 
        sgdr_t_mul = self._knobs['sgdr_t_mul']

        if use_sgdr is True:
            # Apply Stoachastic Gradient Descent with Warm Restarts (SGDR)
            lr = tf.train.cosine_decay_restarts(lr, epoch, sgdr_decay_epochs, t_mul=sgdr_t_mul, alpha=sgdr_alpha)

        return lr

    def _init_session(self):
        # (Re-)create session
        if self._sess is not None:
            self._sess.close()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)

    def _train_model(self, images, classes):
        num_epochs = self._knobs['max_epochs']
        # best_loss_patience_epochs = 10 # No. of epochs where there should be an decrease in best batch loss, otherwise training stops 

        utils.logger.log('Available devices: {}'.format(str(device_lib.list_local_devices())))

        # best_loss, best_loss_patience_count = float('inf'), 0

        self._sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            utils.logger.log('Running epoch {}...'.format(epoch))

            # Initialize dataset
            self._sess.run(self._init_op, feed_dict={
                self._images_ph: np.asarray(images), 
                self._classes_ph: np.asarray(classes)
            })

            avg_batch_loss, avg_batch_acc, n = 0, 0, 0
            while True:
                try:
                    (batch_loss, batch_acc, steps, _) = self._sess.run(
                        [self._loss, self._acc, self._steps, self._train_op],
                        feed_dict={
                            self._is_train_ph: True,
                            self._epoch_ph: epoch
                        }
                    )

                    # Update batch no, loss & acc
                    avg_batch_acc = avg_batch_acc * (n / (n + 1)) + batch_acc * (1 / (n + 1))
                    avg_batch_loss = avg_batch_loss * (n / (n + 1)) + batch_loss * (1 / (n + 1))
                    n += 1
                    
                except tf.errors.OutOfRangeError:
                    break

            # # Determine whether training should stop due to patience
            # if avg_batch_loss < best_loss:
            #     utils.logger.log('Best batch loss so far: {}'.format(avg_batch_loss))
            #     best_loss = avg_batch_loss
            #     best_loss_patience_count = 0
            # else:
            #     best_loss_patience_count += 1
            #     if best_loss_patience_count >= best_loss_patience_epochs:
            #         utils.logger.log('Batch loss has not increased for {} epochs'.format(best_loss_patience_epochs))
            #         break

            utils.logger.log(epoch=epoch,
                            avg_batch_loss=float(avg_batch_loss), 
                            avg_batch_acc=float(avg_batch_acc), 
                            steps=int(steps))
        
    def _evaluate_model(self, images, classes):
        probs = self._predict_with_model(images)
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == np.asarray(classes))
        return acc

    def _predict_with_model(self, images):
        # Initialize dataset (mock classes)
        self._sess.run(self._init_op, feed_dict={
            self._images_ph: np.asarray(images), 
            self._classes_ph: np.zeros((len(images),))
        })

        probs = []
        while True:
            try:
                probs_batch = self._sess.run(self._probs, {
                    self._is_train_ph: False,
                    self._epoch_ph: 0
                })
                probs.extend(probs_batch)
            except tf.errors.OutOfRangeError:
                break

        return np.asarray(probs)

    def _get_arch(self):
        normal_arch = [[0, 2, 0, 0], [0, 4, 0, 1], [0, 4, 1, 1], [1, 0, 0, 1], [0, 2, 1, 1]]
        reduction_arch = [[1, 0, 1, 0], [0, 3, 0, 2], [1, 1, 3, 1], [1, 0, 0, 4], [0, 3, 1, 1]]
        return (normal_arch, reduction_arch)

 
    def _add_aux_head(self, X, in_w, in_h, in_ch, K):
        pool_ksize = 5
        pool_stride = 2
        conv_out_ch = in_ch

        # Pool
        with tf.variable_scope('pool'):
            X = tf.nn.relu(X)
            X = tf.nn.avg_pool(X, ksize=(1, pool_ksize, pool_ksize, 1), strides=(1, pool_stride, pool_stride, 1), 
                            padding='SAME')
        
        # Conv 1x1
        with tf.variable_scope('conv'):
            W = self._create_weights('W', (1, 1, in_ch, conv_out_ch)) 
            X = tf.nn.conv2d(X, W, strides=(1, 1, 1, 1), padding='SAME')
            X = self._add_batch_norm(X, conv_out_ch)
            X = tf.nn.relu(X)
        
        # Global pooling
        X = self._add_global_pooling(X, in_w // pool_stride, in_h // pool_stride, conv_out_ch)

        # Fully connected
        X = self._add_fully_connected(X, (conv_out_ch,), K)
        logits = tf.nn.softmax(X)

        return logits

    def _add_drop_path(self, X, epochs_ratio, layers_ratio):
        keep_prob = self._knobs['drop_path_keep_prob']
        
        # Decrease keep prob deeper into network
        keep_prob = 1 - layers_ratio * (1 - keep_prob)
        
        # Decrease keep prob with increasing epochs 
        keep_prob = 1 - epochs_ratio * (1 - keep_prob)

        # Apply dropout
        keep_prob = tf.to_float(keep_prob)
        batch_size = tf.shape(X)[0]
        noise_shape = (batch_size, 1, 1, 1)
        random_tensor = keep_prob + tf.random_uniform(noise_shape, dtype=tf.float32)
        binary_tensor = tf.floor(random_tensor)
        X = tf.div(X, keep_prob) * binary_tensor

        return X

    def _compute_loss(self, logits, aux_logits_list, tf_vars, classes):
        l2_reg = self._knobs['l2_reg']
        aux_loss_mul = self._knobs['aux_loss_mul'] # Multiplier for auxiliary loss

        # Compute sparse softmax cross entropy loss from logits & labels
        log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=classes)
        total_loss = tf.reduce_mean(log_probs)

        # Apply L2 regularization
        l2_losses = [tf.reduce_sum(var ** 2) for var in tf_vars]
        l2_loss = tf.add_n(l2_losses)
        total_loss += l2_reg * l2_loss

        # Add loss from auxiliary logits
        for aux_logits in aux_logits_list:
            log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=aux_logits, labels=classes)
            aux_loss = tf.reduce_mean(log_probs)
            total_loss += aux_loss_mul * aux_loss

        return total_loss

    def _add_global_pooling(self, X, in_w, in_h, in_ch):
        X = tf.reduce_mean(X, (1, 2))
        X = tf.reshape(X, (-1, in_ch)) # Sanity shape check
        return X

    def _downsample(self, X, ds_x, ds, w, h, ch):
        '''
        Downsample X from `ds_x` downsamples to `ds` downsamples
        '''
        for ds_i in range(ds_x + 1, ds + 1):
            with tf.variable_scope('downsample_{}x'.format(ds_i)):
                X = self._add_factorized_reduction(X, w >> ds_x, h >> ds_x, ch << ds_x)

        X = tf.reshape(X, (-1, w >> ds, h >> ds, ch << ds)) # Sanity shape check
        return X

    def _count_model_parameters(self, tf_vars):
        num_params = 0
        for var in tf_vars:
            num_params += np.prod([dim.value for dim in var.get_shape()])

        return num_params

    def _add_reduction_cell(self, cell_arch, inputs, w, h, ch, ds, 
                            epochs_ratio, layers_ratio, is_train):
        b = len(cell_arch) # no. of blocks
        hidden_states = [] # Stores hidden states for this cell, which includes blocks

        # Downsample previous layers as necessary and add them to hidden states
        for (i, (inp, ds_inp)) in enumerate(inputs):
            with tf.variable_scope('input_{}_downsample'.format(i)):
                inp = self._downsample(inp, ds_inp, ds, w, h, ch)
                hidden_states.append(inp)

        for bi in range(b):
            with tf.variable_scope('block_{}'.format(bi)):
                (idx1, op1, idx2, op2) = cell_arch[bi]
                X1 = hidden_states[idx1]
                X2 = hidden_states[idx2]

                with tf.variable_scope('X1'):
                    X1 = self._add_op(X1, op1, w >> ds, h >> ds, ch << ds)
                    X1 = tf.case(
                        { is_train: (lambda: self._add_drop_path(X1, epochs_ratio, layers_ratio)) }, 
                        default=lambda: X1,
                        exclusive=True
                    )

                with tf.variable_scope('X2'):
                    X2 = self._add_op(X2, op2, w >> ds, h >> ds, ch << ds)
                    X2 = tf.case(
                        { is_train: (lambda: self._add_drop_path(X2, epochs_ratio, layers_ratio)) }, 
                        default=lambda: X2,
                        exclusive=True
                    )
                    
                X = tf.add_n([X1, X2])

            hidden_states.append(X)

        # Combine all hidden states
        # TODO: Maybe only concat unused blocks
        comb_ch = len(hidden_states) * (ch << ds)
        out_ch = ch << ds
        with tf.variable_scope('combine'):
            X = self._concat(hidden_states, w >> ds, h >> ds, comb_ch, out_ch)

        # Downsample output once
        with tf.variable_scope('downsample'):
            X = self._downsample(X, ds, ds + 1, w, h, ch)

        return X

    def _add_normal_cell(self, cell_arch, inputs, w, h, ch, ds, 
                        epochs_ratio, layers_ratio, is_train):
        b = len(cell_arch) # no. of blocks
        hidden_states = [] # Stores hidden states for this cell, which includes blocks

        # Downsample previous layers as necessary and add them to hidden states
        for (i, (inp, ds_inp)) in enumerate(inputs):
            with tf.variable_scope('input_{}_downsample'.format(i)):
                inp = self._downsample(inp, ds_inp, ds, w, h, ch)
                hidden_states.append(inp)

        for bi in range(b):
            with tf.variable_scope('block_{}'.format(bi)):
                (idx1, op1, idx2, op2) = cell_arch[bi]
                X1 = hidden_states[idx1]
                X2 = hidden_states[idx2]

                with tf.variable_scope('X1'):
                    X1 = self._add_op(X1, op1, w >> ds, h >> ds, ch << ds)
                    X1 = tf.case(
                        { is_train: (lambda: self._add_drop_path(X1, epochs_ratio, layers_ratio)) }, 
                        default=lambda: X1,
                        exclusive=True
                    )

                with tf.variable_scope('X2'):
                    X2 = self._add_op(X2, op2, w >> ds, h >> ds, ch << ds)
                    X2 = tf.case(
                        { is_train: (lambda: self._add_drop_path(X2, epochs_ratio, layers_ratio)) }, 
                        default=lambda: X2,
                        exclusive=True
                    )

                X = tf.add_n([X1, X2])

            hidden_states.append(X)

        # Combine all hidden states
        # TODO: Maybe only concat unused blocks
        comb_ch = len(hidden_states) * (ch << ds)
        out_ch = ch << ds
        with tf.variable_scope('combine'):
            X = self._concat(hidden_states, w >> ds, h >> ds, comb_ch, out_ch)

        return X

    def _add_op(self, X, op, w, h, ch):
        ops = {
            0: lambda: self._add_separable_conv_op(X, w, h, ch, filter_size=3, stride=1),
            1: lambda: self._add_separable_conv_op(X, w, h, ch, filter_size=5, stride=1),
            2: lambda: self._add_avg_pool_op(X, w, h, ch, filter_size=3, stride=1),
            3: lambda: self._add_max_pool_op(X, w, h, ch, filter_size=3, stride=1),
            4: lambda: self._add_max_pool_op(X, w, h, ch, filter_size=1, stride=1), # identity
            5: lambda: self._add_conv_op(X, w, h, ch, filter_size=3,stride=1),
            6: lambda: self._add_conv_op(X, w, h, ch, filter_size=5, stride=1)
        }

        X = ops[op]()
        return X

    # def _add_reduction_op(self, X, op, w, h, ch):
    #     ops = {
    #         0: lambda: self._add_separable_conv_op(X, w, h, ch, filter_size=3, stride=2),
    #         1: lambda: self._add_separable_conv_op(X, w, h, ch, filter_size=5, stride=2),
    #         2: lambda: self._add_avg_pool_op(X, w, h, ch, filter_size=3, stride=2),
    #         3: lambda: self._add_max_pool_op(X, w, h, ch, filter_size=3, stride=2),
    #         4: lambda: self._add_max_pool_op(X, w, h, ch, filter_size=1, stride=2), # identity
    #         5: lambda: self._add_conv_op(X, w, h, ch, filter_size=3, stride=2),
    #         6: lambda: self._add_conv_op(X, w, h, ch, filter_size=5, stride=2)
    #     }

    #     X = ops[op]()
    #     return X
    
    def _add_stem_conv(self, X, w, h, in_ch, ch):
        '''
        Model's stem layer
        Converts data to a fixed out channel count of `ch` from 3 RGB channels
        '''
        with tf.variable_scope('stem_conv'):
            W = self._create_weights('W', (3, 3, in_ch, ch))
            X = tf.nn.conv2d(X, W, (1, 1, 1, 1), padding='SAME')
            X = self._add_batch_norm(X, ch)
        X = tf.reshape(X, (-1, w, h, ch)) # Sanity shape check
        return X


    ####################################
    # Block Ops
    ####################################

    def _add_avg_pool_op(self, X, w, h, ch, filter_size, stride):
        with tf.variable_scope('avg_pool_op'):
            X = tf.nn.avg_pool(X, ksize=(1, filter_size, filter_size, 1), strides=[1, stride, stride, 1], padding='SAME')
        X = tf.reshape(X, (-1, w // stride, h // stride, ch)) # Sanity shape check
        return X
    
    def _add_max_pool_op(self, X, w, h, ch, filter_size, stride):
        with tf.variable_scope('max_pool_op'):
            X = tf.nn.max_pool(X, ksize=(1, filter_size, filter_size, 1), strides=[1, stride, stride, 1], padding='SAME')
        X = tf.reshape(X, (-1, w // stride, h // stride, ch)) # Sanity shape check
        return X
    
    def _add_separable_conv_op(self, X, w, h, ch, filter_size, stride, ch_mul=1):
        with tf.variable_scope('separable_conv_op'):
            W_d = self._create_weights('W_d', (filter_size, filter_size, ch, ch_mul))
            W_p = self._create_weights('W_p', (1, 1, ch_mul * ch, ch))
            X = tf.nn.separable_conv2d(X, W_d, W_p, strides=[1, stride, stride, 1], padding='SAME')
            X = self._add_batch_norm(X, ch)
            X = tf.nn.relu(X)
        X = tf.reshape(X, (-1, w // stride, h // stride, ch)) # Sanity shape check
        return X

    def _add_conv_op(self, X, w, h, ch, filter_size, stride):
        with tf.variable_scope('conv_op'):
            W = self._create_weights('W', (filter_size, filter_size, ch, ch))
            X = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='SAME')
            X = self._add_batch_norm(X, ch)
            X = tf.nn.relu(X)
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
    
    def _concat(self, inputs, in_w, in_h, in_ch, out_ch):
        '''
        Concatenate inputs (which must have the same `in_w` x `in_h`) across a combined `in_ch` channels to produce `out_ch` channels
        '''
        with tf.variable_scope('concat'):
            X = tf.concat(inputs, axis=3)
            X = tf.nn.relu(X)
            W = self._create_weights('W', (1, 1, in_ch, out_ch))
            X = tf.nn.conv2d(X, W, strides=(1, 1, 1, 1), padding='SAME')
            X = self._add_batch_norm(X, out_ch)

        X = tf.reshape(X, (-1, in_w, in_h, out_ch)) # Sanity shape check
        return X

    def _add_fully_connected(self, X, in_shape, out_ch):
        with tf.variable_scope('fully_connected'):
            ch = np.prod(in_shape)
            X = tf.reshape(X, (-1, ch))
            W = self._create_weights('W', (ch, out_ch))
            X = tf.matmul(X, W)

        X = tf.reshape(X, (-1, out_ch)) # Sanity shape check
        return X

    def _add_factorized_reduction(self, X, in_w, in_h, in_ch):
        '''
        Output is of shape (in_w // 2, in_h // 2, in_ch * 2)
        '''
        assert in_w % 2 == 0 and in_h % 2 == 0, 'Width & height ({} & {}) must both be even!'.format(in_w, in_h)

        with tf.variable_scope('fac_reduc'):
            # Split area into 2 halves 
            half_1 = tf.nn.avg_pool(X, ksize=(1, 1, 1, 1), strides=(1, 2, 2, 1), padding='VALID')
            shifted_X = tf.pad(X, ((0, 0), (0, 1), (0, 1), (0, 0)))[:, 1:, 1:, :]
            half_2 = tf.nn.avg_pool(shifted_X, ksize=(1, 1, 1, 1), strides=(1, 2, 2, 1), padding='VALID')

            # Apply 1 x 1 convolution to each half separately
            W_half_1 = self._create_weights('W_half_1', (1, 1, in_ch, in_ch))
            X_half_1 = tf.nn.conv2d(half_1, W_half_1, (1, 1, 1, 1), padding='SAME')
            W_half_2 = self._create_weights('W_half_2', (1, 1, in_ch, in_ch))
            X_half_2 = tf.nn.conv2d(half_2, W_half_2, (1, 1, 1, 1), padding='SAME')
            
            # Concat both halves across channels
            X = tf.concat([X_half_1, X_half_2], axis=3)

            # Apply batch normalization
            X = self._add_batch_norm(X, in_ch * 2)

        X = tf.reshape(X, (-1, in_w // 2, in_h // 2, in_ch * 2)) # Sanity shape check

        return X

    def _add_batch_norm(self, X, in_ch, decay=0.9, epsilon=1e-5):
        with tf.variable_scope('batch_norm'):
            offset = tf.get_variable('offset', (in_ch,), 
                                    initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            scale = tf.get_variable('scale', (in_ch,), 
                                    initializer=tf.constant_initializer(1.0, dtype=tf.float32))
            moving_mean = tf.get_variable('moving_mean', (in_ch,), trainable=False, 
                                        initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            moving_variance = tf.get_variable('moving_variance', (in_ch,), trainable=False, 
                                            initializer=tf.constant_initializer(1.0, dtype=tf.float32))
            (X, mean, variance) = tf.nn.fused_batch_norm(X, scale, offset, epsilon=epsilon)
            update_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
            update_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
            
            with tf.control_dependencies([update_mean, update_variance]):
                X = tf.identity(X)

            return X

    def _create_weights(self, name, shape, initializer=None):
        if initializer is None:
            initializer = tf.contrib.keras.initializers.he_normal()
        return tf.get_variable(name, shape, initializer=initializer)
    
    def _get_all_variables(self):
        tf_vars = [var for var in tf.trainable_variables()]
        return tf_vars

if __name__ == '__main__':
    tune_model(
        TfEnasChild, 
        train_dataset_uri='data/cifar_10_for_image_classification_train.zip',
        val_dataset_uri='data/cifar_10_for_image_classification_train.zip',
        num_trials=100,
        enable_gpu=True
    )