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
from rafiki.model import BaseModel, InvalidModelParamsException, test_model_class, \
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
            'max_epochs': FixedKnob(10),
            'batch_size': FixedKnob(128),
            'learning_rate': FixedKnob(0.05), 
            'start_ch': FixedKnob(36),
            'l2_reg': FixedKnob(2e-4),
            'dropout_keep_prob': FixedKnob(0.8),
            'opt_momentum': FixedKnob(0.9),
            'sgdr_alpha': FixedKnob(0.02),
            'sgdr_first_decay_steps': FixedKnob(5000),
            'sgdr_t_mul': FixedKnob(2),
            'num_layers': FixedKnob(10),
            'normal_cell_arch': DynamicListKnob(1, 12, cell_arch_block),
            'reduction_cell_arch': DynamicListKnob(1, 12, cell_arch_block)
        }

    @staticmethod
    def get_train_config():
        return {
            'advisor_type': AdvisorType.RANDOM
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

    def evaluate(self, dataset_uri):
        max_image_size = self._knobs['max_image_size']
        norm_mean = self._train_params['norm_mean']
        norm_std = self._train_params['norm_std']

        dataset = utils.dataset.load_dataset_of_image_files(dataset_uri, max_image_size=max_image_size,
                                                            mode='RGB')
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        (images, _, _) = utils.dataset.normalize_images(images, norm_mean, norm_std)
        with self._graph.as_default():
            accuracy = self._evaluate_model(images, classes)
            utils.logger.log('Validation accuracy: {}'.format(accuracy))

        return accuracy

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
        w = self._train_params['image_size']
        h = self._train_params['image_size']
        in_ch = 3 # Num channels of input images
        
        images_ph = tf.placeholder(tf.int8, name='images_ph', shape=(None, w, h, in_ch))
        classes_ph = tf.placeholder(tf.int64, name='classes_ph', shape=(None,))
        is_train = tf.placeholder(tf.bool, name='is_train_ph')
        
        dataset = tf.data.Dataset.from_tensor_slices((images_ph, classes_ph)).batch(N)
        dataset_itr = dataset.make_initializable_iterator()
        (images, classes) = dataset_itr.get_next()

        # Preprocess images
        X = self._preprocess(images, is_train, w, h, in_ch)

        # Do inference
        (probs, preds, logits) = self._inference(X, is_train)

        # Determine all model parameters and count them
        tf_vars = self._get_all_variables()
        model_params_count = self._count_model_parameters(tf_vars)
        utils.logger.log('Model has {} parameters'.format(model_params_count))

        # Compute training loss & accuracy
        loss = self._compute_loss(logits, tf_vars, classes)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, classes), tf.float32))

        # Optimize training loss
        (train_op, steps) = self._optimize(loss, tf_vars)

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

    def _inference(self, X, is_train):
        K = self._train_params['K'] # No. of classes
        in_ch = 3 # Num channels of input images
        w = self._train_params['image_size'] # Current input width
        h = self._train_params['image_size'] # Current input height
        ch = self._knobs['start_ch'] # Current no. of channels
        dropout_keep_prob = self._knobs['dropout_keep_prob']
        L = self._knobs['num_layers'] # Total number of layers

        (normal_arch, reduction_arch) = self._get_arch()
        
        # Stores previous layers. layers[i] = (<previous layer (i - 1) as input to layer i>, <# of downsamples>, channels)
        layers = []

        # "Stem" convolution layer (layer 0)
        X = self._add_stem_conv(X, w, h, in_ch, ch) 
        layers.append((X, 0, ch))

        # Core layers of cells
        for l in range(1, L + 1):
            with tf.variable_scope('layer_{}'.format(l)):
                if l in [L // 3, L // 3 * 2 + 1]:
                    # Add reduction cell
                    (X, ds, ch) = self._add_cell(reduction_arch, layers, l, w, h)

                    # Downsample output
                    X = self._add_factorized_reduction(X, w // 2**ds, h // 2**ds, ch)
                    ds += 1
                    ch *= 2
                else:
                    # Add normal cell
                    (X, ds, ch) = self._add_cell(normal_arch, layers, l, w, h)

                layers.append((X, ds, ch))

        # Global average pooling
        (X, _, _) = layers[-1] # Get final layer
        X = self._add_global_pooling(X, w // 2**ds, h // 2**ds, ch)

        # TODO: Maybe add auxiliary heads

        # Add dropout
        X = tf.case(
            { is_train: lambda: tf.nn.dropout(X, dropout_keep_prob) }, 
            default=lambda: X,
            exclusive=True
        )

        # Compute logits from X
        logits = self._compute_logits(X, (ch,), K)

        # Compute probabilities and predictions
        probs = tf.nn.softmax(logits)
        preds = tf.argmax(logits, axis=1, output_type=tf.int64)
        
        return (probs, preds, logits) 

    def _preprocess(self, images, is_train, w, h, in_ch):
        def preprocess(x):
            x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
            x = tf.image.random_crop(x, [w, h, in_ch])
            x = tf.image.random_flip_left_right(x)

            # TODO: Add CutOut

            return x

        # Only preprocess images during train
        images = tf.case(
            { is_train: (lambda: tf.map_fn(preprocess, images, back_prop=False)) }, 
            default=lambda: images,
            exclusive=True
        )

        X = tf.cast(images, tf.float32)
        return X

    def _optimize(self, loss, tf_vars):
        lr_max = self._knobs['learning_rate'] # Learning rate
        opt_momentum = self._knobs['opt_momentum'] # Momentum optimizer momentum
        alpha = self._knobs['sgdr_alpha'] 
        first_decay_steps = self._knobs['sgdr_first_decay_steps']
        t_mul = self._knobs['sgdr_t_mul']

        # Initialize steps variable
        steps = tf.Variable(0, name='steps', dtype=tf.int32, trainable=False)

        # Apply Stoachastic Gradient Descent with Warm Restarts (SGDR)
        lr = tf.train.cosine_decay_restarts(lr_max, steps, first_decay_steps, t_mul=t_mul, alpha=alpha)

        grads = tf.gradients(loss, tf_vars)
        opt = tf.train.MomentumOptimizer(lr, opt_momentum, use_locking=True, use_nesterov=True)
        train_op = opt.apply_gradients(zip(grads, tf_vars), global_step=steps)

        return (train_op, steps)

    def _init_session(self):
        # (Re-)create session
        if self._sess is not None:
            self._sess.close()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)

    def _train_model(self, images, classes):
        num_epochs = self._knobs['max_epochs']

        utils.logger.log('Available devices: {}'.format(str(device_lib.list_local_devices())))

        self._sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            utils.logger.log(epoch=epoch)

            # Initialize dataset
            self._sess.run(self._init_op, feed_dict={
                self._images_ph: np.asarray(images), 
                self._classes_ph: np.asarray(classes)
            })

            while True:
                try:
                    (loss_batch, acc_batch, steps_batch, _) = self._sess.run(
                        [self._loss, self._acc, self._steps, self._train_op],
                        feed_dict={
                            self._is_train_ph: True
                        }
                    )
                    utils.logger.log(loss=float(loss_batch), acc=float(acc_batch), steps=int(steps_batch))
                except tf.errors.OutOfRangeError:
                    break
        
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
                    self._is_train_ph: False
                })
                probs.extend(probs_batch)
            except tf.errors.OutOfRangeError:
                break

        return np.asarray(probs)

    def _get_all_variables(self):
        tf_vars = [var for var in tf.trainable_variables()]
        return tf_vars

    def _count_model_parameters(self, tf_vars):
        num_params = 0
        print('Model parameters:')
        for var in tf_vars:
            print(var)
            num_params += np.prod([dim.value for dim in var.get_shape()])

        return num_params

    def _compute_logits(self, X, in_shape, num_classes):
        with tf.variable_scope('softmax'):
            ch = np.prod(in_shape)
            X = tf.reshape(X, (-1, ch))
            W = self._create_weights('W', (ch, num_classes))
            X = tf.matmul(X, W)
            y = tf.nn.softmax(X)

        return y

    def _compute_loss(self, logits, tf_vars, classes):
        l2_reg = self._knobs['l2_reg']

        # Compute sparse softmax cross entropy loss from logits & labels
        log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=classes)
        loss = tf.reduce_mean(log_probs)

        # Apply L2 regularization
        l2_losses = [tf.reduce_sum(var ** 2) for var in tf_vars]
        l2_loss = tf.add_n(l2_losses)
        loss += l2_reg * l2_loss

        return loss

    def _add_global_pooling(self, X, w, h, in_ch):
        X = tf.reduce_mean(X, (1, 2))
        X = tf.reshape(X, (-1, in_ch)) # Sanity shape check
        return X

    def _combine_outputs(self, outputs, w, h):
        '''
        Combines the multiple outputs of various shapes.
        Returns (X, ds, ch)
        '''
        # Downsample each output to the maximum downsamples across all outputs
        ds = max([ds for (_, ds, _) in outputs])
        for i in range(len(outputs)):
            (X, ds_x, ch_x) = outputs[i]
            while ds_x < ds:
                with tf.variable_scope('downsample_x{}_{}'.format(i, ds + 1)):
                    X = self._add_factorized_reduction(X, w // 2**ds_x, h // 2**ds_x, ch_x)
                ds_x += 1
                ch_x *= 2
            outputs[i] = (X, ds_x, ch_x)

        # Here, all outputs have the same downsamples, width x height
        # Concat them and conv them to minimum channels across all outputs
        ch = min([ch for (_, _, ch) in outputs])
        comb_ch = sum([ch for (_, _, ch) in outputs])
        with tf.variable_scope('combine_conv'):
            X = tf.concat([X for (X, _, _) in outputs], axis=3)
            X = tf.nn.relu(X)
            W = self._create_weights('W', (1, 1, comb_ch, ch))
            X = tf.nn.conv2d(X, W, strides=(1, 1, 1, 1), padding='SAME')
            X = self._add_batch_norm(X)

        X = tf.reshape(X, (-1, w // 2**ds, h // 2**ds, ch)) # Sanity shape check
        
        return (X, ds, ch)


    def _add_factorized_reduction(self, X, w, h, ch):
        '''
        Output is of shape (w // 2, h // 2, ch * 2)
        '''
        assert w % 2 == 0 and h % 2 == 0, 'Width & height ({} & {}) must both be even!'.format(w, h)

        with tf.variable_scope('fac_reduc'):
            # Split area into 2 halves 
            half_1 = tf.nn.avg_pool(X, ksize=(1, 1, 1, 1), strides=(1, 2, 2, 1), padding='VALID')
            shifted_X = tf.pad(X, ((0, 0), (0, 1), (0, 1), (0, 0)))[:, 1:, 1:, :]
            half_2 = tf.nn.avg_pool(shifted_X, ksize=(1, 1, 1, 1), strides=(1, 2, 2, 1), padding='VALID')

            # Apply 1 x 1 convolution to each half separately
            W_half_1 = self._create_weights('W_half_1', (1, 1, ch, ch))
            X_half_1 = tf.nn.conv2d(half_1, W_half_1, (1, 1, 1, 1), padding='SAME')
            W_half_2 = self._create_weights('W_half_2', (1, 1, ch, ch))
            X_half_2 = tf.nn.conv2d(half_2, W_half_2, (1, 1, 1, 1), padding='SAME')
            
            # Concat both halves across channels
            X = tf.concat([X_half_1, X_half_2], axis=3)

            # Apply batch normalization
            X = self._add_batch_norm(X)

        X = tf.reshape(X, (-1, w // 2, h // 2, ch * 2)) # Sanity shape check

        return X

    # def _add_pooling(self, X, w, h, ch):
    #     assert w % 2 == 0 and h % 2 == 0, 'Width & height ({} & {}) must both be even!'.format(w, H)

    #     with tf.variable_scope('pool'):
    #         X = tf.nn.max_pool(X, ksize=(1, 2, 2, 1), 
    #                         strides=(1, 2, 2, 1), padding='SAME')

    #     X = tf.reshape(X, (-1, w // 2, h // 2, ch)) # Sanity shape check
    #     return X                

    def _add_cell(self, cell_arch, layers, l, w, h):
        b = len(cell_arch) # no. of blocks
        blocks = [] # Stores the list of blocks in this cell as (X, ds, ch)

        def idx_to_layer(idx):
            if idx == 0:
                # Previous layer
                return layers[max(l - 1, 0)]
            elif idx == 1:
                # Previous previous layer
                return layers[max(l - 2, 0)]
            else:
                # Some previous block in cell
                return blocks[idx - 2] 

        for bi in range(b):
            with tf.variable_scope('block_{}'.format(bi)):
                (idx1, op1, indx2, op2) = cell_arch[bi]
                (X1, ds1, ch1) = idx_to_layer(idx1)
                (X2, ds2, ch2) = idx_to_layer(indx2)

                with tf.variable_scope('X1'):
                    X1 = self._add_op(X1, op1, w // 2**ds1, h // 2**ds1, ch1)

                with tf.variable_scope('X2'):
                    X2 = self._add_op(X2, op2, w // 2**ds2, h // 2**ds2, ch2)

                # TODO: Apply drop path to each output

                with tf.variable_scope('combine'):
                    (X, ds_block, ch_block) = self._combine_outputs([(X1, ds1, ch1), (X2, ds2, ch2)], w, h)

            blocks.append((X, ds_block, ch_block))

        # Combine all blocks' outputs
        # TODO: Maybe only concat unused blocks
        with tf.variable_scope('combine'):
            (X, ds, ch) = self._combine_outputs(blocks, w, h)

        return (X, ds, ch)

    def _add_op(self, X, op, w, h, ch):
        ops = {
            0: lambda: self._add_separable_conv_op(X, w, h, ch, filter_size=3),
            1: lambda: self._add_separable_conv_op(X, w, h, ch, filter_size=5),
            2: lambda: self._add_avg_pool_op(X, w, h, ch, filter_size=3),
            3: lambda: self._add_max_pool_op(X, w, h, ch, filter_size=3),
            4: lambda: X, # identity
            5: lambda: self._add_conv_op(X, w, h, ch, filter_size=3),
            6: lambda: self._add_conv_op(X, w, h, ch, filter_size=5)
        }

        X = ops[op]()
        return X

    # def _add_skips(self, X, skips, layers, w, h, ch):
    #     ch_comb = (len(skips) + 1) * ch

    #     with tf.variable_scope('skips'):
    #         # Accumulate all layers' outputs into an array according to `skips`, including X, then concat them
    #         outs = []
    #         for i in range(len(skips)):
    #             outs.append(tf.cond(tf.equal(skips[i], 1),
    #                         lambda: layers[i],
    #                         lambda: tf.zeros_like(layers[i])))
                            
    #         outs.append(X)
    #         X = tf.concat(outs, axis=3)

    #         # Apply stablizing convolution
    #         W = self._create_weights('W', (1, 1, ch_comb, ch))
    #         X = tf.nn.conv2d(X, W, (1, 1, 1, 1), padding='SAME')
    #         X = self._add_batch_norm(X)
    #         X = tf.nn.relu(X)
        
    #     X = tf.reshape(X, (-1, w, h, ch)) # Sanity shape check

    #     return X
    
    # Model's stem layer
    # Converts data to a fixed out channel count of `ch` from 3 RGB channels
    def _add_stem_conv(self, X, w, h, in_ch, ch):
        with tf.variable_scope('stem_conv'):
            W = self._create_weights('W', (3, 3, in_ch, ch))
            X = tf.nn.conv2d(X, W, (1, 1, 1, 1), padding='SAME')
            X = self._add_batch_norm(X)
        X = tf.reshape(X, (-1, w, h, ch)) # Sanity shape check
        return X

    def _add_batch_norm(self, x, decay=0.9, epsilon=1e-5, name='batch_norm'):
        shape = (x.get_shape()[3])

        with tf.variable_scope(name):
            offset = tf.get_variable('offset', shape, 
                                    initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            scale = tf.get_variable('scale', shape, 
                                    initializer=tf.constant_initializer(1.0, dtype=tf.float32))
            moving_mean = tf.get_variable('moving_mean', shape, trainable=False, 
                                        initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            moving_variance = tf.get_variable('moving_variance', shape, trainable=False, 
                                            initializer=tf.constant_initializer(1.0, dtype=tf.float32))
            (x, mean, variance) = tf.nn.fused_batch_norm(x, scale, offset, epsilon=epsilon)
            update_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
            update_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
            
            with tf.control_dependencies([update_mean, update_variance]):
                x = tf.identity(x)

            return x

    def _create_weights(self, name, shape, initializer=None):
        if initializer is None:
            initializer = tf.contrib.keras.initializers.he_normal()
        return tf.get_variable(name, shape, initializer=initializer)

    def _get_arch(self):
        normal_arch = [[0, 2, 0, 0], [0, 4, 0, 1], [0, 4, 1, 1], [1, 0, 0, 1], [0, 2, 1, 1]]
        reduction_arch = [[1, 0, 1, 0], [0, 3, 0, 2], [1, 1, 3, 1], [1, 0, 0, 4], [0, 3, 1, 1]]
        return (normal_arch, reduction_arch)

    # OPS

    def _add_avg_pool_op(self, X, w, h, ch, filter_size):
        with tf.variable_scope('avg_pool_op'):
            X = tf.nn.avg_pool(X, ksize=(1, filter_size, filter_size, 1), strides=[1, 1, 1, 1], padding='SAME')
        X = tf.reshape(X, (-1, w, h, ch)) # Sanity shape check
        return X
    
    def _add_max_pool_op(self, X, w, h, ch, filter_size):
        with tf.variable_scope('max_pool_op'):
            X = tf.nn.max_pool(X, ksize=(1, filter_size, filter_size, 1), strides=[1, 1, 1, 1], padding='SAME')
        X = tf.reshape(X, (-1, w, h, ch)) # Sanity shape check
        return X
    
    def _add_separable_conv_op(self, X, w, h, ch, filter_size, ch_mul=1):
        with tf.variable_scope('separable_conv_op'):
            W_d = self._create_weights('W_d', (filter_size, filter_size, ch, ch_mul))
            W_p = self._create_weights('W_p', (1, 1, ch_mul * ch, ch))
            X = tf.nn.separable_conv2d(X, W_d, W_p, strides=[1, 1, 1, 1], padding='SAME')
            X = self._add_batch_norm(X)
            X = tf.nn.relu(X)
        X = tf.reshape(X, (-1, w, h, ch)) # Sanity shape check
        return X

    def _add_conv_op(self, X, w, h, ch, filter_size):
        with tf.variable_scope('conv_op'):
            W = self._create_weights('W', (filter_size, filter_size, ch, ch))
            X = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
            X = self._add_batch_norm(X)
            X = tf.nn.relu(X)
        X = tf.reshape(X, (-1, w, h, ch)) # Sanity shape check
        return X

if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='TfEnasChild',
        task=TaskType.IMAGE_CLASSIFICATION,
        enable_gpu=True,
        dependencies={
            ModelDependency.TENSORFLOW: '1.12.0'
        },
        train_dataset_uri='data/cifar_10_for_image_classification_train.zip',
        val_dataset_uri='data/cifar_10_for_image_classification_val.zip',
        queries=[
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 7, 0, 37, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 27, 84, 11, 0, 0, 0, 0, 0, 0, 119, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 88, 143, 110, 0, 0, 0, 0, 22, 93, 106, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 53, 129, 120, 147, 175, 157, 166, 135, 154, 168, 140, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 11, 137, 130, 128, 160, 176, 159, 167, 178, 149, 151, 144, 0, 0], 
            [0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 3, 0, 0, 115, 114, 106, 137, 168, 153, 156, 165, 167, 143, 157, 158, 11, 0], 
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 89, 139, 90, 94, 153, 149, 131, 151, 169, 172, 143, 159, 169, 48, 0], 
            [0, 0, 0, 0, 0, 0, 2, 4, 1, 0, 0, 0, 98, 136, 110, 109, 110, 162, 135, 144, 149, 159, 167, 144, 158, 169, 119, 0], 
            [0, 0, 2, 2, 1, 2, 0, 0, 0, 0, 26, 108, 117, 99, 111, 117, 136, 156, 134, 154, 154, 156, 160, 141, 147, 156, 178, 0], 
            [3, 0, 0, 0, 0, 0, 0, 21, 53, 92, 117, 111, 103, 115, 129, 134, 143, 154, 165, 170, 154, 151, 154, 143, 138, 150, 165, 43], 
            [0, 0, 23, 54, 65, 76, 85, 118, 128, 123, 111, 113, 118, 127, 125, 139, 133, 136, 160, 140, 155, 161, 144, 155, 172, 161, 189, 62], 
            [0, 68, 94, 90, 111, 114, 111, 114, 115, 127, 135, 136, 143, 126, 127, 151, 154, 143, 148, 125, 162, 162, 144, 138, 153, 162, 196, 58], 
            [70, 169, 129, 104, 98, 100, 94, 97, 98, 102, 108, 106, 119, 120, 129, 149, 156, 167, 190, 190, 196, 198, 198, 187, 197, 189, 184, 36], 
            [16, 126, 171, 188, 188, 184, 171, 153, 135, 120, 126, 127, 146, 185, 195, 209, 208, 255, 209, 177, 245, 252, 251, 251, 247, 220, 206, 49], 
            [0, 0, 0, 12, 67, 106, 164, 185, 199, 210, 211, 210, 208, 190, 150, 82, 8, 0, 0, 0, 178, 208, 188, 175, 162, 158, 151, 11], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        ]
    )
