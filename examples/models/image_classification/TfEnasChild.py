import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow import keras
from tensorflow.python.client import device_lib
from tensorflow.python.training import moving_averages
import json
import os
import tempfile
import numpy as np
import base64

from rafiki.config import APP_MODE
from rafiki.advisor import IntegerKnob, CategoricalKnob, FloatKnob, FixedKnob, ListKnob, Metadata, MetadataKnob
from rafiki.model import utils, tune_model, BaseModel
from rafiki.constants import TaskType, ModelDependency

class TfEnasChild(BaseModel):
    '''
    Implements the child model of "Efficient Neural Architecture Search via Parameter Sharing" (ENAS) for image classification.
    
    Paper: https://arxiv.org/abs/1802.03268
    '''

    TF_COLLECTION_SHARED = 'SHARED'

    @staticmethod
    def get_knob_config():
        def cell_arch_item(i, num_blocks):
            b = i // 4 # block no
            idx = i % 4 # item index within block
        
            # First half of blocks are for normal cell
            if b < num_blocks:
                if idx in [0, 2]:
                    return CategoricalKnob(list(range(b + 2))) # input index 1/2
                elif idx in [1, 3]:
                    return CategoricalKnob([0, 1, 2, 3, 4, 5]) # op for input 1/2
            
            # Last half of blocks are for reduction cell
            else:
                b -= num_blocks # block no
                if idx in [0, 2]:
                    return CategoricalKnob(list(range(b + 2))) # input index 1/2
                elif idx in [1, 3]:
                    return CategoricalKnob([0, 1, 2, 3, 4, 5]) # op for input 1/2
                    
        return {
            'trial_count': MetadataKnob(Metadata.TRIAL_COUNT),
            'total_trials': MetadataKnob(Metadata.TOTAL_TRIALS),
            'max_image_size': FixedKnob(32),
            'trial_epochs': FixedKnob(1),
            'batch_size': FixedKnob(64),
            'learning_rate': FixedKnob(0.05), 
            'initial_block_ch': FixedKnob(36),
            'stem_ch': FixedKnob(108),
            'reg_decay': FixedKnob(2e-4),
            'dropout_keep_prob': FixedKnob(0.8),
            'opt_momentum': FixedKnob(0.9),
            'use_sgdr': FixedKnob(True),
            'sgdr_alpha': FixedKnob(0.002),
            'sgdr_decay_epochs': FixedKnob(10),
            'sgdr_t_mul': FixedKnob(2),  
            'num_layers': FixedKnob(15), 
            'aux_loss_mul': FixedKnob(0.4),
            'drop_path_keep_prob': FixedKnob(0.6),
            'cutout_size': FixedKnob(0),
            'grad_clip_norm': FixedKnob(0),
            'log_monitored_values_steps': FixedKnob(100),
            'cell_archs': ListKnob(2 * 5 * 4, lambda i: cell_arch_item(i, 5)),
            'use_cell_arch_type': FixedKnob('') # '' | 'ENAS' | 'NASNET-A'
        }

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        self._graph = tf.Graph()
        self._sess = None
        
    def train(self, dataset_uri, shared_params):
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
            if len(shared_params) > 0:
                self._load_shareable_vars(shared_params)
            self._add_logging()
            self._train_model(images, classes)
            utils.logger.log('Evaluating model on train dataset...')
            acc = self._evaluate_model(images, classes)
            utils.logger.log('Train accuracy: {}'.format(acc))
            return self._get_shareable_vars()

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
        # Save model parameters
        model_file_path = os.path.join(params_dir, 'model')
        saver = tf.train.Saver(self._tf_vars)
        saver.save(self._sess, model_file_path)

        # Save pre-processing params
        train_params_file_path = os.path.join(params_dir, 'train_params.json')
        with open(train_params_file_path, 'w') as f:
            f.write(json.dumps(self._train_params))

        # Dump train summaries
        summaries_dir_path = os.path.join(params_dir, 'summaries')
        os.mkdir(summaries_dir_path)
        writer = tf.summary.FileWriter(summaries_dir_path, self._graph)
        for summary in self._train_summaries:
            writer.add_summary(summary)

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

    def _get_shareable_vars(self):
        shareable_tf_vars = tf.get_collection(self.TF_COLLECTION_SHARED)
        values = self._sess.run(shareable_tf_vars)
        shareable_vars = {
            tf_var.name: value
            for (tf_var, value)
            in zip(shareable_tf_vars, values)
        }
        return shareable_vars

    def _load_shareable_vars(self, shareable_vars):
        shareable_tf_vars = tf.get_collection(self.TF_COLLECTION_SHARED)
        shared_vars = 0
        var_assigns = []
        for tf_var in shareable_tf_vars:
            if tf_var.name in shareable_vars:
                var_assign = tf_var.assign(shareable_vars[tf_var.name])
                var_assigns.append(var_assign)
                shared_vars += 1

        utils.logger.log('Restoring {} / {} shareable variables...'.format(shared_vars, len(shareable_tf_vars)))
        self._sess.run(var_assigns)

    def _build_model(self):
        N = self._knobs['batch_size'] 
        w = self._train_params['image_size']
        h = self._train_params['image_size']
        total_trials = self._knobs['total_trials']
        trial_epochs = self._knobs['trial_epochs']

        total_epochs = trial_epochs * total_trials
        in_ch = 3 # Num channels of input images
        
        # To add values to monitor
        self._monitored_values = {}
        
        images_ph = tf.placeholder(tf.int8, name='images_ph', shape=(None, w, h, in_ch))
        classes_ph = tf.placeholder(tf.int32, name='classes_ph', shape=(None,))
        is_train = tf.placeholder(tf.bool, name='is_train_ph', shape=())
        epoch = tf.placeholder(tf.int32, name='epoch_ph', shape=())

        epochs_ratio = epoch / total_epochs
        
        dataset = tf.data.Dataset.from_tensor_slices((images_ph, classes_ph)).batch(N)
        dataset_itr = dataset.make_initializable_iterator()
        (images, classes) = dataset_itr.get_next()

        # Preprocess images
        X = self._preprocess(images, is_train, w, h, in_ch)

        # Do inference
        (probs, preds, logits, aux_logits_list) = self._inference(X, epochs_ratio, is_train)
        
        # Compute training loss & accuracy
        tf_vars = self._get_all_variables()
        (total_loss, loss, reg_loss, aux_loss) = self._compute_loss(logits, aux_logits_list, tf_vars, classes)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, classes), tf.float32))

        # Optimize training loss
        (train_op, steps, lr, grads_global_norm) = self._optimize(total_loss, tf_vars, epoch)

        self._monitored_values.update({
            'loss': loss,
            'aux_loss': aux_loss,
            'reg_loss': reg_loss,
            'lr': lr,
            'grads_global_norm': grads_global_norm
        })
        self._probs = probs
        self._init_op = dataset_itr.initializer
        self._train_op = train_op
        self._images_ph = images_ph
        self._classes_ph = classes_ph
        self._tf_vars = tf_vars
        self._is_train_ph = is_train
        self._epoch_ph = epoch
        self._acc = acc
        self._steps = steps

    def _add_logging(self):
        # Log available devices 
        utils.logger.log('Available devices: {}'.format(str(device_lib.list_local_devices())))

        # Count model parameters
        model_params_count = self._count_model_parameters(self._tf_vars)
        utils.logger.log('Model has {} parameters'.format(model_params_count))

        # Make summaries 
        for (name, value) in self._monitored_values.items():
            tf.summary.scalar(name, value)

        self._summary_op = tf.summary.merge_all()

    def _inference(self, X, epochs_ratio, is_train):
        K = self._train_params['K'] # No. of classes
        in_ch = 3 # Num channels of input images
        w = self._train_params['image_size'] # Initial input width
        h = self._train_params['image_size'] # Initial input height
        dropout_keep_prob = self._knobs['dropout_keep_prob']
        L = self._knobs['num_layers'] # Total number of layers
        stem_ch = self._knobs['stem_ch'] # No. of channels for stem convolution
        initial_block_ch = self._knobs['initial_block_ch'] # Initial no. of channels for operations in block

        (normal_arch, reduction_arch) = self._get_arch()
        
        # Layers with reduction cells (otherwise, normal cells)
        reduction_layers = [L // 3, L // 3 * 2 + 1] 

        # Layers with auxiliary heads
        # Aux heads speed up training of good feature repsentations early in the network
        # Add aux heads only if downsampling width can happen 3 times
        aux_head_layers = [reduction_layers[-1] + 1] if w % (2 << 3) == 0 else []

        # Stores previous layers. layers[i] = (<previous layer (i - 1) as input to layer i>, <width>, <height>, <channels>)
        layers = []
        aux_logits_list = [] # Stores list of logits from aux heads

        # "Stem" convolution layer (layer -1)
        with tf.variable_scope('layer_stem'):
            X = self._do_conv(X, w, h, in_ch, stem_ch, filter_size=3) # 3x3 convolution
            layers.append((X, w, h, stem_ch))

        # Core layers of cells
        block_ch = initial_block_ch
        for l in range(1, L + 1):
            with tf.variable_scope('layer_{}'.format(l)):
                layers_ratio = l / (L + 1)
                prev_layers = [layers[-2] if len(layers) > 1 else layers[-1], layers[-1]]
                drop_path_keep_prob = tf.cond(is_train, 
                                        lambda: self._get_drop_path_keep_prob(layers_ratio, epochs_ratio), 
                                        lambda: tf.constant(1, dtype=tf.float32))

                # Either add a reduction cell or normal cell
                if l in reduction_layers:
                    with tf.variable_scope('reduction_cell'):
                        block_ch *= 2
                        (X, w, h, ch) = self._add_reduction_cell(reduction_arch, prev_layers, block_ch,
                                                                drop_path_keep_prob)
                else:
                    with tf.variable_scope('normal_cell'):
                        (X, w, h, ch) = self._add_normal_cell(normal_arch, prev_layers, block_ch,
                                                            drop_path_keep_prob)

                # Maybe add auxiliary heads 
                if l in aux_head_layers:
                    with tf.variable_scope('aux_head'):
                        aux_logits = self._add_aux_head(X, w, h, ch, K)
                    aux_logits_list.append(aux_logits)

            layers.append((X, w, h, ch))
    
            # Track final layer's drop path keep prob
            if l == L:
                self._monitored_values['final_drop_path_keep_prob'] = drop_path_keep_prob
    
        # Global average pooling
        (X, w, h, ch) = layers[-1] # Get final layer
        X = self._add_global_pooling(X, w, h, ch)

        # Add dropout
        X = tf.cond(is_train, lambda: tf.nn.dropout(X, dropout_keep_prob), lambda: X)

        # Compute logits from X
        X = self._add_fully_connected(X, (ch,), K)
        logits = tf.nn.softmax(X)

        # Compute probabilities and predictions
        probs = tf.nn.softmax(logits)
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        
        return (probs, preds, logits, aux_logits_list) 

    def _get_drop_path_keep_prob(self, layers_ratio, epochs_ratio):
        drop_path_keep_prob = self._knobs['drop_path_keep_prob'] # Base keep prob for drop path

        # Decrease keep prob deeper into network
        keep_prob = 1 - layers_ratio * (1 - drop_path_keep_prob)
        
        # Decrease keep prob with increasing epochs 
        keep_prob = 1 - epochs_ratio * (1 - keep_prob)

        return tf.cast(keep_prob, tf.float32)

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
        images = tf.cond(is_train, 
                        lambda: tf.map_fn(preprocess, images, back_prop=False),
                        lambda: images)

        X = tf.cast(images, tf.float32)
        return X

    def _optimize(self, loss, tf_vars, epoch):
        opt_momentum = self._knobs['opt_momentum'] # Momentum optimizer momentum
        grad_clip_norm = self._knobs['grad_clip_norm'] # L2 norm to clip gradients by

        # Initialize steps variable
        steps = tf.Variable(0, name='steps', dtype=tf.int32, trainable=False)

        # Compute learning rate, gradients
        lr = self._get_learning_rate(epoch)
        grads = tf.gradients(loss, tf_vars)

        # Clip gradients
        if grad_clip_norm > 0:
            grads = [tf.clip_by_norm(x, grad_clip_norm) for x in grads]

        # Compute global norm of gradients
        grads_global_norm = tf.global_norm(grads)

        # Init optimizer
        opt = tf.train.MomentumOptimizer(lr, opt_momentum, use_locking=True, use_nesterov=True)
        train_op = opt.apply_gradients(zip(grads, tf_vars), global_step=steps)

        return (train_op, steps, lr, grads_global_norm)

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
        trial_count = self._knobs['trial_count']
        trial_epochs = self._knobs['trial_epochs']
        log_monitored_values_steps = self._knobs['log_monitored_values_steps']
        prev_epochs = trial_count * trial_epochs # No. of epochs that has run for past trials

        train_summaries = []

        self._sess.run(tf.global_variables_initializer())

        for trial_epoch in range(trial_epochs):
            epoch = trial_epoch + prev_epochs
            utils.logger.log('Running epoch {} (trial epoch {})...'.format(epoch, trial_epoch))

            # Initialize dataset
            self._sess.run(self._init_op, feed_dict={
                self._images_ph: np.asarray(images), 
                self._classes_ph: np.asarray(classes)
            })

            # To track monitored values & accuracy
            (monitored_names, monitored_values) = zip(*self._monitored_values.items())
            accs = []

            while True:
                try:
                    (_, summary, acc, steps, *values) = self._sess.run(
                        [self._train_op, self._summary_op, self._acc, self._steps, *monitored_values],
                        feed_dict={
                            self._is_train_ph: True,
                            self._epoch_ph: epoch
                        }
                    )

                    train_summaries.append(summary)
                    accs.append(acc)
                    
                    # Periodically, log monitored values
                    if steps % log_monitored_values_steps == 0:
                        utils.logger.log(steps=steps, **{ k: v for (k, v) in zip(monitored_names, values) })
                    
                except tf.errors.OutOfRangeError:
                    break

            # Log mean batch accuracy and epoch
            mean_acc = np.mean(accs)
            utils.logger.log(epoch=epoch, mean_acc=mean_acc)

        self._train_summaries = train_summaries
            
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
        cell_archs = self._knobs['cell_archs']
        use_cell_arch_type = self._knobs['use_cell_arch_type']

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
                raise ValueError('Invalid cell architecture type: "{}"'.format(use_cell_arch_type))

        num_blocks = 5
        normal_arch = [cell_archs[(4 * i):(4 * i + 4)] for i in range(num_blocks)]
        reduction_arch = [cell_archs[(4 * i):(4 * i + 4)] for i in range(num_blocks, num_blocks + num_blocks)]
        return (normal_arch, reduction_arch)
 
    def _add_aux_head(self, X, in_w, in_h, in_ch, K):
        pool_ksize = 5
        pool_stride = 2
        conv_ch = 128
        global_conv_ch = 768

        w = in_w
        h = in_h
        ch = in_ch

        # Pool
        with tf.variable_scope('pool'):
            X = tf.nn.relu(X)
            X = tf.nn.avg_pool(X, ksize=(1, pool_ksize, pool_ksize, 1), strides=(1, pool_stride, pool_stride, 1), 
                            padding='SAME')
        w //= pool_stride
        h //= pool_stride

        # Conv 1x1
        with tf.variable_scope('conv_0'):
            X = self._do_conv(X, w, h, ch, conv_ch, filter_size=1, do_relu=True, no_reg=True)
        ch = conv_ch

        # Global conv
        with tf.variable_scope('conv_1'):
            X = self._do_conv(X, w, h, ch, global_conv_ch, filter_size=w, do_relu=True, no_reg=True)
        ch = global_conv_ch
        
        # Global pooling
        X = self._add_global_pooling(X, w, h, ch)

        # Fully connected
        X = self._add_fully_connected(X, (ch,), K, no_reg=True)
        aux_logits = tf.nn.softmax(X)

        return aux_logits

    def _compute_loss(self, logits, aux_logits_list, tf_vars, classes):
        reg_decay = self._knobs['reg_decay']
        aux_loss_mul = self._knobs['aux_loss_mul'] # Multiplier for auxiliary loss

        # Compute sparse softmax cross entropy loss from logits & labels
        log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=classes)
        loss = tf.reduce_mean(log_probs)

        # Add regularization loss
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = reg_decay * tf.add_n(reg_losses)

        # Add loss from auxiliary logits
        aux_loss = tf.constant(0, dtype=tf.float32)
        for aux_logits in aux_logits_list:
            log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=aux_logits, labels=classes)
            aux_loss += aux_loss_mul * tf.reduce_mean(log_probs)

        total_loss = loss + reg_loss + aux_loss      

        return (total_loss, loss, reg_loss, aux_loss)

    def _add_global_pooling(self, X, in_w, in_h, in_ch):
        X = tf.reduce_mean(X, (1, 2))
        X = tf.reshape(X, (-1, in_ch)) # Sanity shape check
        return X

    def _count_model_parameters(self, tf_vars):
        num_params = 0
        # utils.logger.log('Model parameters:')
        for var in tf_vars:
            # utils.logger.log(str(var))
            num_params += np.prod([dim.value for dim in var.get_shape()])

        return num_params

    def _add_reduction_cell(self, cell_arch, inputs, block_ch, drop_path_keep_prob):
        ni = len(inputs) # no. of inputs
        b = len(cell_arch) # no. of blocks
        hidden_states = [] # Stores hidden states for this cell, which includes blocks

        # Initial width & height for this cell as the final input
        (_, w, h, _) = inputs[-1]

        # Calibrate inputs as necessary and add them to hidden states
        for (i, (inp, w_inp, h_inp, ch_inp)) in enumerate(inputs):
            with tf.variable_scope('input_{}_calibrate'.format(i)):
                inp = self._calibrate(inp, w_inp, h_inp, ch_inp, w, h, block_ch)
                hidden_states.append(inp)

        # Make each block, also recording whether each block is used 
        hidden_state_used_counts = [0 for _ in range(ni + b)]
        for bi in range(b):
            with tf.variable_scope('block_{}'.format(bi)):
                (idx1, op1, idx2, op2) = cell_arch[bi]
                X1 = hidden_states[idx1]
                hidden_state_used_counts[idx1] += 1
                X2 = hidden_states[idx2]
                hidden_state_used_counts[idx2] += 1

                with tf.variable_scope('X1'):
                    # Don't halve dimensions if X1 is a fellow block
                    if idx1 < ni:
                        X1 = self._add_op(X1, op1, w, h, block_ch, stride=2)
                    else:
                        X1 = self._add_op(X1, op1, w >> 1, h >> 1, block_ch)
                    X1 = self._do_drop_path(X1, drop_path_keep_prob)

                with tf.variable_scope('X2'):
                    if idx2 < ni:
                        X2 = self._add_op(X2, op2, w, h, block_ch, stride=2)
                    else:
                        X2 = self._add_op(X2, op2, w >> 1, h >> 1, block_ch)
                    X2 = self._do_drop_path(X2, drop_path_keep_prob)
                    
                X = tf.add_n([X1, X2])

            hidden_states.append(X)

        # Combine all unused hidden states
        comb_states = [X for (i, X) in enumerate(hidden_states) if i > ni and hidden_state_used_counts[i] == 0] 
        comb_ch = len(comb_states) * block_ch
        with tf.variable_scope('combine'):
            X = tf.concat(comb_states, axis=3)

        X = tf.reshape(X, (-1, w >> 1, h >> 1, comb_ch)) # Sanity shape check

        return (X, w >> 1, h >> 1, comb_ch)

    def _add_normal_cell(self, cell_arch, inputs, block_ch, drop_path_keep_prob):
        ni = len(inputs) # no. of inputs
        b = len(cell_arch) # no. of blocks
        hidden_states = [] # Stores hidden states for this cell, which includes blocks

        # Initial width & height for this cell as the final input
        (_, w, h, _) = inputs[-1]

        # Calibrate inputs as necessary and add them to hidden states
        for (i, (inp, w_inp, h_inp, ch_inp)) in enumerate(inputs):
            with tf.variable_scope('input_{}_calibrate'.format(i)):
                inp = self._calibrate(inp, w_inp, h_inp, ch_inp, w, h, block_ch)
                hidden_states.append(inp)

        # Make each block, also recording whether each block is used 
        hidden_state_used_counts = [0 for _ in range(ni + b)]
        for bi in range(b):
            with tf.variable_scope('block_{}'.format(bi)):
                (idx1, op1, idx2, op2) = cell_arch[bi]
                X1 = hidden_states[idx1]
                hidden_state_used_counts[idx1] += 1
                X2 = hidden_states[idx2]
                hidden_state_used_counts[idx2] += 1

                with tf.variable_scope('X1'):
                    X1 = self._add_op(X1, op1, w, h, block_ch, stride=1)
                    X1 = self._do_drop_path(X1, drop_path_keep_prob)

                with tf.variable_scope('X2'):
                    X2 = self._add_op(X2, op2, w, h, block_ch, stride=1)
                    X2 = self._do_drop_path(X2, drop_path_keep_prob)

                X = tf.add_n([X1, X2])

            hidden_states.append(X)

        # Combine all unused hidden states
        comb_states = [X for (i, X) in enumerate(hidden_states) if i > ni and hidden_state_used_counts[i] == 0] 
        comb_ch = len(comb_states) * block_ch
        with tf.variable_scope('combine'):
            X = tf.concat(comb_states, axis=3)

        X = tf.reshape(X, (-1, w, h, comb_ch)) # Sanity shape check

        return (X, w, h, comb_ch)

    def _add_op(self, X, op, w, h, ch, stride=1):
        '''
        Applies a specific operation to input
        '''
        ops = {
            0: lambda: self._add_separable_conv_op(X, w, h, ch, filter_size=3, stride=stride),
            1: lambda: self._add_separable_conv_op(X, w, h, ch, filter_size=5, stride=stride),
            2: lambda: self._add_avg_pool_op(X, w, h, ch, filter_size=3, stride=stride),
            3: lambda: self._add_max_pool_op(X, w, h, ch, filter_size=3, stride=stride),
            4: lambda: self._add_max_pool_op(X, w, h, ch, filter_size=1, stride=stride), # identity
            5: lambda: self._add_separable_conv_op(X, w, h, ch, filter_size=7, stride=stride),
            6: lambda: self._add_conv_op(X, w, h, ch, filter_size=3, stride=stride),
            7: lambda: self._add_conv_op(X, w, h, ch, filter_size=5, stride=stride)
        }
        X = ops[op]()
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
    
    def _add_separable_conv_op(self, X, w, h, ch, filter_size, stride, ch_mul=1, num_stacks=2):
        with tf.variable_scope('separable_conv_op'):
            # For each stack of separable convolution (default of 2)
            for stack_no in range(num_stacks):
                # Only have > 1 stride for first stack 
                stack_stride = stride if stack_no == 0 else 1 

                with tf.variable_scope('stack_{}'.format(stack_no)):
                    W_d = self._make_var('W_d', (filter_size, filter_size, ch, ch_mul))
                    W_p = self._make_var('W_p', (1, 1, ch_mul * ch, ch))
                    X = tf.nn.relu(X)
                    X = tf.nn.separable_conv2d(X, W_d, W_p, strides=(1, stack_stride, stack_stride, 1), padding='SAME')
                    X = self._add_batch_norm(X, ch)

        X = tf.reshape(X, (-1, w // stride, h // stride, ch)) # Sanity shape check
        return X

    def _add_conv_op(self, X, w, h, ch, filter_size, stride):
        with tf.variable_scope('conv_op'):
            W = self._make_var('W', (filter_size, filter_size, ch, ch))
            X = tf.nn.relu(X)
            X = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='SAME')
            X = self._add_batch_norm(X, ch)
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

    def _do_drop_path(self, X, keep_prob):
        # Apply dropout
        keep_prob = tf.cast(keep_prob, tf.float32)
        batch_size = tf.shape(X)[0]
        noise_shape = (batch_size, 1, 1, 1)
        random_tensor = keep_prob + tf.random_uniform(noise_shape, dtype=tf.float32)
        binary_tensor = tf.floor(random_tensor)
        X = tf.div(X, keep_prob) * binary_tensor
        return X

    def _do_conv(self, X, w, h, in_ch, ch, filter_size=1, do_relu=False, no_reg=False):
        with tf.variable_scope('conv'):
            W = self._make_var('W', (filter_size, filter_size, in_ch, ch), no_reg=no_reg)
            if do_relu:
                X = tf.nn.relu(X)
            X = tf.nn.conv2d(X, W, (1, 1, 1, 1), padding='SAME')
            X = self._add_batch_norm(X, ch)

        X = tf.reshape(X, (-1, w, h, ch)) # Sanity shape check
        return X

    def _calibrate(self, X, w, h, ch, w_out, h_out, ch_out):
        '''
        Calibrate input of shape (-1, w, h, ch) to (-1, w_out, h_out, ch_out), assuming (w, h) / (w_out, h_out) is power of 2
        '''
        # Downsample with factorized reduction
        downsample_no = 0
        while w > w_out or h > h_out:
            downsample_no += 1
            with tf.variable_scope('downsample_{}x'.format(downsample_no)):
                X = self._add_factorized_reduction(X, w, h, ch)
                ch <<= 1
                w >>= 1
                h >>= 1

        # Convert channel counts with 1x1 conv
        if ch != ch_out:
            with tf.variable_scope('convert_conv'):
                X = self._do_conv(X, w, h, ch, ch_out, filter_size=1, do_relu=True)

        X = tf.reshape(X, (-1, w_out, h_out, ch_out)) # Sanity shape check
        return X

    def _add_fully_connected(self, X, in_shape, out_ch, no_reg=False):
        with tf.variable_scope('fully_connected'):
            ch = np.prod(in_shape)
            X = tf.reshape(X, (-1, ch))
            W = self._make_var('W', (ch, out_ch), no_reg=no_reg)
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
            W_half_1 = self._make_var('W_half_1', (1, 1, in_ch, in_ch))
            X_half_1 = tf.nn.conv2d(half_1, W_half_1, (1, 1, 1, 1), padding='SAME')
            W_half_2 = self._make_var('W_half_2', (1, 1, in_ch, in_ch))
            X_half_2 = tf.nn.conv2d(half_2, W_half_2, (1, 1, 1, 1), padding='SAME')
            
            # Concat both halves across channels
            X = tf.concat([X_half_1, X_half_2], axis=3)

            # Apply batch normalization
            X = self._add_batch_norm(X, in_ch * 2)

        X = tf.reshape(X, (-1, in_w // 2, in_h // 2, in_ch * 2)) # Sanity shape check

        return X

    def _add_batch_norm(self, X, in_ch, decay=0.9, epsilon=1e-5):
        with tf.variable_scope('batch_norm'):
            offset = self._make_var('offset', (in_ch,),
                                    initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            scale = self._make_var('scale', (in_ch,), 
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

    def _make_var(self, name, shape, no_share=False, no_reg=False, initializer=None):
        if initializer is None:
            initializer = tf.contrib.keras.initializers.he_normal()

        # Ensure that name is unique by shape too
        name += '-shape-{}'.format('x'.join([str(x) for x in shape]))

        var = tf.get_variable(name, shape, initializer=initializer)

        # Mark var as shared
        if not no_share:
            tf.add_to_collection(self.TF_COLLECTION_SHARED, var)

        # Add L2 regularization node for var
        if not no_reg:
            l2_loss = tf.nn.l2_loss(var)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2_loss)
        
        return var
    
    def _get_all_variables(self):
        tf_vars = [var for var in tf.trainable_variables()]
        return tf_vars

if __name__ == '__main__':
    tune_model(
        TfEnasChild, 
        train_dataset_uri='data/cifar_10_for_image_classification_train.zip',
        val_dataset_uri='data/cifar_10_for_image_classification_val.zip'
    )