import tensorflow as tf
from tensorflow.python.client import device_lib
import os
import numpy as np
import math
import random
from datetime import datetime
from collections import namedtuple

from rafiki.model import BaseModel, utils, IntegerKnob, CategoricalKnob, FloatKnob, FixedKnob
from rafiki.advisor import tune_model

_Model = namedtuple('_Model', ['images_ph', 'classes_ph', 'is_train_ph', 'step', 'probs', 'acc', 'loss',
                                'train_op', 'init_op', 'summary_op'])

class TfAllCnnModelC(BaseModel):
    '''
    Model C in https://arxiv.org/pdf/1412.6806.pdf
    '''
    TF_COLLECTION_MONITORED = 'MONITORED'
    
    @staticmethod
    def get_knob_config():
        return {
            'max_image_size': FixedKnob(32),
            'batch_size': FixedKnob(128),
            'max_trial_epochs': FixedKnob(200),
            'max_train_val_samples': FixedKnob(1024),
            'opt_momentum': FloatKnob(0.7, 1, is_exp=True),
            'lr': FloatKnob(1e-4, 1, is_exp=True),
            'lr_decay': FixedKnob(0.1),
            'weight_decay': FloatKnob(1e-5, 1e-2, is_exp=True),
            'input_dropout_rate': FloatKnob(0, 0.3),
            'pool_dropout_rate_1': FloatKnob(0, 0.7),
            'pool_dropout_rate_2': FloatKnob(0, 0.7),
            'early_stop_patience_epochs': FixedKnob(5),
        }

    def __init__(self, **knobs):
        self._knobs = knobs
        
    def train(self, dataset_uri, *args):
        (train_images, train_classes, train_val_images, 
            train_val_classes, self._train_params) = self._load_train_dataset(dataset_uri)
        (self._graph, self._sess, self._model, monitored_values) = self._build_model()
        with self._graph.as_default():
            self._train_summaries = self._train_model(train_images, train_classes, 
                                                    train_val_images, train_val_classes, 
                                                    monitored_values)
        
    def evaluate(self, dataset_uri):
        (images, classes) = self._load_val_dataset(dataset_uri, train_params=self._train_params)
        with self._graph.as_default():
            utils.logger.log('Evaluating model on validation dataset...')
            acc = self._evaluate_model(images, classes)
            utils.logger.log('Validation accuracy: {}'.format(acc))
        return acc

    def predict(self, queries):
        # TODO
        pass

    def save_parameters(self, params_dir):
        # TODO
        pass

    def load_parameters(self, params_dir):
        # TODO
        pass

    def _train_model(self, train_images, train_classes, 
                    train_val_images, train_val_classes, monitored_values):
        trial_epochs = self._get_trial_epochs()
        early_stop_patience = self._knobs['early_stop_patience_epochs']
        m = self._model

        # Define plots for monitored values
        for (name, _) in monitored_values.items():
            utils.logger.define_plot('"{}" Over Time'.format(name), [name])

        train_summaries = [] # List of (<steps>, <summary>) collected during training
        early_stop_condition = EarlyStopCondition(patience=early_stop_patience)
        
        for epoch in range(trial_epochs):
            utils.logger.log('Running epoch {}...'.format(epoch))

            # Run through train dataset
            train_loss = RunningAverage()
            train_acc = RunningAverage()
            stepper = self._feed_dataset_to_model(train_images, [m.train_op, m.summary_op, 
                                                    m.step, m.acc, m.loss, *monitored_values.values()], 
                                                    is_train=True, classes=train_classes)
            for (_, summary, batch_step, batch_acc, batch_loss, *values) in stepper:
                train_summaries.append((batch_step, summary))
                train_loss.add(batch_loss)
                train_acc.add(batch_acc)

            utils.logger.log(epoch=epoch, train_loss=train_loss.get(), train_acc=train_acc.get(),
                            **{ name: v for (name, v) in zip(monitored_values.keys(), values) })

            # Run through train-val dataset, if exists
            if len(train_val_images) > 0:
                train_val_loss = RunningAverage()
                train_val_acc = RunningAverage()
                stepper = self._feed_dataset_to_model(train_val_images, [m.loss, m.acc], classes=train_val_classes)
                for (batch_loss, batch_acc) in stepper:
                    train_val_loss.add(batch_loss)
                    train_val_acc.add(batch_acc)
                
                utils.logger.log(epoch=epoch, train_val_acc=train_val_acc.get(), 
                                train_val_loss=train_val_loss.get())

                # Early stop on train-val batch loss
                if early_stop_condition.check(train_val_loss.get()):
                    utils.logger.log('Average train-val batch loss has not improved for {} epochs'.format(early_stop_condition.patience))
                    utils.logger.log('Early stopping...')
                    break

        return train_summaries

    def _evaluate_model(self, images, classes):
        probs = self._predict_with_model(images)
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == np.asarray(classes))
        return acc

    def _predict_with_model(self, images):
        m = self._model
        all_probs = []
        stepper = self._feed_dataset_to_model(images, [m.probs])
        for (batch_probs,) in stepper:
            all_probs.extend(batch_probs)
        return np.asarray(all_probs)

    def _build_model(self):
        w = self._train_params['image_size']
        h = self._train_params['image_size']
        in_ch = 3

        graph = tf.Graph()

        with graph.as_default():
            # Define input placeholders to graph
            images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, w, h, in_ch)) # Images
            classes_ph = tf.placeholder(tf.int32, name='classes_ph', shape=(None,)) # Classes
            is_train_ph = tf.placeholder(tf.bool, name='is_train_ph', shape=()) # Are we training or predicting?
            step = self._make_var('step', (), dtype=tf.int32, trainable=False, initializer=tf.initializers.constant(0))

            # Preprocess
            (images, classes, init_op) = self._preprocess(images_ph, classes_ph)
            
            # Forward
            logits = self._forward(images, is_train_ph)

            # Compute probabilities, predictions, accuracy
            probs = tf.nn.softmax(logits)
            preds = tf.argmax(logits, axis=1, output_type=tf.int32)
            acc = tf.reduce_mean(tf.cast(tf.equal(preds, classes), tf.float32))

            # Compute loss
            loss = self._compute_loss(logits, classes)

            # Optimize
            train_op = self._optimize(loss, step)

            # Count model parameters 
            self._count_model_parameters()

            # Monitor values
            (summary_op, monitored_values) = self._add_monitoring_of_values()

            # Session
            sess = self._make_session()

            model = _Model(images_ph, classes_ph, is_train_ph, step, probs, acc, loss, train_op, init_op, summary_op)

        return (graph, sess, model, monitored_values)

    def _feed_dataset_to_model(self, images, run_ops, is_train=False, classes=None):
        m = self._model

        # Shuffle dataset if training
        if is_train:
            zipped = list(zip(images, classes))
            random.shuffle(zipped)
            (images, classes) = zip(*zipped)
        
        # Initialize dataset (mock classes if required)
        self._sess.run(m.init_op, feed_dict={
            m.images_ph: images, 
            m.classes_ph: classes if classes is not None else np.zeros((len(images),))
        })

        while True:
            try:
                results = self._sess.run(run_ops, feed_dict={
                    m.is_train_ph: is_train
                })
                yield results
            except tf.errors.OutOfRangeError:
                break

    def _preprocess(self, images, classes):
        batch_size = self._knobs['batch_size']

        dataset = tf.data.Dataset.from_tensor_slices((images, classes)) \
                    .batch(batch_size)
        dataset_itr = dataset.make_initializable_iterator()
        (images, classes) = dataset_itr.get_next()
        init_op = dataset_itr.initializer
        return (images, classes, init_op)

    def _forward(self, X, is_train):
        w = self._train_params['image_size']
        h = self._train_params['image_size']
        K = self._train_params['K']
        input_dropout_rate = self._knobs['input_dropout_rate']
        pool_dropout_rate_1 = self._knobs['pool_dropout_rate_1']
        pool_dropout_rate_2 = self._knobs['pool_dropout_rate_2']
        in_ch = 3
        chs = [96, 192]

        # Dropout input
        X = tf.cond(is_train, lambda: tf.nn.dropout(X, 1 - input_dropout_rate), lambda: X)

        # Layers
        with tf.variable_scope('layer_1'):
            X = self._add_conv(X, w, h, in_ch=in_ch, out_ch=chs[0], filter_size=3, padding='SAME', is_input=True)
        with tf.variable_scope('layer_2'):
            X = self._add_conv(X, w, h, in_ch=chs[0], out_ch=chs[0], filter_size=3, padding='SAME')
        with tf.variable_scope('layer_3_pool'):
            X = self._do_pool(X, w, h, in_ch=chs[0], filter_size=3)
            X = tf.cond(is_train, lambda: tf.nn.dropout(X, 1 - pool_dropout_rate_1), lambda: X)
        with tf.variable_scope('layer_4'):
            X = self._add_conv(X, w >> 1, h >> 1, in_ch=chs[0], out_ch=chs[1], filter_size=3, padding='SAME')
        with tf.variable_scope('layer_5'):
            X = self._add_conv(X, w >> 1, h >> 1, in_ch=chs[1], out_ch=chs[1], filter_size=3, padding='SAME')
        with tf.variable_scope('layer_6'):
            X = self._add_conv(X, w >> 1, h >> 1, in_ch=chs[1], out_ch=chs[1], filter_size=3, padding='SAME')
        with tf.variable_scope('layer_7_pool'):
            X = self._do_pool(X, w >> 1, h >> 1, in_ch=chs[1], filter_size=3)
            X = tf.cond(is_train, lambda: tf.nn.dropout(X, 1 - pool_dropout_rate_2), lambda: X)
        with tf.variable_scope('layer_8'):
            X = self._add_conv(X, w >> 2, h >> 2, in_ch=chs[1], out_ch=chs[1], filter_size=3, padding='VALID')
        with tf.variable_scope('layer_9'):
            X = self._add_conv(X, (w >> 2) - 2, (h >> 2) - 2, in_ch=chs[1], out_ch=chs[1], filter_size=1)
        with tf.variable_scope('layer_10'):
            X = self._add_conv(X, (w >> 2) - 2, (h >> 2) - 2, in_ch=chs[1], out_ch=K, filter_size=1)
            
        logits = self._do_global_avg_pool(X, (w >> 2) - 2, (h >> 2) - 2, in_ch=K)
        return logits
    
    def _add_conv(self, X, in_w, in_h, in_ch, out_ch, filter_size=1, no_reg=False, padding='SAME', is_input=False):
        W = self._make_var('W', (filter_size, filter_size, in_ch, out_ch), no_reg=no_reg, initializer=tf.initializers.he_normal())
        X = tf.nn.conv2d(X, W, (1, 1, 1, 1), padding=padding)
        X = tf.nn.relu(X)
        out_w = in_w - filter_size + 1 if padding == 'VALID' else in_w
        out_h = in_h - filter_size + 1 if padding == 'VALID' else in_h
        X = tf.reshape(X, (-1, out_w, out_h, out_ch)) # Sanity shape check
        return X

    def _get_learning_rate(self, step):
        N = self._train_params['N']
        batch_size = self._knobs['batch_size']
        lr = self._knobs['lr']
        lr_decay = self._knobs['lr_decay']
        trial_epochs = self._get_trial_epochs()
        lr_decay_epochs = [int(0.5 * trial_epochs), int(0.75 * trial_epochs)]

        steps_per_epoch = math.ceil(N / batch_size)
        epoch = step // steps_per_epoch

        lr = tf.constant(lr)
        for decay_epoch in lr_decay_epochs:
            lr = tf.cond(tf.less(decay_epoch, epoch), lambda: lr * lr_decay, lambda: lr)
        
        self._mark_for_monitoring('lr', lr)
        
        return lr

    def _optimize(self, loss, step):
        opt_momentum = self._knobs['opt_momentum'] # Momentum optimizer momentum
        lr = self._get_learning_rate(step)

        tf_trainable_vars = tf.trainable_variables()
        grads = tf.gradients(loss, tf_trainable_vars)
        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=opt_momentum)
        train_op = opt.apply_gradients(zip(grads, tf_trainable_vars), global_step=step)

        return train_op

    def _compute_loss(self, logits, classes):
        weight_decay = self._knobs['weight_decay']

        log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=classes)
        loss = tf.reduce_mean(log_probs)

        # Add regularization loss
        # Equivalent to weight decay according to https://arxiv.org/pdf/1711.05101.pdf
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = weight_decay * tf.add_n(reg_losses)

        total_loss = loss + reg_loss

        return total_loss

    def _load_train_dataset(self, dataset_uri):
        max_train_val_samples = self._knobs['max_train_val_samples']
        max_image_size = self._knobs['max_image_size']

        dataset = utils.dataset.load_dataset_of_image_files(dataset_uri, max_image_size=max_image_size, 
                                                        mode='RGB')
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        train_val_samples = min(dataset.size // 5, max_train_val_samples) # up to 1/5 of samples for train-val
        (train_images, train_classes) = (images[train_val_samples:], classes[train_val_samples:])
        (train_val_images, train_val_classes) = (images[:train_val_samples], classes[:train_val_samples])

        # Normalize train & train-val data
        (train_images, norm_mean, norm_std) = utils.dataset.normalize_images(train_images)
        (train_val_images, _, _) = utils.dataset.normalize_images(train_val_images, norm_mean, norm_std)

        train_params = {
            'norm_mean': norm_mean,
            'norm_std': norm_std,
            'image_size': dataset.image_size,
            'N': dataset.size,
            'K': dataset.classes
        }

        utils.logger.log('Train dataset has {} samples'.format(len(train_images)))
        utils.logger.log('Train-val dataset has {} samples'.format(len(train_val_images)))

        return (train_images, train_classes, train_val_images, train_val_classes, train_params)

    def _load_val_dataset(self, dataset_uri, train_params):
        image_size = train_params['image_size']
        norm_mean = train_params['norm_mean']
        norm_std = train_params['norm_std']

        dataset = utils.dataset.load_dataset_of_image_files(dataset_uri, max_image_size=image_size, 
                                                        mode='RGB')
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])

        # Normalize val data
        (images, _, _) = utils.dataset.normalize_images(images, norm_mean, norm_std)

        return (images, classes)

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

    def _make_session(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        return sess

    def _get_trial_epochs(self):
        max_trial_epochs = self._knobs['max_trial_epochs']
        return max_trial_epochs

    ####################################
    # Utils
    ####################################


    def _do_global_avg_pool(self, X, in_w, in_h, in_ch):
        X = tf.reduce_mean(X, (1, 2))
        X = tf.reshape(X, (-1, in_ch)) # Sanity shape check
        return X

    def _do_pool(self, X, in_w, in_h, in_ch, filter_size=3, padding='SAME'):
        stride = 2
        X = tf.nn.max_pool(X, ksize=(1, filter_size, filter_size, 1), 
                        strides=(1, stride, stride, 1), padding=padding)
        out_w = (in_w - filter_size + 1) >> 1 if padding == 'VALID' else in_w >> 1
        out_h = (in_h - filter_size + 1) >> 1 if padding == 'VALID' else in_h >> 1
        X = tf.reshape(X, (-1, out_w, out_h, in_ch)) # Sanity shape check
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

    def _make_var(self, name, shape, dtype=None, no_reg=False, initializer=None, trainable=True):
        # Ensure that name is unique by shape too
        name += '-shape-{}'.format('x'.join([str(x) for x in shape]))

        var = tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable)

        # Add L2 regularization node for var
        if trainable and not no_reg:
            l2_loss = tf.nn.l2_loss(var)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2_loss)
        
        return var

class RunningAverage():
    def __init__(self):
        self._avg = 0
        self._count = 0
            
    def add(self, val):
        self._avg = self._avg * self._count / (self._count + 1) + val / (self._count + 1)
        self._count += 1
        
    def get(self) -> float:
        return self._avg

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

class EarlyStopCondition():
    '''
    :param int patience: How many steps should the condition tolerate before calling early stop (-1 for no stop)
    '''
    def __init__(self, patience=5, if_max=False):
        self._patience = patience
        self._if_max = if_max
        self._last_best = float('inf') if not if_max else float('-inf')
        self._wait_count = 0

    @property
    def patience(self):
        return self._patience
    
    # Returns whether should early stop
    def check(self, value) -> bool:
        if self._patience < 0: # No stop
            return False

        if (not self._if_max and value < self._last_best) or \
            (self._if_max and value > self._last_best):
            self._wait_count = 0
            self._last_best = value
        else:
            self._wait_count += 1

        if self._wait_count >= self._patience:
            return True
        else:
            return False

if __name__ == '__main__':
    tune_model(
        TfAllCnnModelC, 
        # train_dataset_uri='data/fashion_mnist_for_image_classification_train.zip',
        # val_dataset_uri='data/fashion_mnist_for_image_classification_val.zip',
        # test_dataset_uri='data/fashion_mnist_for_image_classification_test.zip',
        train_dataset_uri='data/cifar_10_for_image_classification_train.zip',
        val_dataset_uri='data/cifar_10_for_image_classification_val.zip',
        test_dataset_uri='data/cifar_10_for_image_classification_test.zip',
        total_trials=100,
        should_save=False
    )
