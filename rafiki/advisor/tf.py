import tensorflow as tf
import numpy as np
import bisect

from .advisor import BaseKnobAdvisor, UnsupportedKnobTypeError
from .knob import ListKnob, CategoricalKnob

class EnasKnobAdvisor(BaseKnobAdvisor):
    '''
    Implements the controller of "Efficient Neural Architecture Search via Parameter Sharing" (ENAS) for image classification.
    
    Paper: https://arxiv.org/abs/1802.03268
    '''
    def start(self, knob_config):
        self._knob_config = self._validate_knob_config(knob_config)
        self._list_knob_models = self._build_models()

    def propose(self):
        knobs = {}
        for (name, knob) in self._knob_config.items():
            knobs[name] = self._propose_for_knob(name, knob)

        return knobs

    def feedback(self, score, knobs):
        for (name, value) in knobs.items():
            knob = self._knob_config[name]
            self._feedback_for_knob(name, knob, value, score)
    
    def _validate_knob_config(self, knob_config):
        for knob in knob_config.values():
            if isinstance(knob, ListKnob):
                # Supports only `ListKnob` of `CategoricalKnob`
                for knob in knob.items:
                    if not isinstance(knob, CategoricalKnob):
                        raise UnsupportedKnobTypeError('Only `ListKnob` of `CategoricalKnob` is supported')
            else:
                raise UnsupportedKnobTypeError(knob.__class__)

        return knob_config

    def _feedback_for_knob(self, name, knob, knob_value, score):
        if isinstance(knob, ListKnob):
            list_knob_model = self._list_knob_models[name]
            list_knob_model.feedback(knob_value, score)

    def _propose_for_knob(self, name, knob):
        if isinstance(knob, ListKnob):
            list_knob_model = self._list_knob_models[name]
            return list_knob_model.propose()

    def _build_models(self):
        knob_config = self._knob_config
        list_knobs = [(name, knob) for (name, knob) in knob_config.items() if isinstance(knob, ListKnob)]

        # Build a model for each list knob
        list_knob_models = {}
        for (name, list_knob) in list_knobs:
            with tf.variable_scope(name):
                list_knob_models[name] = EnasKnobAdvisorListModel(list_knob)

        return list_knob_models

class EnasKnobAdvisorListModel():
    def __init__(self, knob):
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        self._knob = knob

        with self._graph.as_default():
            self._build_model()
            self._make_train_op()
            self._start_session()

            tf_vars = self._get_all_variables()
            model_params_count = self._count_model_parameters(tf_vars)
            print('Model has {} parameters'.format(model_params_count))

    def propose(self):
        item_knobs = self._knob.items
        item_idxs = self._item_idxs

        with self._graph.as_default():
            item_idxs_real = self._sess.run(item_idxs)
            items = [item_knob.values[idx] for (item_knob, idx) in zip(item_knobs, item_idxs_real)]

        return items

    def feedback(self, items, score):
        item_knobs = self._knob.items
        item_idxs = [item_knob.values.index(item) for (item_knob, item) in zip(item_knobs, items)]

        print('Training from feedback ({}, {})...'.format(items, score))
        with self._graph.as_default():
            (reward_base, loss, reward, item_logits, _) = self._sess.run(
                [self._reward_base, self._loss, self._reward, self._item_logits, self._train_op],
                feed_dict={
                    self._item_idxs_ph: item_idxs,
                    self._score_ph: score
                }
            )

            print('Reward: {}'.format(reward))
            print('Reward baseline: {}'.format(reward_base))
            print('Loss: {}'.format(loss))
            # print('Logits: {}'.format(list(item_logits)))

    def _start_session(self):
        self._sess.run(tf.global_variables_initializer())

    def _make_train_op(self):
        knob = self._knob
        item_logits = self._item_logits
        N = len(knob) # Length of list
        base_decay = 0.99
        learning_rate = 0.001 * 50
        adam_beta1 = 0
        adam_epsilon = 1e-3
        entropy_weight = 0.0001

        # Placeholders for item indexes and associated score
        item_idxs_ph = tf.placeholder(dtype=tf.int32, shape=(N,))
        score_ph = tf.placeholder(dtype=tf.float32)

        # Compute log probs & entropy
        sample_log_probs = self._compute_sample_log_probs(item_idxs_ph, item_logits)
        sample_entropy = self._compute_sample_entropy(item_logits)

        # Compute reward
        # Adding entropy encourages exploration
        reward = score_ph
        reward += entropy_weight * sample_entropy

        # Baseline reward for REINFORCE
        reward_base = tf.Variable(0., name='reward_base', dtype=tf.float32, trainable=False)

        # Update baseline whenever reward updates
        base_update = tf.assign_sub(reward_base, (1 - base_decay) * (reward_base - reward))
        with tf.control_dependencies([base_update]):
            reward = tf.identity(reward)

        # Compute loss
        loss = sample_log_probs * (reward - reward_base)

        # Add optimizer
        tf_vars = self._get_all_variables()
        steps = tf.Variable(0, name='steps', dtype=tf.int32, trainable=False)
        grads = tf.gradients(loss, tf_vars)
        opt = tf.train.AdamOptimizer(learning_rate, beta1=adam_beta1, epsilon=adam_epsilon,
                                    use_locking=True)
        train_op = opt.apply_gradients(zip(grads, tf_vars), global_step=steps)
        
        self._train_op = train_op
        self._loss = loss
        self._reward = reward
        self._reward_base = reward_base
        self._item_idxs_ph = item_idxs_ph
        self._score_ph = score_ph

    def _build_model(self):
        knob = self._knob
        N = len(knob) # Length of list
        H = 32 # Number of units in LSTM
        lstm_num_layers = 2

        # List of counts corresponding to the no. of values for each list item
        Ks = [len(item_knob.values) for item_knob in knob.items]

        # Build LSTM
        lstm  = self._build_lstm(lstm_num_layers, H)

        # Initial embedding passed to LSTM
        initial_embed = self._make_var('item_embed_initial', (1, H))

        # TODO: Add attention
        
        item_idxs = []
        item_logits = []
        lstm_states = [None]
        item_embeds = [initial_embed]
        for i in range(N):
            K = Ks[i] # No of categories for output

            with tf.variable_scope('item_{}'.format(i)):
                # Run input through LSTM to get output
                (X, lstm_state) = self._apply_lstm(item_embeds[-1], lstm, H, prev_state=lstm_states[-1])
                lstm_states.append(lstm_state)

                # Add fully connected layer and transform to `K` channels
                logits = self._add_fully_connected(X, (1, H), K)
                item_logits.append(logits)

                # Draw and save item index from probability distribution by `X`
                item_idx = self._sample_from_logits(logits)
                item_idxs.append(item_idx)

                # If not the final item
                if i < N - 1:
                    # Run item index through embedding lookup
                    embeds = self._make_var('W_embeds', (K, H)) 
                    item_embed = tf.reshape(tf.nn.embedding_lookup(embeds, item_idx), (1, -1))
                    item_embeds.append(item_embed)

        self._item_idxs = item_idxs
        self._item_logits = item_logits

    def _compute_sample_log_probs(self, item_idxs, item_logits):
        N = len(item_logits)
        sample_log_probs = tf.constant(0., dtype=tf.float32, name='sample_log_probs')

        for i in range(N):
            idx = item_idxs[i]
            logits = item_logits[i]
            log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(logits, (1, -1)), 
                                                                        labels=tf.reshape(idx, (1,)))
            sample_log_probs += log_probs[0]
        
        return sample_log_probs

    def _compute_sample_entropy(self, item_logits):
        N = len(item_logits)
        sample_entropy = tf.constant(0., dtype=tf.float32, name='sample_entropy')

        for i in range(N):
            logits = item_logits[i]
            entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.reshape(logits, (1, -1)),
                                                            labels=tf.reshape(tf.nn.softmax(logits), (1, -1)))
            entropy = tf.stop_gradient(entropy)
            sample_entropy += entropy[0]
        
        return sample_entropy
    
    ####################################
    # Utils
    ####################################

    def _sample_from_logits(self, logits):
        idx = tf.multinomial(tf.reshape(logits, (1, -1)), 1)[0][0]
        return idx

    def _count_model_parameters(self, tf_vars):
        num_params = 0
        # print('Model parameters:')
        for var in tf_vars:
            # print(var)
            num_params += np.prod([dim.value for dim in var.get_shape()])

        return num_params

    def _add_fully_connected(self, X, in_shape, out_ch):
        with tf.variable_scope('fully_connected'):
            ch = np.prod(in_shape)
            X = tf.reshape(X, (-1, ch))
            W = self._make_var('W', (ch, out_ch))
            b = self._make_var('b', (1, out_ch))
            X = tf.matmul(X, W) + b
        X = tf.reshape(X, (-1, out_ch)) # Sanity shape check
        return X

    def _apply_lstm(self, X, lstm, H, prev_state=None):
        '''
        Assumes 1 time step
        '''
        N = tf.shape(X)[0]
        prev_state = prev_state if prev_state is not None else lstm.zero_state(N, dtype=tf.float32)
        X = tf.reshape(X, (N, 1, H))

        # Pass input through LSTM layers
        (X, state) = tf.nn.dynamic_rnn(lstm, X, initial_state=prev_state)

        X = tf.reshape(X, (N, H))

        return (X, state)

    def _build_lstm(self, lstm_num_layers, H):
        lstm_cells = []
        with tf.variable_scope('lstm'):
            for i in range(lstm_num_layers):
                with tf.variable_scope('layer_{}'.format(i)):
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(H)
                    lstm_cells.append(lstm_cell)

        lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
        return lstm

    def _make_var(self, name, shape, initializer=None):
        if initializer is None:
            initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        return tf.get_variable(name, shape, initializer=initializer)

    def _get_all_variables(self):
        tf_vars = [var for var in tf.trainable_variables()]
        return tf_vars