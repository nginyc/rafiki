import tensorflow as tf
import numpy as np
import bisect
from enum import Enum
import math
import logging
from collections import defaultdict

from rafiki.model import ListKnob, CategoricalKnob, FixedKnob, PolicyKnob
from rafiki.param_store import ParamsType

from .advisor import BaseAdvisor, UnsupportedKnobError, Proposal

logger = logging.getLogger(__name__)

class EnasTrainStrategy(Enum):
    ORIGINAL = 'ORIGINAL' # Cycle between 1 train - X evals, always use GLOBAL_RECENT 
    ISOLATED = 'ISOLATED' # Perform original ENAS locally at each worker, always use LOCAL_RECENT

class EnasAdvisor(BaseAdvisor):
    '''
    Implements the controller of "Efficient Neural Architecture Search via Parameter Sharing" (ENAS) for image classification.
    
    Paper: https://arxiv.org/abs/1802.03268
    '''
    @staticmethod
    def is_compatible(knob_config):
        # Supports only ListKnob of CategoricalKnobs, FixedKnob, and PolicyKnob
        for knob in knob_config.values():
            if isinstance(knob, ListKnob):
                for knob in knob.items:
                    if not isinstance(knob, CategoricalKnob):
                        return False
            elif not isinstance(knob, (FixedKnob, PolicyKnob)):
                return False

        # Prompt user that ENAS prefers having certain policies
        policies = [x.policy for (name, x) in knob_config.items() if isinstance(x, PolicyKnob)]
        if 'QUICK_TRAIN' not in policies or 'QUICK_EVAL' not in policies:
            print('To speed up ENAS, having `QUICK_TRAIN` and `QUICK_EVAL` policies is preferred.')

        return True
        
    def __init__(self, knob_config, train_strategy=EnasTrainStrategy.ISOLATED, 
                batch_size=10, num_eval_per_cycle=300, num_final_evals=10, 
                do_final_train=True):
        self._batch_size = batch_size
        self._num_eval_per_cycle = num_eval_per_cycle
        self._num_final_evals = num_final_evals
        self._evals = [] # [(score, proposal)]
        self._do_final_train = do_final_train
        self._worker_to_trial_nos = defaultdict(list) # { <worker_id>: [<trial no>]}
        self._train_strategy = EnasTrainStrategy(train_strategy)
        
        if self._train_strategy == EnasTrainStrategy.ORIGINAL:
            self._get_trial_type = self._get_trial_type_original
        elif self._train_strategy == EnasTrainStrategy.ISOLATED:
            self._get_trial_type = self._get_trial_type_isolated
        else:
            raise NotImplementedError()

        (self._fixed_knob_config, knob_config) = self.extract_knob_type(knob_config, FixedKnob)
        (self._policy_knob_config, knob_config) = self.extract_knob_type(knob_config, PolicyKnob)
        self._list_knob_models = self._build_models(knob_config, batch_size)

    def propose(self, worker_id, trial_no, total_trials, concurrent_trial_nos=[]):
        (params_type, trial_type) = self._get_trial_type(worker_id, trial_no, total_trials, concurrent_trial_nos)
        
        if trial_type is None:
            return Proposal({}, is_valid=False)

        # Keep track of trial nos at each worker
        self._worker_to_trial_nos[worker_id].append(trial_no)

        if trial_type == 'TRAIN':
            knobs = self._propose_knobs(['QUICK_TRAIN'])
            return Proposal(knobs, 
                            params_type=params_type, 
                            should_eval=False,
                            should_save_to_disk=False)
        elif trial_type == 'EVAL':
            knobs = self._propose_knobs(['QUICK_EVAL'])
            return Proposal(knobs,
                            params_type=params_type, 
                            should_train=False, 
                            should_save_to_disk=False)
        elif trial_type == 'FINAL_EVAL':
            knobs = self._propose_knobs()
            return Proposal(knobs,
                            params_type=params_type, 
                            should_train=False, 
                            should_save_to_disk=True)
        elif trial_type == 'FINAL_TRAIN':
            # Do standard model training from scratch with final knobs
            knobs = self._propose_final_knobs()
            return Proposal(knobs, 
                            params_type=params_type,
                            should_save_to_disk=True)

    def feedback(self, score, proposal: Proposal):
        knobs = proposal.knobs
        num_final_evals = self._num_final_evals

        # Keep track of last X evals' scores
        self._evals = [*self._evals[:(num_final_evals-1)], (score, proposal)]

        for (name, list_knob_model) in self._list_knob_models.items():
            knob_value = knobs[name]
            list_knob_model.feedback(knob_value, score)

    def _propose_final_knobs(self):
        # If hasn't collected any evals, propose from model
        if len(self._evals) == 0:
            return self._propose_knobs()

        # Otherwise, determine best eval and use it
        self._evals.sort()
        (_, proposal) = self._evals.pop()

        return proposal.knobs

    def _propose_knobs(self, policies=[]):
        knobs = {}
        for (name, list_knob_model) in self._list_knob_models.items():
            knobs[name] = list_knob_model.propose()

        # Add fixed & policy knobs
        knobs = self.merge_fixed_knobs(knobs, self._fixed_knob_config)
        knobs = self.merge_policy_knobs(knobs, self._policy_knob_config, policies)
        
        return knobs

    def _build_models(self, knob_config, batch_size):
        list_knobs = [(name, knob) for (name, knob) in knob_config.items()]

        # Build a model for each list knob
        list_knob_models = {}
        for (name, list_knob) in list_knobs:
            with tf.variable_scope(name):
                list_knob_models[name] = EnasListKnobAdvisor(list_knob, batch_size)

        return list_knob_models

    def _get_trial_type_original(self, worker_id, trial_no, total_trials, concurrent_trial_nos):
        num_final_trains = 1 if self._do_final_train else 0
        num_final_evals = self._num_final_evals
        num_eval_per_cycle = self._num_eval_per_cycle
        E = self._batch_size
        T = num_eval_per_cycle + 1 # Period

        # Schedule: |--<train & eval cycles>---||--<final evals> ---||--<final train>--|
        # Final trains & evals can be parallelized
        
        is_final_train = trial_no > total_trials - num_final_trains
        final_train_preceding_trial_no = total_trials - num_final_trains + 1 

        is_final_eval = trial_no > total_trials - num_final_trains - num_final_evals
        final_eval_preceding_trial_no = total_trials - num_final_trains - num_final_evals + 1 

        # If not final eval or train, it is in the <train + eval> phase
        # Every (X + 1) trials, only train 1 epoch for the first trial
        # The other X trials is for training the controller
        # For a "train" trial, wait until previous trials to finish
        # Within a batch, "eval" trials can be done in parallel
        # But must wait for the previous "eval" batch to finish
        is_train = ((trial_no - 1) % T == 0)
        train_trial_no = ((trial_no - 1) // T) * T + 1 # Corresponding train trial
        minibatch_head_trial_no = train_trial_no + (trial_no - train_trial_no - 1) // E * E + 1  # Corresponding head trial for minibatch
        train_preceding_trial_no = train_trial_no
        eval_preceding_trial_no = minibatch_head_trial_no

        if is_final_train:
            if self._if_preceding_trials_are_running(final_train_preceding_trial_no, concurrent_trial_nos):
                return (None, None)
            return (ParamsType.NONE, 'FINAL_TRAIN')
        
        if is_final_eval:
            if self._if_preceding_trials_are_running(final_eval_preceding_trial_no, concurrent_trial_nos):
                return (None, None)
            return (ParamsType.GLOBAL_RECENT, 'FINAL_EVAL')

        if is_train:
            if self._if_preceding_trials_are_running(train_preceding_trial_no, concurrent_trial_nos):
                return (None, None)
            
            return (ParamsType.GLOBAL_RECENT, 'TRAIN')
        else:            
            if self._if_preceding_trials_are_running(eval_preceding_trial_no, concurrent_trial_nos):
                return (None, None)

            return (ParamsType.GLOBAL_RECENT, 'EVAL')

    def _get_trial_type_isolated(self, worker_id, trial_no, total_trials, concurrent_trial_nos):
        num_final_trains = 1 if self._do_final_train else 0
        num_final_evals = self._num_final_evals
        num_eval_per_cycle = self._num_eval_per_cycle
        T = num_eval_per_cycle + 1 # Period
        trial_nos = self._worker_to_trial_nos[worker_id]
        local_trial_no = len(trial_nos) + 1

        # Schedule: |--<train & eval cycles>---||--<final evals> ---||--<final train>--|
        # Final trains & evals can be parallelized
        
        is_final_train = trial_no > total_trials - num_final_trains
        final_train_preceding_trial_no = total_trials - num_final_trains + 1 

        is_final_eval = trial_no > total_trials - num_final_trains - num_final_evals
        final_eval_preceding_trial_no = total_trials - num_final_trains - num_final_evals + 1 

        # If not final eval or train, it is in the <train + eval> phase
        # Every (X + 1) trials, only train 1 epoch for the first trial
        # The other X trials is for training the controller
        # Use *local* trial no to determine this
        # Since trials on each worker are executed relatively sequentially, no need to check validity
        is_train_trial = ((local_trial_no - 1) % T == 0)

        if is_final_train:
            if self._if_preceding_trials_are_running(final_train_preceding_trial_no, concurrent_trial_nos):
                return (None, None)
            return (ParamsType.NONE, 'FINAL_TRAIN')
        
        if is_final_eval:
            if self._if_preceding_trials_are_running(final_eval_preceding_trial_no, concurrent_trial_nos):
                return (None, None)
            return (ParamsType.LOCAL_RECENT, 'FINAL_EVAL')
        
        if is_train_trial:
            return (ParamsType.LOCAL_RECENT, 'TRAIN')
        else:
            return (ParamsType.LOCAL_RECENT, 'EVAL')

    def _if_preceding_trials_are_running(self, trial_no, concurrent_trial_nos):
        if len(concurrent_trial_nos) == 0:
            return False

        min_trial_no = min(concurrent_trial_nos)
        return min_trial_no < trial_no

class EnasListKnobAdvisor():
    def __init__(self, knob, batch_size):
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        self._knob = knob
        self._batch_size = batch_size
        self._batch_items = [] # A running batch of items for feedback
        self._batch_scores = [] # A running batch of corresponding scores for feedback

        with self._graph.as_default():
            (self._item_logits, self._out_item_idxs, 
                self._train_op, self._losses, self._rewards,
                self._item_idxs_ph, self._scores_ph) = self._build_model(self._knob)
            self._start_session()

    def propose(self):
        items = self._predict_with_model()
        return items

    def feedback(self, items, score):
        self._batch_items.append(items)
        self._batch_scores.append(score)
        
        if len(self._batch_items) < self._batch_size:
            return

        self._train_model(self._batch_items, self._batch_scores)
        self._batch_items = []
        self._batch_scores = []

    def _predict_with_model(self):
        item_knobs = self._knob.items
        out_item_idxs = self._out_item_idxs

        with self._graph.as_default():
            item_idxs_real = self._sess.run(out_item_idxs)
            items = [item_knob.values[idx].value for (item_knob, idx) in zip(item_knobs, item_idxs_real)]
            return items

    def _train_model(self, batch_items, batch_scores):
        item_knobs = self._knob.items

        # Convert item values to indexes
        batch_item_idxs = [[[x.value for x in item_knob.values].index(value) for (item_knob, value) in zip(item_knobs, items)] for items in batch_items]

        logger.info('Training controller...')

        with self._graph.as_default():
            (losses, rewards, _) = self._sess.run(
                [self._losses, self._rewards, self._train_op],
                feed_dict={
                    self._item_idxs_ph: batch_item_idxs,
                    self._scores_ph: batch_scores
                }
            )

            # print('Rewards: {}'.format(rewards))
            # print('Losses: {}'.format(losses))

    def _start_session(self):
        self._sess.run(tf.global_variables_initializer())

    def _build_model(self, knob):
        assert isinstance(knob, ListKnob)

        batch_size = self._batch_size
        N = len(knob) # Length of list

        # Convert each item value to its value representation (for embeddings)
        (value_reps_by_item, K) = self._convert_values_to_reps(knob)

        # Placeholders for item indexes and associated score
        item_idxs_ph = tf.placeholder(dtype=tf.int32, shape=(batch_size, N))
        scores_ph = tf.placeholder(dtype=tf.float32, shape=(batch_size,))

        (item_logits, out_item_idxs) = self._forward(value_reps_by_item, K)
        (train_op, losses, rewards) = self._make_train_op(item_logits, item_idxs_ph, scores_ph)

        model_params_count = self._count_model_parameters()

        return (item_logits, out_item_idxs, train_op, losses, rewards, item_idxs_ph, scores_ph)

    # Convert each item value to its value representation
    def _convert_values_to_reps(self, knob):
        knob_value_to_rep = {}
        max_rep = 0
        for item_knob in knob.items:
            for knob_value in item_knob.values:
                if knob_value in knob_value_to_rep:
                    continue

                knob_value_to_rep[knob_value] = max_rep
                max_rep += 1

        value_reps_by_item = [[knob_value_to_rep[x] for x in item_knob.values] for item_knob in knob.items]
        return (value_reps_by_item, len(knob_value_to_rep))
      
    def _make_train_op(self, item_logits, item_idxs, scores):
        batch_size = self._batch_size
        base_decay = 0.99
        learning_rate = 0.0035
        adam_beta1 = 0
        adam_epsilon = 1e-3
        entropy_weight = 0.0001

        # Compute log probs & entropy
        sample_log_probs = self._compute_sample_log_probs(item_idxs, item_logits)
        sample_entropy = self._compute_sample_entropy(item_logits)

        # Compute rewards in a batch
        # Adding entropy encourages exploration
        rewards = scores
        rewards += entropy_weight * sample_entropy

        # Baseline reward for REINFORCE
        reward_base = tf.Variable(0., name='reward_base', dtype=tf.float32, trainable=False)

        # Update baseline whenever reward updates
        base_update = tf.assign_sub(reward_base, (1 - base_decay) * (reward_base - tf.reduce_mean(rewards)))
        with tf.control_dependencies([base_update]):
            rewards = tf.identity(rewards)

        # Compute losses in a batch
        losses = sample_log_probs * (rewards - reward_base)

        # Add optimizer
        tf_vars = self._get_all_variables()
        steps = tf.Variable(0, name='steps', dtype=tf.int32, trainable=False)
        grads = tf.gradients(losses, tf_vars)
        grads = [x / tf.constant(batch_size, dtype=tf.float32) for x in grads] # Average all gradients
        opt = tf.train.AdamOptimizer(learning_rate, beta1=adam_beta1, epsilon=adam_epsilon,
                                    use_locking=True)
        train_op = opt.apply_gradients(zip(grads, tf_vars), global_step=steps)
        
        return (train_op, losses, rewards)

    def _forward(self, reps_by_item, K):
        # ``K`` = <no. of unique item value reps>
        N = len(reps_by_item) # Length of list
        H = 32 # Number of units in LSTM
        lstm_num_layers = 2
        temperature = 0
        tanh_constant = 1.1

        # Build LSTM
        lstm = self._build_lstm(lstm_num_layers, H)

        # Initial embedding passed to LSTM
        initial_embed = self._make_var('item_embed_initial', (1, H))

        # Embedding for item values
        embeds = self._make_var('item_value_embeds', (K, H)) 

        # TODO: Add attention
        
        out_item_idxs = []
        item_logits = []
        lstm_states = [None]
        item_embeds = [initial_embed]

        for i in range(N):
            L = len(reps_by_item[i]) # No. of candidate values for item
            reps = tf.stack(reps_by_item[i])

            with tf.variable_scope('item_{}'.format(i)):
                # Run input through LSTM to get output
                (X, lstm_state) = self._apply_lstm(item_embeds[-1], lstm, H, prev_state=lstm_states[-1])
                lstm_states.append(lstm_state)

                # Add fully connected layer and transform to ``L`` channels
                logits = self._add_fully_connected(X, (1, H), L)
                logits = self._add_temperature(logits, temperature)
                logits = self._add_tanh_constant(logits, tanh_constant)
                item_logits.append(logits)
                
                # Draw and save item index from probability distribution from logits
                item_idx = self._sample_from_logits(logits)
                out_item_idxs.append(item_idx)

                # If not the final item
                if i < N - 1:
                    # Run item value rep through embedding lookup
                    rep = reps[item_idx]
                    item_embed = tf.reshape(tf.nn.embedding_lookup(embeds, rep), (1, -1))
                    item_embeds.append(item_embed)

        return (item_logits, out_item_idxs)

    def _compute_sample_log_probs(self, item_idxs, item_logits):
        N = len(item_logits)
        batch_size = tf.shape(item_idxs)[0] # item_idxs is of shape (bs, N)
        
        item_idxs = tf.transpose(item_idxs, (1, 0)) # item_idxs is of shape (N, bs)
        sample_log_probs = tf.zeros((batch_size,), dtype=tf.float32, name='sample_log_probs')

        for i in range(N):
            idxs = item_idxs[i] # Indexes for item i in a batch
            logits = item_logits[i] # Logits for item i
            logits = tf.reshape(tf.tile(logits, (batch_size, 1)), (batch_size, -1)) 
            idxs = tf.reshape(idxs, (batch_size,))
            log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=idxs)
            sample_log_probs += log_probs
        
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

    def _count_model_parameters(self):
        tf_trainable_vars = tf.trainable_variables()
        num_params = 0
        # print('Model parameters:')
        for var in tf_trainable_vars:
            # print(str(var))
            num_params += np.prod([dim.value for dim in var.get_shape()])

        # print('Model has {} parameters'.format(num_params))
        return num_params

    def _add_temperature(self, logits, temperature):
        if temperature > 0:
            logits = logits / temperature
        
        return logits
    
    def _add_tanh_constant(self, logits, tanh_constant):
        if tanh_constant > 0:
            logits = tanh_constant * tf.tanh(logits)
        
        return logits

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