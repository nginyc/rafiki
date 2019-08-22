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
import numpy as np
import logging
from collections import defaultdict

from rafiki.constants import BudgetOption
from rafiki.model import ArchKnob, FixedKnob, PolicyKnob

from .constants import ParamsType
from .advisor import BaseAdvisor, Proposal

logger = logging.getLogger(__name__)

ENAS_BATCH_SIZE = 10
ENAS_NUM_EVAL_PER_CYCLE = 300
ENAS_NUM_FINAL_EVALS = 10
ENAS_FINAL_HOURS = 12 # Last X hours to conduct final evals & final trains
ENAS_NUM_FINAL_TRAINS = 1

class EnasAdvisor(BaseAdvisor):
    '''
    Performs cell-based "Efficient Neural Architecture Search via Parameter Sharing" (ENAS) for image classification.
    In a distributed setting, performs ENAS locally at each worker using LOCAL_RECENT parameters.
    
    Original paper: https://arxiv.org/abs/1802.03268
    '''
    @staticmethod
    def is_compatible(knob_config, budget):
        # Supports only FixedKnob, ArchKnob and PolicyKnob
        # And must have param sharing, quick train, quick eval, skip train & downscale
        # And time budget must be sufficient to cover final evals & final trains
        time_hours = float(budget.get(BudgetOption.TIME_HOURS, 0))
        return BaseAdvisor.has_only_knob_types(knob_config, [FixedKnob, ArchKnob, PolicyKnob]) and \
            BaseAdvisor.has_policies(knob_config, ['SHARE_PARAMS', 'DOWNSCALE', 'EARLY_STOP', 'QUICK_EVAL', 'SKIP_TRAIN']) and \
            time_hours >= ENAS_FINAL_HOURS 

    def __init__(self, knob_config, budget):
        super().__init__(knob_config, budget)
        self._batch_size = ENAS_BATCH_SIZE
        self._num_eval_per_cycle = ENAS_NUM_EVAL_PER_CYCLE
        self._num_final_evals = ENAS_NUM_FINAL_EVALS
        self._num_final_trains = ENAS_NUM_FINAL_TRAINS
        self._final_hours = ENAS_FINAL_HOURS
        self._final_evals = [] # [(score, proposal)]
        self._final_trains = []  # [(score, proposal)]
        self._worker_to_num_trials = defaultdict(int) # { <worker_id>: <how many trials has worker run?> }
        (self._fixed_knob_config, knob_config) = self.extract_knob_type(knob_config, FixedKnob)
        (self._policy_knob_config, knob_config) = self.extract_knob_type(knob_config, PolicyKnob)
        self._list_knob_models = self._build_models(knob_config, self._batch_size)

    def propose(self, worker_id, trial_no):
        proposal_type = self._get_proposal_type(worker_id)
        meta = {'proposal_type': proposal_type}

        # Keep track of trial at each worker
        self._worker_to_num_trials[worker_id] += 1

        if proposal_type == 'TRAIN':
            knobs = self._propose_knobs(['DOWNSCALE', 'EARLY_STOP'])
            return Proposal(trial_no, knobs,
                            params_type=ParamsType.LOCAL_RECENT, 
                            to_eval=False, 
                            to_cache_params=True, 
                            to_save_params=False, 
                            meta=meta)
        elif proposal_type == 'EVAL':
            knobs = self._propose_knobs(['DOWNSCALE', 'QUICK_EVAL', 'SKIP_TRAIN'])
            return Proposal(trial_no, knobs, 
                            params_type=ParamsType.LOCAL_RECENT, 
                            to_save_params=False, 
                            meta=meta)
        elif proposal_type == 'FINAL_EVAL':
            knobs = self._propose_knobs(['DOWNSCALE', 'SKIP_TRAIN'])
            return Proposal(trial_no, knobs, 
                            params_type=ParamsType.LOCAL_RECENT, 
                            meta=meta)
        elif proposal_type == 'FINAL_TRAIN':
            # Do standard model training from scratch with final knobs
            knobs = self._propose_final_knobs()
            return Proposal(trial_no, knobs, meta=meta)
        elif proposal_type is None:
            return None

    def feedback(self, worker_id, result):
        proposal = result.proposal
        knobs = proposal.knobs
        score = result.score
        proposal_type = proposal.meta.get('proposal_type') 

        # Ignore null scores
        if score is None:
            return

        # Keep track of results of final evals & trains
        if proposal_type == 'FINAL_EVAL':
            self._final_evals.append((score, proposal))
        elif proposal_type == 'FINAL_TRAIN':
            self._final_trains.append((score, proposal))

        for (name, list_knob_model) in self._list_knob_models.items():
            knob_value = knobs[name]
            list_knob_model.feedback(knob_value, score)

    def _propose_final_knobs(self, policies=None):
        final_evals = self._final_evals

        # If hasn't collected any evals, propose from model
        if len(final_evals) == 0:
            return self._propose_knobs(policies)

        # Otherwise, determine best recent proposal and use it
        final_evals.sort(key=lambda x: x[0])
        (score, proposal) = final_evals.pop()
        knobs = proposal.knobs

        # Add policy knobs
        knobs = self.merge_policy_knobs(knobs, self._policy_knob_config, policies or [])

        return knobs

    def _propose_knobs(self, policies=None):
        knobs = {}
        for (name, list_knob_model) in self._list_knob_models.items():
            knobs[name] = list_knob_model.propose()

        # Add fixed & policy knobs
        knobs = self.merge_fixed_knobs(knobs, self._fixed_knob_config)
        knobs = self.merge_policy_knobs(knobs, self._policy_knob_config, policies or [])
        
        return knobs

    def _build_models(self, knob_config, batch_size):
        list_knobs = [(name, knob) for (name, knob) in knob_config.items()]

        # Build a model for each list knob
        list_knob_models = {}
        for (name, list_knob) in list_knobs:
            with tf.variable_scope(name):
                list_knob_models[name] = EnasArchAdvisor(list_knob, batch_size)

        return list_knob_models

    def _get_proposal_type(self, worker_id):
        # If we have enough final trains, we stop
        if len(self._final_trains) >= self._num_final_trains:
            return None

        T = self._num_eval_per_cycle + 1 # Period
        worker_num_trials = self._worker_to_num_trials[worker_id]
        local_trial_no = worker_num_trials + 1

        # Schedule: |--<train & eval cycles>---||--<final evals> ---||--<final train>--|

        # If it's final hours, perform final evals & trains
        if self.get_train_hours_left() <= self._final_hours:
            # If we have enough final evals, we perform final trains
            # Otherwise, perform more final evals
            if len(self._final_evals) >= self._num_final_evals:
                return 'FINAL_TRAIN'
            else:
                return 'FINAL_EVAL'

        # If not final eval or train, it is in the <train + eval> phase
        # Every (X + 1) trials, only train 1 epoch for the first trial
        # The other X trials is for training the controller
        # Use *local* trial no to determine this
        # Since trials on each worker are executed relatively sequentially, no need to check validity
        is_train_trial = ((local_trial_no - 1) % T == 0)

        if is_train_trial:
            return 'TRAIN'
        else:
            return 'EVAL'


class EnasArchAdvisor():
    def __init__(self, knob, batch_size):
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        self._knob = knob
        self._batch_size = batch_size
        self._items_batch = [] # A running batch of items for feedback
        self._scores_batch = [] # A running batch of corresponding scores for feedback

        with self._graph.as_default():
            (self._item_logits, self._out_item_idxs, 
                self._train_op, self._losses, self._rewards,
                self._item_idxs_ph, self._scores_ph) = self._build_model(self._knob)
            self._start_session()

    def propose(self):
        items = self._predict_with_model()
        return items

    def feedback(self, items, score):
        self._items_batch.append(items)
        self._scores_batch.append(score)
        
        if len(self._items_batch) < self._batch_size:
            return

        self._train_model(self._items_batch, self._scores_batch)
        self._items_batch = []
        self._scores_batch = []

    def _predict_with_model(self):
        knob_items = self._knob.items
        out_item_idxs = self._out_item_idxs

        with self._graph.as_default():
            item_idxs_real = self._sess.run(out_item_idxs)
            items = [values[idx].value for (values, idx) in zip(knob_items, item_idxs_real)]
            return items

    def _train_model(self, items_batch, batch_scores):
        knob_items = self._knob.items

        # Convert item values to indexes
        batch_item_idxs = [[[x.value for x in values].index(value) for (values, value) in zip(knob_items, items)] for items in items_batch]

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
        assert isinstance(knob, ArchKnob)

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
        for values in knob.items:
            for knob_value in values:
                if knob_value in knob_value_to_rep:
                    continue

                knob_value_to_rep[knob_value] = max_rep
                max_rep += 1

        value_reps_by_item = [[knob_value_to_rep[x] for x in values] for values in knob.items]
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