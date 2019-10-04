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

from copy import deepcopy

class InvalidDAGError(Exception): pass

def build_dag(sub_train_jobs, ensemble):
    adjacency_list = {}
    ensemble_sub_train_job = None
    if ensemble is not None:
        for sub_train_job in sub_train_jobs:
            if sub_train_job.model_id == ensemble.id:
                ensemble_sub_train_job = sub_train_job

    for sub_train_job in sub_train_jobs:
        if ensemble_sub_train_job is not None and sub_train_job.id == ensemble_sub_train_job.id:
            adjacency_list[ensemble_sub_train_job.id] = []
        else:
            adjacency_list[sub_train_job.id] = [] if ensemble_sub_train_job is None else [ensemble_sub_train_job.id]
    return adjacency_list

def validate_dag(adjacency_list):
    try:
        _get_topological_order(adjacency_list)
        return True
    except InvalidDAGException:
        return False

def get_children(sub_train_job_id, adjacency_list):
    return adjacency_list[sub_train_job_id]

def get_parents(sub_train_job_id, adjacency_list):
    parents = []
    for node, adjacent_nodes in adjacency_list.items():
        if sub_train_job_id in adjacent_nodes:
            parents.append(node)
    return parents

def get_nodes_with_zero_incoming_degrees(adjacency_list):
    nodes_with_zero_incoming_degrees = set(list(adjacency_list.keys()))
    for node, adjacent_nodes in adjacency_list.items():
        for adjacent_node in adjacent_nodes:
            nodes_with_zero_incoming_degrees.discard(adjacent_node)
    return list(nodes_with_zero_incoming_degrees)

def _get_topological_order(adjacency_list):
    adjacency_list = deepcopy(adjacency_list)
    queue = get_nodes_with_zero_incoming_degrees(adjacency_list)
    topological_order = []

    while queue:
        node = queue.pop()
        topological_order.append(node)

        adjacency_list.pop(node, None)
        for node in get_nodes_with_zero_incoming_degrees(adjacency_list):
            if node not in queue:
                queue.append(node)
    
    if adjacency_list:
        raise InvalidDAGException
    else:
        return topological_order  