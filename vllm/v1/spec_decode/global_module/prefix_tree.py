# Copyright 2025 Ziyi Qiu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ray
import zmq
import math
import msgpack
from enum import Enum

def best_path_node(nodes):
    best_child = None
    best_token = None
    max_reward = -math.inf
    for node in nodes:
        for key in node.children.keys():
            if node.children[key].reward > max_reward:
                best_token = key
                best_child = node.children[key]
    return best_token, best_child

class TrieNode:
    def __init__(self):
        self.children = {}
        self.reward = 0
        
    def best_child(self):
        best_child = None
        best_token = None
        max_reward = -math.inf
        for key in self.children.keys():
            if self.children[key].reward > max_reward:
                best_token = key
                best_child = self.children[key]
        return best_token, best_child
    
    def clear(self):
        self.children.clear()
        self.reward = 0


class CongestionState(Enum):
    SLOW_START = 1
    CONGESTION_AVOIDANCE = 2
    SLOW_INCREASE = 3

class RewardAwareSuffixTree:
    def __init__(self):
        self.root = TrieNode()
        self.subpath_index = {} # use tokens to represent nodes
        self.state = CongestionState.SLOW_START
        self.wnd_size: int = 3 # number of predicted tokens
        self.ssthresh = 16
        self.max_wnd = 28
        self.spec_enable = True
        
    def exist(self, path):
        node = self.root
        for char in path:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def add_node(self, path, reward):
        """
        :param path: List of token
        :param reward: reward score
        """
        if self.exist(path):
            return
        node = self.root
        for char in path:
            if char not in node.children:
                node.children[char] = TrieNode()
                if char not in self.subpath_index:
                    self.subpath_index[char] = [node.children[char]]
                else:
                    self.subpath_index[char].append(node.children[char])
            node = node.children[char]
            node.reward += reward
    
    def find_path_nodes(self, prefix: list) -> list:
        """
        Fast search path. Prefix may not start from root
        """
        if not prefix:
            return []

        first_element = prefix[0]
        if first_element not in self.subpath_index:
            return []

        start_nodes = self.subpath_index[first_element]
        match_nodes = []
        
        # O(N_starts * L) where N_starts is the number of starting nodes and L is prefix length
        for start_node in start_nodes:
            current_node = start_node
            match_length = 1
            for element in prefix[1:]:
                if element in current_node.children:
                    current_node = current_node.children[element]
                    match_length += 1
                else:
                    break
            if match_length == len(prefix):
                match_nodes.append(current_node)

        return match_nodes
    
    def update_wnd_size(self):
        if self.state == CongestionState.SLOW_START:
            self.wnd_size = min(self.wnd_size * 2, self.ssthresh)
            if self.wnd_size == self.ssthresh:
                self.state = CongestionState.SLOW_INCREASE
        elif self.state == CongestionState.CONGESTION_AVOIDANCE:
            self.wnd_size = max(self.wnd_size // 2, 3)
        elif self.state == CongestionState.SLOW_INCREASE:
            self.wnd_size = min(self.wnd_size + 1, self.max_wnd)

    def predict(self, prefix, accept_length=1):
        if self.state == CongestionState.SLOW_START:
            if accept_length == 1:
                self.state = CongestionState.CONGESTION_AVOIDANCE
            elif accept_length < self.wnd_size:
                self.state = CongestionState.SLOW_INCREASE
        elif self.state == CongestionState.CONGESTION_AVOIDANCE:
            if accept_length >= self.wnd_size:
                self.state = CongestionState.SLOW_INCREASE
        elif self.state == CongestionState.SLOW_INCREASE:
            if accept_length < self.wnd_size:
                self.state = CongestionState.CONGESTION_AVOIDANCE
            else:
                self.state = CongestionState.SLOW_INCREASE
        self.update_wnd_size()

        predicted_tokens = []
        
        matched_nodes = self.find_path_nodes(prefix)
        # print(f"predict matched nodes num: {len(matched_nodes)}")
        next_token, next_node = best_path_node(matched_nodes)
        if next_token:
            predicted_tokens.append(next_token)
            for i in range(self.wnd_size - 1):
                next_token, next_node = next_node.best_child()
                if next_token:
                    predicted_tokens.append(next_token)
                else:
                    break
        
        return predicted_tokens
    
    def clear(self):
        self.root.clear()
        self.subpath_index.clear()

@ray.remote(num_cpus=1)
class SuffixTreeGroup:
    def __init__(self, port: 5555):
        self._dict: dict[str, RewardAwareSuffixTree] = {}
        self.predict_times = 0
        self.effective_times = 0
        self.total_right_length = 0
        self.port = port

    def __len__(self):
        return len(self._dict)
    
    def add_tree(self, prompt_id):
        if prompt_id in self._dict:
            return
        self._dict[prompt_id] = RewardAwareSuffixTree()
    
    def tree_append_node(self, prompt_id, seq, reward):
        if prompt_id not in self._dict:
            raise ValueError(f"{prompt_id} not in GlobalRewardAwareSuffixTreeGroup")
        else:
            self._dict[prompt_id].add_node(seq, reward)
    
    def _predict(self, prompt_id, prefix, accept_length):
        self.predict_times += 1
        if accept_length > 1:
            self.effective_times += 1
        self.total_right_length += (accept_length - 1)
        return self._dict[prompt_id].predict(prefix, accept_length)
    
    def predict(self, prompt_id, prefix, accept_length):
        if (not prompt_id in self._dict) or (not prefix):
            return []
        else:
            return self._predict(prompt_id, prefix, accept_length)
    
    def predict_batch(self, prompt_id_list, prefix_list, accept_length_list):
        import time
        begin_time = time.time()
        result = []
        for i in range(len(prompt_id_list)):
            prompt_id = prompt_id_list[i]
            if (not prompt_id in self._dict) or (not prefix_list[i]):
                result.append([])
            else:
                self.predict_times += 1
                if accept_length_list[i] > 1:
                    self.effective_times += 1
                self.total_right_length += (accept_length_list[i] - 1)
                result.append([self._dict[prompt_id].predict(prefix_list[i], accept_length_list[i])])
        # print(f"Every time predict costs {time.time() - begin_time} s")
        return result
    
    def set(self, key, value):
        self._dict[key] = value
    
    def get(self, key, default=None):
        return self._dict.get(key, default)

    def exist(self, key):
        return key in self._dict

    def get_prompt_ids(self):
        return list(self._dict.keys())
    
    def delete(self, key):
        if key in self._dict:
            del self._dict[key]
            
    def compute_metrics(self):
        return {
            'effective_times': self.effective_times,
            'predict_times': self.predict_times,
            'total_right_length': self.total_right_length,
        }

    def clear(self):
        self._dict.clear()
        self.predict_times = 0
        self.effective_times = 0
        self.total_right_length = 0