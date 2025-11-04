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
import math
from enum import Enum

# 遍历nodes中的所有节点，找到其中分数最高的子节点及其代表的token
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


# 后缀树节点类
class TrieNode:
    def __init__(self):
        self.children = {}  # 直接子节点构成的字典，key为该子节点所代表的token，value为该子节点实例
        self.reward = 0 # 当前节点的分数
        
    # 遍历当前节点的所有子节点，找到分数最高的子节点及其代表的token
    def best_child(self):
        best_child = None
        best_token = None
        max_reward = -math.inf
        for key in self.children.keys():
            if self.children[key].reward > max_reward:
                best_token = key
                best_child = self.children[key]
        return best_token, best_child
    
    # 递归清除当前节点的子节点和当前节点，及其分数
    def clear(self):
        self.children.clear()
        self.reward = 0


# 拥塞状态
class CongestionState(Enum):
    SLOW_START = 1
    CONGESTION_AVOIDANCE = 2
    SLOW_INCREASE = 3


# 后缀树类（为每个prompt维护一个）（实际上是前缀树）
class RewardAwareSuffixTree:
    def __init__(self):
        """
        属性
        - root: TrieNode, 后缀树的根节点
        - subpath_index: dict, key: token, value: 所有以该token为key的节点的引用的列表
        - state: CongestionState, 当前拥塞状态
        - wnd_size: int, 该次预测的token数量
        - ssthresh: int, 慢启动阈值，从慢启动阶段过渡到慢增长阶段的临界值
        - max_wnd: int, 最大窗口大小
        """
        self.root = TrieNode()
        self.subpath_index = {} # use tokens to represent nodes
        self.state = CongestionState.SLOW_START
        self.wnd_size: int = 3 # number of predicted tokens
        self.ssthresh = 16
        self.max_wnd = 28
        self.spec_enable = True
        
    # 检查路径path是否存在与该后缀树中
    def exist(self, path):
        node = self.root
        for char in path:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    # 递归将一个路径path加入树中，更新树的subpath_index和每个节点的children和reward
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
    
    # 检查某个前缀prefix在树中出现的位置，返回所有匹配的节点列表（感觉性能不好，主要原因是前缀树使用subpath_index查找）
    def find_path_nodes(self, prefix: list) -> list:
        """
        Fast search path. Prefix may not start from root
        """
        # prefix前缀序列，即已经根据prompt生成的token序列的后缀
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
    
    # 更新本次预测的token数量
    def update_wnd_size(self):
        if self.state == CongestionState.SLOW_START:
            self.wnd_size = min(self.wnd_size * 2, self.ssthresh)
            if self.wnd_size == self.ssthresh:
                self.state = CongestionState.SLOW_INCREASE
        elif self.state == CongestionState.CONGESTION_AVOIDANCE:
            self.wnd_size = max(self.wnd_size // 2, 3)
        elif self.state == CongestionState.SLOW_INCREASE:
            self.wnd_size = min(self.wnd_size + 1, self.max_wnd)

    # 完成更新wnd_size，推测解码，返回预测的token序列
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
    
    # 清空树
    def clear(self):
        self.root.clear()
        self.subpath_index.clear()


@ray.remote(num_cpus=1)
class SuffixTreeGroup:
    def __init__(self):
        # _dict：key: prompt_id（prompt字符串）, value: RewardAwareSuffixTree实例
        self._dict: dict[str, RewardAwareSuffixTree] = {}
        self.predict_times = 0
        self.effective_times = 0
        self.total_right_length = 0

    def __len__(self):
        return len(self._dict)
    
    # 为prompt新建维护一个树
    def add_tree(self, prompt_id):
        self._dict[prompt_id] = RewardAwareSuffixTree()
    
    # 将新rollout出的序列更新进树中
    def tree_append_node(self, prompt_id, seq, reward):
        if prompt_id not in self._dict:
            raise ValueError(f"{prompt_id} not in GlobalRewardAwareSuffixTreeGroup")
        else:
            self._dict[prompt_id].add_node(seq, reward)
    
    # 根据prefix前缀，预测接下来的accept_length个token
    def predict(self, prompt_id, prefix, accept_length):
        self.predict_times += 1
        if accept_length > 1:
            self.effective_times += 1
        self.total_right_length += (accept_length - 1)
        return self._dict[prompt_id].predict(prefix, accept_length)
    
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


# 为多个SuffixTreeGroup类封装了对外接口
_num_groups: int = 4 # fixed
history_tree_handle = []
class GlobalRewardAwareSuffixTreeGroup:
    """
    All functions are non-blocking, return futures.
    """
    def __init__(self):
        self.groups = []
        self.prompt_ids = set()
        for i in range(_num_groups):
            try:
                # actor_handle：ray actor的引用句柄
                actor_handle = ray.get_actor(f"global_tree_{i}")
                self.groups.append(actor_handle)
            except ValueError:
                print(f"Could not find the global actor.")

    def update_prompt_ids(self):
        prompt_ids = []
        for actor_handle in self.groups:
            prompt_ids.extend(ray.get(actor_handle.get_prompt_ids.remote()))
        self.prompt_ids = set(prompt_ids)

    def __len__(self):
        return len(self.groups)

    def _get_partition(self, prompt_id: str):
        group_index = hash(prompt_id) % _num_groups
        return self.groups[group_index]
    
    def add_tree(self, prompt_id):
        actor = self._get_partition(prompt_id)
        return actor.add_tree.remote(prompt_id)
    
    def tree_append_node(self, prompt_id, seq, reward):
        actor = self._get_partition(prompt_id)
        return actor.tree_append_node.remote(prompt_id, seq, reward)

    def predict(self, prompt_id, prefix, accept_length):
        actor = self._get_partition(prompt_id)
        return actor.predict.remote(prompt_id, prefix, accept_length)

    def delete(self, prompt_id):
        actor = self._get_partition(prompt_id)
        return actor.delete.remote(prompt_id)

    def exist(self, prompt_id):
        # return (prompt_id in self.prompt_ids)
        actor = self._get_partition(prompt_id)
        return actor.exist.remote(prompt_id)

    def clear(self):
        return [p.clear.remote() for p in self.groups]
    
    def compute_metrics(self):
        tasks = [group.compute_metrics.remote() for group in self.groups]
        metrics_list = ray.get(tasks)
        data = {}
        for key in metrics_list[0].keys():
            data[key] = sum([metrics[key] for metrics in metrics_list])
        effective_percent = data['effective_times'] / data['predict_times'] if data['predict_times'] != 0 \
                                else 0
        effective_length = data['total_right_length'] / data['effective_times'] if data['effective_times'] != 0 \
                                else 0
        res = {
            'speculative_decoding/effective_percent': effective_percent,
            'speculative_decoding/effective_length': effective_length,
        }
        return res


def init_history_trees():
    global history_tree_handle
    for i in range(_num_groups):
        actor_handle = SuffixTreeGroup.options(name=f"global_tree_{i}").remote()
        history_tree_handle.append(actor_handle)
    for i in range(_num_groups):
        try:
            ray.get_actor(f"global_tree_{i}")
            print(f"Actor {i} registered successfully.")
        except ValueError:
            print(f"Actor {i} failed to register.")


def get_history_trees():
    history_trees = GlobalRewardAwareSuffixTreeGroup()
    return history_trees