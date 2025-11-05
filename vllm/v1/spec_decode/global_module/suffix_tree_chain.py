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


# 仅包含数据结构的后缀树类（用于ray.put传输）
class SuffixTree:
    def __init__(self):
        """
        - root: TrieNode, 后缀树的根节点
        - subpath_index: dict, key: token, value: 所有以该token为key的节点的引用的列表
        """
        self.root = TrieNode()
        self.subpath_index = {} # use tokens to represent nodes


# 后缀树类（为每个prompt维护一个）（实际上是前缀树）
class SuffixTreePack:
    def __init__(self):
        """
        属性
        - tree: SuffixTree, 后缀树数据结构
        - state: CongestionState, 当前拥塞状态
        - wnd_size: int, 该次预测的token数量
        - ssthresh: int, 慢启动阈值，从慢启动阶段过渡到慢增长阶段的临界值
        - max_wnd: int, 最大窗口大小
        """
        self.tree = SuffixTree()
        self.tree_ray_handle: ray.ObjectRef = None
        self.state = CongestionState.SLOW_START
        self.wnd_size: int = 3
        self.ssthresh = 16
        self.max_wnd = 28
        self.spec_enable = True
        
    # 检查路径path是否存在与该后缀树中
    def exist(self, path):
        node = self.tree.root
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
        node = self.tree.root
        for char in path:
            if char not in node.children:
                node.children[char] = TrieNode()
                if char not in self.tree.subpath_index:
                    self.tree.subpath_index[char] = [node.children[char]]
                else:
                    self.tree.subpath_index[char].append(node.children[char])
            node = node.children[char]
            node.reward += reward
            
    # 清空树
    def clear(self):
        self.tree.root.clear()
        self.tree.subpath_index.clear()


# 检查某个前缀prefix在树中出现的位置，返回所有匹配的节点列表
def find_path_nodes(tree: SuffixTree, prefix: list) -> list:
    """
    Fast search path. Prefix may not start from root
    """
    # prefix前缀序列，即已经根据prompt生成的token序列的后缀
    if not prefix:
        return []
    first_element = prefix[0]
    if first_element not in tree.subpath_index:
        return []
    start_nodes = tree.subpath_index[first_element]
    match_nodes = []
    
    # O(N_starts * L) where N_starts is the number of starting nodes and L isprefix length
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
def update_wnd_size(wnd_size, state, ssthresh, max_wnd):
    if state == CongestionState.SLOW_START:
        wnd_size = min(wnd_size * 2, ssthresh)
        if wnd_size == ssthresh:
            state = CongestionState.SLOW_INCREASE
    elif state == CongestionState.CONGESTION_AVOIDANCE:
        wnd_size = max(wnd_size // 2, 3)
    elif state == CongestionState.SLOW_INCREASE:
        wnd_size = min(wnd_size + 1, max_wnd)
    return wnd_size, state


# 完成更新wnd_size，推测解码，返回预测的token序列
@ray.remote
def predict(suffix_tree_handle: ray.ObjectRef, wnd_size, state, ssthresh, max_wnd, prefix, accept_length=1):
    # suffix_tree_handle, wnd_size, state, ssthresh, max_wnd来自SuffixTreePack类的属性
    suffix_tree = ray.get(suffix_tree_handle)

    if state == CongestionState.SLOW_START:
        if accept_length == 1:
            state = CongestionState.CONGESTION_AVOIDANCE
        elif accept_length < wnd_size:
            state = CongestionState.SLOW_INCREASE
    elif state == CongestionState.CONGESTION_AVOIDANCE:
        if accept_length >= wnd_size:
            state = CongestionState.SLOW_INCREASE
    elif state == CongestionState.SLOW_INCREASE:
        if accept_length < wnd_size:
            state = CongestionState.CONGESTION_AVOIDANCE
        else:
            state = CongestionState.SLOW_INCREASE
    wnd_size, state = update_wnd_size(wnd_size, state, ssthresh, max_wnd)
    predicted_tokens = []
    
    matched_nodes = find_path_nodes(suffix_tree, prefix)
    # print(f"predict matched nodes num: {len(matched_nodes)}")
    next_token, next_node = best_path_node(matched_nodes)
    if next_token:
        predicted_tokens.append(next_token)
        for i in range(wnd_size - 1):
            next_token, next_node = next_node.best_child()
            if next_token:
                predicted_tokens.append(next_token)
            else:
                break
    
    return wnd_size, state, predicted_tokens


class SuffixTreeGroup:
    def __init__(self):
        # dict：key: prompt_id（prompt字符串）, value: RewardAwareSuffixTree实例
        self.dict: dict[str, SuffixTreePack] = {}
        self.predict_times = 0
        self.effective_times = 0
        self.total_right_length = 0
        self.prompt_ids = set()
    
    def __len__(self):
        return len(self.dict)
    
    # 为prompt新建维护一个树
    def add_tree(self, prompt_id):
        if prompt_id in self._dict:
            return
        self.dict[prompt_id] = SuffixTreePack()
    
    # 将新rollout出的序列更新进树中
    def tree_append_node(self, prompt_id, seq, reward):
        if prompt_id not in self.dict:
            raise ValueError(f"{prompt_id} not in GlobalRewardAwareSuffixTreeGroup")
        else:
            self.dict[prompt_id].add_node(seq, reward)
    
    # 根据prefix前缀，预测接下来的accept_length个token
    def predict(self, prompt_id, prefix, accept_length):
        self.predict_times += 1
        if accept_length > 1:
            self.effective_times += 1
        self.total_right_length += (accept_length - 1)
        predict_ref = predict.remote(
            self.dict[prompt_id].tree_ray_handle,
            self.dict[prompt_id].wnd_size,
            self.dict[prompt_id].state,
            self.dict[prompt_id].ssthresh,
            self.dict[prompt_id].max_wnd,
            prefix,
            accept_length
        )
        self.dict[prompt_id].wnd_size, self.dict[prompt_id].state, predict_tokens = ray.get(predict_ref)
        return predict_tokens
    
    def set(self, key, value):
        self.dict[key] = value
    
    def get(self, key, default=None):
        return self.dict.get(key, default)

    def exist(self, key):
        return key in self.dict

    def get_prompt_ids(self):
        return list(self.dict.keys())
    
    def delete(self, key):
        if key in self.dict:
            del self.dict[key].tree_ray_handle
            del self.dict[key]
            
    def compute_metrics(self):
        effective_percent = self.effective_times / self.predict_times if self.predict_times != 0 \
                                else 0
        effective_length = self.total_right_length / self.effective_times if self.effective_times != 0 \
                                else 0
        res = {
            'speculative_decoding/effective_percent': effective_percent,
            'speculative_decoding/effective_length': effective_length,
        }
        return res

    def clear(self):
        for i in self.dict.keys():
            del self.dict[i].tree_ray_handle
        self.dict.clear()
        self.predict_times = 0
        self.effective_times = 0
        self.total_right_length = 0
        
    def update_prompt_ids(self):
        prompt_ids = self.get_prompt_ids()
        self.prompt_ids = set(prompt_ids)


def init_history_trees():
    suffix_tree = SuffixTreeGroup()
    if isinstance(suffix_tree, SuffixTreeGroup):
        print(f"Instance registered successfully.")
    else:
        print(f"Instance failed to create.")


def get_history_trees():
    history_trees = SuffixTreeGroup()
    return history_trees