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
def best_path_node(tree: 'SuffixTree', node_ids: list[int]):
    """从多个候选节点出发，选择其所有子节点中reward最大的下一跳。
    返回 (best_token, best_child_id)。若无子节点，返回 (None, None)。"""
    best_child_id = None
    best_token = None
    max_reward = -math.inf
    for node_id in node_ids:
        children_map = tree.parent_to_children.get(node_id, {})
        for token, child_id in children_map.items():
            child_reward = tree.nodes[child_id]['reward']
            if child_reward > max_reward:
                max_reward = child_reward
                best_token = token
                best_child_id = child_id
    return best_token, best_child_id


# 拥塞状态
class CongestionState(Enum):
    SLOW_START = 1
    CONGESTION_AVOIDANCE = 2
    SLOW_INCREASE = 3


# 后缀树类（用于ray.put传输）（实际上是前缀树）
class SuffixTree:
    def __init__(self):
        """
        - next_node_id: 分配下一节点的id
        - nodes: 节点表，包含所有节点的一些信息，key：node_id，value：每个value是一个含"parent_id", "token", "reward"三个key的字典
        - children: 邻接表，key：(parent_id, token)元组，value：child_id（用于查找parent是否有的指定token的children）
        - parent_to_children: 父到子索引，key：parent_id，value：{token -> child_id}字典（用于查找parent的所有children的token->id信息）
        - token_to_node_ids: key：token，value：包含该token的节点ID列表（原subpath_index）
        """
        self.next_node_id = 1  # 0保留给root
        self.nodes = {
            0: {
                'parent_id': -1,
                'token': None,
                'reward': 0,
            }
        }
        self.children = {}
        self.parent_to_children = {}
        self.token_to_node_ids = {}
    
    def __getstate__(self):
        """直接返回已维护的扁平结构"""
        return {
            'next_node_id': self.next_node_id,
            'nodes': self.nodes,
            'children': self.children,
            'parent_to_children': self.parent_to_children,
            'token_to_node_ids': self.token_to_node_ids,
        }
    
    def __setstate__(self, state):
        """恢复扁平结构"""
        self.next_node_id = state['next_node_id']
        self.nodes = state['nodes']
        self.children = state['children']
        self.parent_to_children = state.get('parent_to_children', {})
        self.token_to_node_ids = state.get('token_to_node_ids', {})

    def exist_path(self, path: list) -> bool:
        if not path:
            return False
        current_id = 0
        for token in path:
            key = (current_id, token)
            if key not in self.children:
                return False
            current_id = self.children[key]
        return True

    def add_path(self, path: list, reward: float) -> None:
        if not path:
            return
        current_id = 0
        for token in path:
            key = (current_id, token)
            if key not in self.children:
                new_id = self.next_node_id
                self.next_node_id += 1
                # 更新children
                self.children[key] = new_id
                # 更新parent_to_children
                if current_id not in self.parent_to_children:
                    self.parent_to_children[current_id] = {}
                self.parent_to_children[current_id][token] = new_id
                # 更新nodes
                self.nodes[new_id] = {
                    'parent_id': current_id,
                    'token': token,
                    'reward': 0,
                }
                # 更新token_to_node_ids
                self.token_to_node_ids.setdefault(token, []).append(new_id) # 如果不存在该键值对则创建
            current_id = self.children[key]
            self.nodes[current_id]['reward'] += reward

    def clear_flat(self):
        self.next_node_id = 1
        self.nodes = {
            0: {
                'parent_id': -1,
                'token': None,
                'reward': 0,
            }
        }
        self.children = {}
        self.parent_to_children = {}
        self.token_to_node_ids = {}


# 后缀树类封装（为每个prompt维护一个）
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
        
    # 检查路径path是否存在于该后缀树中
    def exist(self, path):
        return self.tree.exist_path(path)
    
    # 递归将一个路径path加入树中，更新树的subpath_index和每个节点的children和reward
    def add_node(self, path, reward):
        """
        :param path: List of token
        :param reward: reward score
        """
        if self.exist(path):
            return
        self.tree.add_path(path, reward)
            
    # 清空树
    def clear(self):
        self.tree.clear_flat()


# 检查某个前缀prefix在树中出现的位置，返回所有匹配的节点列表
def find_path_nodes(tree: SuffixTree, prefix: list) -> list:
    # prefix前缀序列，即已经根据prompt生成的token序列的后缀
    if not prefix:
        return []
    first_element = prefix[0]
    start_ids = tree.token_to_node_ids.get(first_element, [])
    match_ids = []
    # O(N_starts * L)
    for start_id in start_ids:
        current_id = start_id
        match_length = 1
        ok = True
        for element in prefix[1:]:
            next_id = tree.parent_to_children.get(current_id, {}).get(element)
            if next_id is None:
                ok = False
                break
            current_id = next_id
            match_length += 1
        if ok and match_length == len(prefix):
            match_ids.append(current_id)
    return match_ids


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
    
    matched_node_ids = find_path_nodes(suffix_tree, prefix)
    next_token, next_node_id = best_path_node(suffix_tree, matched_node_ids)
    if next_token is not None:
        predicted_tokens.append(next_token)
        # 下面沿着单条最大奖励路径逐步扩展
        current_id = next_node_id
        for i in range(wnd_size - 1):
            children_map = suffix_tree.parent_to_children.get(current_id, {})
            if not children_map:
                break
            # 选择当前节点的最佳子节点
            best_tok = None
            best_child = None
            best_reward = -math.inf
            for token, id in children_map.items():
                r = suffix_tree.nodes[id]['reward']
                if r > best_reward:
                    best_reward = r
                    best_tok = token
                    best_child = id
            if best_tok is None:
                break
            predicted_tokens.append(best_tok)
            current_id = best_child
    
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
        if prompt_id in self.dict:
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
        return predict.remote(
            self.dict[prompt_id].tree_ray_handle,
            self.dict[prompt_id].wnd_size,
            self.dict[prompt_id].state,
            self.dict[prompt_id].ssthresh,
            self.dict[prompt_id].max_wnd,
            prefix,
            accept_length
        )
    
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