import ray
import math
import msgpack
import diskcache
from enum import Enum
from filelock import FileLock

CACHE_PATH = './output/cache/'

class CongestionState(Enum):
    SLOW_START = 1
    CONGESTION_AVOIDANCE = 2
    SLOW_INCREASE = 3

class MemcachedTrieDirect:
    def __init__(self):
        self.client = diskcache.Cache(CACHE_PATH)
        self.lock = f"{CACHE_PATH}.lock"
        
        self.state = {}
        self.wnd_size = {}
        self.ssthresh = 16
        self.max_wnd = 32
        self.metrics = ray.get_actor(f"spec_metrics")
    
    def _node_key(self, prompt_id, path):
        return f"{prompt_id}_node_{path}"
    
    def _index_key(self, prompt_id, token):
        return f"{prompt_id}_index_{token}"
    
    def _meta_key(self, prompt_id):
        return f"{prompt_id}_meta"
    
    def _set(self, key, value):
        with FileLock(self.lock):
            self.client[key] = value
    
    def find_path_nodes_direct(self, prompt_id, prefix: list) -> list:
        if not prompt_id or not prefix:
            return []

        first_element = prefix[0]
        
        index_key = self._index_key(prompt_id, first_element)
        start_node_paths_data = self.client.get(index_key)
        
        if start_node_paths_data is None:
            return []

        start_node_paths = msgpack.packb(start_node_paths_data)
        match_node_paths = []
        
        for start_node_path in start_node_paths:
            current_path = start_node_path
            match_length = 1
            
            for element in prefix[1:]:
                current_node_data = self.client.get(self._node_key(prompt_id, current_path))
                if current_node_data is None:
                    break
                    
                current_node = msgpack.unpackb(current_node_data)
                
                if element in current_node['children']:
                    child_path = f"{current_path}_{element}"
                    current_path = child_path
                    match_length += 1
                else:
                    break
                if match_length == len(prefix):
                    match_node_paths.append(current_path)
        
        return match_node_paths

    def get_node_direct(self, prompt_id, node_path):
        node_data = self.client.get(self._node_key(prompt_id, node_path))
        if node_data:
            return msgpack.unpackb(node_data)
        return None

    def best_path_node_direct(self, prompt_id, node_paths):
        best_node_path = None
        best_reward = -math.inf
        
        for node_path in node_paths:
            node_data = self.client.get(self._node_key(prompt_id, node_path))
            if node_data:
                node = msgpack.unpackb(node_data)
                if node['reward'] > best_reward:
                    best_reward = node['reward']
                    best_node_path = node_path
        
        if best_node_path:
            best_node = self.get_node_direct(prompt_id, best_node_path)
            best_token = None
            if best_node and best_node['children']:
                for token, child_path in best_node['children'].items():
                    child_node = self.get_node_direct(prompt_id, child_path)
                    if child_node and child_node['reward'] > best_reward:
                        best_reward = child_node['reward']
                        best_token = token
            
            return best_token, best_node_path
        
        return None, None

    def add_path_direct(self, prompt_id, path_tokens, reward: int):
        current_path = "root"
        
        root_key = self._node_key(prompt_id, "root")
        root_data = self.client.get(root_key)
        if not root_data:
            root_node = {'reward': 0, 'children': {}}
            self._set(root_key, msgpack.packb(root_node))
        
        for i, token in enumerate(path_tokens):
            current_node_data = self.client.get(self._node_key(prompt_id, current_path))
            current_node = msgpack.unpackb(current_node_data) if current_node_data else {'reward': 0, 'children': {}}
            
            child_path = f"{current_path}_{token}"
            
            child_key = self._node_key(prompt_id, child_path)
            child_data = self.client.get(child_key)
            child_node = msgpack.unpackb(child_data) if child_data else {'reward': 0, 'children': {}}
            
            child_node['reward'] += reward
            
            current_node['children'][token] = child_path
            self._set(self._node_key(prompt_id, current_path), msgpack.packb(current_node))
            self._set(child_key, msgpack.packb(child_node))
            
            index_key = self._index_key(prompt_id, token)
            index_data = self.client.get(index_key)
            index_nodes = msgpack.unpackb(index_data) if index_data else []
            
            if child_path not in index_nodes:
                index_nodes.append(child_path)
                self._set(index_key, msgpack.packb(index_nodes))
            
            current_path = child_path

    def update_wnd_size(self, prompt_id):
        if self.state[prompt_id] == CongestionState.SLOW_START:
            self.wnd_size[prompt_id] = min(self.wnd_size[prompt_id] * 2, self.ssthresh) if prompt_id in self.wnd_size else 3
            if self.wnd_size[prompt_id] == self.ssthresh:
                self.state[prompt_id] = CongestionState.SLOW_INCREASE
        elif self.state[prompt_id] == CongestionState.CONGESTION_AVOIDANCE:
            self.wnd_size[prompt_id] = max(self.wnd_size[prompt_id] // 2, 3)
        elif self.state[prompt_id] == CongestionState.SLOW_INCREASE:
            self.wnd_size[prompt_id] = min(self.wnd_size[prompt_id] + 1, self.max_wnd)

    def predict_direct(self, prompt_id, prefix, accept_length=1):
        root_key = self._node_key(prompt_id, "root")
        root_data = self.client.get(root_key)
        if root_data is None:
            return []

        self.metrics.add_predict_times.remote(accept_length)
        if prompt_id not in self.state:
            self.state[prompt_id] = CongestionState.SLOW_START
            self.wnd_size[prompt_id] = 3
        else:
            if self.state[prompt_id] == CongestionState.SLOW_START:
                if accept_length == 1:
                    self.state[prompt_id] = CongestionState.CONGESTION_AVOIDANCE
                elif accept_length < self.wnd_size[prompt_id]:
                    self.state[prompt_id] = CongestionState.SLOW_INCREASE
            elif self.state[prompt_id] == CongestionState.CONGESTION_AVOIDANCE:
                if accept_length >= self.wnd_size[prompt_id]:
                    self.state[prompt_id] = CongestionState.SLOW_INCREASE
            elif self.state[prompt_id] == CongestionState.SLOW_INCREASE:
                if accept_length < self.wnd_size[prompt_id]:
                    self.state[prompt_id] = CongestionState.CONGESTION_AVOIDANCE
                else:
                    self.state[prompt_id] = CongestionState.SLOW_INCREASE
        
            self.update_wnd_size(prompt_id)

        predicted_tokens = []
        matched_node_paths = self.find_path_nodes_direct(prompt_id, prefix)
        
        if not matched_node_paths:
            return predicted_tokens
        
        next_token, next_node_path = self.best_path_node_direct(prompt_id, matched_node_paths)
        
        if next_token and next_node_path:
            predicted_tokens.append(next_token)
            current_node_path = next_node_path
            
            for i in range(self.wnd_size - 1):
                current_node = self.get_node_direct(prompt_id, current_node_path)
                if not current_node:
                    break
                    
                best_child_token = None
                best_child_reward = -math.inf
                
                for token, child_path in current_node['children'].items():
                    child_node = self.get_node_direct(prompt_id, child_path)
                    if child_node and child_node['reward'] > best_child_reward:
                        best_child_reward = child_node['reward']
                        best_child_token = token
                        current_node_path = child_path
                
                if best_child_token:
                    predicted_tokens.append(best_child_token)
                else:
                    break
    
        return predicted_tokens

    def clear(self):
        self.clear_tree()
        self.clear_state()

    def clear_tree(self):
        self.client.clear()

    def clear_state(self):
        self.wnd_size.clear()
        self.state.clear()

    def delete_prompt_id(self, prompt_id):
        for key in ray.get(self.metrics.get_prompt_keys.remote(prompt_id)):
            self.client.delete(key)
    
    def compute_metrics(self):
        res = ray.get(self.metrics.compute_metrics.remote())
        ray.get(self.metrics.clear_metrics.remote())
        return res

@ray.remote(num_cpus=1)
class SpecMetrics():
    def __init__(self):
        self.predict_times = 0
        self.effective_times = 0
        self.total_right_length = 0
        self.prompt_id_keys = {}

    def add_prompt_keys(self, prompt_id, key):
        if prompt_id in self.prompt_id_keys:
            self.prompt_id_keys[prompt_id].append(key)
        else:
            self.prompt_id_keys[prompt_id] = [key]

    def get_prompt_keys(self, prompt_id):
        if prompt_id in self.prompt_id_keys:
            key_list = self.prompt_id_keys[prompt_id]
            del self.prompt_id_keys[prompt_id]
            return key_list
        else:
            return []

    def add_predict_times(self, accept_length):
        self.predict_times += 1
        if accept_length > 1:
            self.effective_times += 1
        self.total_right_length += (accept_length - 1)
    
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
    
    def clear_metrics(self):
        self.predict_times = 0
        self.effective_times = 0
        self.total_right_length = 0

@ray.remote(num_cpus=1)
class CacheWriteManager():
    def __init__(self):
        self.history_trees = MemcachedTrieDirect()

    def delete(self, prompt_id):
        self.history_trees.delete_prompt_id(prompt_id)
    
    def tree_append_node(self, prompt_id, seq, reward):
        self.history_trees.add_path_direct(prompt_id, seq, reward)

global_actor = []
def init_history_trees():
    global history_tree_handle
    spec_metrics = SpecMetrics.options(name=f"spec_metrics").remote()
    global_actor.append(spec_metrics)
    cache_write_manager = CacheWriteManager.options(name=f"cache_write_manager").remote()
    global_actor.append(cache_write_manager)
    try:
        ray.get_actor(f"spec_metrics")
        ray.get_actor(f"cache_write_manager")
    except ValueError:
        print(f"HistoSpec global actor failed to register.")

def get_history_trees():
    history_trees = MemcachedTrieDirect()
    return history_trees

def get_cache_write_manager():
    return ray.get_actor(f"cache_write_manager")