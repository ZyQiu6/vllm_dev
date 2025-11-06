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
from .memcache import get_history_trees

@ray.remote(num_cpus=1)
class TreeManager:
    def __init__(self):
        self.memcache = get_history_trees()
        self.predict_times = 0
        self.effective_times = 0
        self.total_right_length = 0

    def __len__(self):
        return len(self._dict)
    
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
        return self.memcache.predict_direct(prompt_id, prefix, accept_length)
    
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
                result.append([self.memcache.predict_direct(prompt_id, prefix_list[i], accept_length_list[i])])
        # print(f"Every time predict costs {time.time() - begin_time} s")
        return result
            
    def compute_metrics(self):
        return {
            'effective_times': self.effective_times,
            'predict_times': self.predict_times,
            'total_right_length': self.total_right_length,
        }

    def clear(self):
        self.memcache.clear()
        self.predict_times = 0
        self.effective_times = 0
        self.total_right_length = 0
    
    def delete(self, prefix):
        return self.memcache.delete_prefix(prefix)