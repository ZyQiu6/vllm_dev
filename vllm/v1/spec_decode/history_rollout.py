from typing import Optional

import ray
import numpy as np

from vllm.config import VllmConfig
from vllm.v1.spec_decode.global_module.suffix_tree import get_history_trees

class HistoryRolloutProposer:
    def __init__(self, vllm_config: VllmConfig):
        # Minimum length of the HistoryRolloutTree to match.
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        # Maximum length of the HistoryRolloutTree to match.
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        # self.k = vllm_config.speculative_config.num_speculative_tokens

    def propose(
        self,
        accept_length: int,
        sampled_token_ids: list[int],
        prompt_token_ids: list[int]
    ) -> Optional[np.ndarray]:
        """Proposes the next sequence of tokens based on history rollout
        speculative decoding pattern.
        """
        prompt_id = str(hash(tuple(prompt_token_ids)))
        history_trees = get_history_trees()
        if not history_trees.exist(prompt_id):
            return []
        else:
            print(f"{prompt_id} found in history_trees")
        history_tree = ray.get(history_trees.get.remote(prompt_id))
        draft_tokens = []
        prefix_len_candidates = range(self.max_n, self.min_n, -1)
        for prefix_len in prefix_len_candidates:
            if len(sampled_token_ids) >= prefix_len:
                prefix = sampled_token_ids[-prefix_len:]
                draft_tokens = history_tree.predict(history_tree, prefix, accept_length)
                if draft_tokens:
                    break
        return draft_tokens

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass