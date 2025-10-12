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
        if not ray.get(history_trees.exist.remote(prompt_id)):
            return []
        draft_tokens = []
        prefix_len_candidates = range(self.max_n, self.min_n, -1)
        for prefix_len in prefix_len_candidates:
            if len(sampled_token_ids) >= prefix_len:
                prefix = sampled_token_ids[-prefix_len:]
                draft_tokens = ray.get(history_trees.predict.remote(prompt_id, prefix, accept_length))
                if len(draft_tokens) > 0:
                    break
        return draft_tokens

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass