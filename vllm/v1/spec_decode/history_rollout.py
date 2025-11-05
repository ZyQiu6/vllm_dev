from typing import Optional

import ray
import numpy as np

from vllm.config import VllmConfig
from vllm.v1.spec_decode.global_module.suffix_tree import get_history_trees

class HistoryRolloutProposer:
    def __init__(self, vllm_config: VllmConfig):
        # Minimum length of the HistoryRolloutTree to match.
        self.min_n = 2
        # Maximum length of the HistoryRolloutTree to match.
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        # self.k = vllm_config.speculative_config.num_speculative_tokens
        self.history_trees = get_history_trees()
        self.prompt_lookup = vllm_config.speculative_config.prompt_lookup_max

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
        
        draft_tokens = []
        if len(sampled_token_ids) >= self.min_n:
            prefix = sampled_token_ids[-self.min_n:]
            draft_tokens = ray.get(self.history_trees.predict(prompt_id, prefix, accept_length))
        return draft_tokens

    def propose_batch(
        self,
        accept_length_list: list[int],
        sampled_token_id_list: list[list[int]],
        prompt_token_id_list: list[list[int]]
    ) -> Optional[np.ndarray]:
        """Proposes the next sequence of tokens based on history rollout
        speculative decoding pattern.
        """
        batch_size = len(accept_length_list)
        prompt_id_list = [str(hash(tuple(prompt_token_id))) for prompt_token_id in prompt_token_id_list]
        prefix_length_list = [self.min_n for _ in range(batch_size)]
        batch_draft_tokens = self.history_trees.post_predict_batch(
            prompt_id_list,
            sampled_token_id_list,
            prefix_length_list,
            accept_length_list
        )
        return batch_draft_tokens

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass