from typing import Optional

import numpy as np

from vllm.config import VllmConfig

class HistoryRolloutProposer:
    def __init__(self, vllm_config: VllmConfig):
        # Minimum length of the HistoryRolloutTree to match.
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        # Maximum length of the HistoryRolloutTree to match.
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        # self.k = vllm_config.speculative_config.num_speculative_tokens
        self.debug_flag = True

    def propose(
        self,
        accept_length: int,
        sampled_token_ids: list[int],
        prompt_token_ids: list[int],
        history_trees: dict,
    ) -> Optional[np.ndarray]:
        """Proposes the next sequence of tokens based on n-gram pattern 
        matching in the context. The function finds matches of the last n 
        tokens in the previous context, and returns k tokens that followed 
        that match.
        
        Args:
            context_token_ids: Numpy array of token IDs representing the 
                               context sequence.

        Returns:
            np.ndarray: The sequence of tokens that followed 
                        the matched n-gram in the context.
            None: If no matching n-gram pattern is found.

        Example:
            If context_token_ids = [1,2,3,4,2,3], min_n = 2, max_n = 3, and
            k = 4:
            - The last 3 (= max_n) tokens [4,2,3] cannot find a match.
            - The last 2 tokens [2,3] will be matched against the previous 
              4 tokens [1,2,3,4].
            - Finding a match of [2,3] would return the tokens that 
              followed that pattern. Here we will return [4,2,3] because 
              we only have three tokens after the match.
        """
        batch_drafts = []
        if self.debug_flag:
            print(f"prompt_token_ids: {prompt_token_ids}")
            self.debug_flag = False
        prompt_id = str(hash(tuple(prompt_token_ids)))
        if prompt_id not in history_trees:
            return []
        else:
            print(f"{prompt_id} found in history_trees")
        history_tree = history_trees[prompt_id]
        draft_tokens = None
        prefix_len_candidates = range(self.max_n, self.min_n, -1)
        for prefix_len in prefix_len_candidates:
            if len(sampled_token_ids) >= prefix_len:
                prefix = sampled_token_ids[-prefix_len:]
                draft_tokens = history_tree.predict(history_tree, prefix, accept_length)
                if draft_tokens:
                    break
        batch_drafts.append(draft_tokens)
        return batch_drafts

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass