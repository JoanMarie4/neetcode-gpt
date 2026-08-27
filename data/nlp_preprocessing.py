import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        all_sentences = positive + negative

        vocab = set()
        for s in all_sentences:
            for w in s.split():
                vocab.add(w)
        vocab = sorted(vocab)

        word_ids = {w: idx + 1 for idx, w in enumerate(vocab)}

        tensors = [torch.tensor([word_ids[w] for w in s.split()]) for s in all_sentences]

        return nn.utils.rnn.pad_sequence(tensors, batch_first=True)