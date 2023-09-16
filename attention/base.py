from abc import ABC

import torch.nn as nn
import torch


class BaseTransformer(nn.Module, ABC):
    def __init__(self):
        super(BaseTransformer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.MASK_TOKEN = "[MASK]"
        self.SEP_TOKEN = "[SEP]"
        self.CLS_TOKEN = "[CLS]"
        self.PAD_TOKEN = "[PAD]"
        self.UNK_TOKEN = "[UNK]"

    def predict_masked(self, text: str, top_k=5):
        """Predict masked tokens given a text sequence, return top k predictions and scores"""

    def get_embedding_from_vocab(self, inputs):
        """Get embeddings from the MaskedLM"""
