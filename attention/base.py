import torch.nn as nn
import torch


class BaseTransformer(nn.Module):
    def __init__(self):
        super(BaseTransformer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.MASK_TOKEN = "[MASK]"
        self.SEP_TOKEN = "[SEP]"
        self.CLS_TOKEN = "[CLS]"
        self.PAD_TOKEN = "[PAD]"
        self.UNK_TOKEN = "[UNK]"
