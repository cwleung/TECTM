import torch.nn as nn
from transformers import BertModel


class BertBase(nn.Module):
    """TO BE IMPLEMENTED """

    def __init__(self):
        super(BertBase, self).__init__()
        # select the pretrained model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[1]
        return last_hidden_state

    def get_embedding(self):
        embedding_matrix = self.embeddings.word_embeddings.weight
        return embedding_matrix
