import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from attention.base import BaseTransformer


class BERTModel(BaseTransformer):
    """BERT model for masked token prediction"""

    def __init__(self, model_name_or_path='distilbert-base-uncased'):
        super(BERTModel, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def predict_masked(self, text: str, top_k=5):
        """Predict masked tokens given a text sequence, return top k predictions and scores"""
        tokenized_text = self.tokenizer.tokenize(text, return_tensors="pt")
        masked_index = tokenized_text.index(self.MASK_TOKEN)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]
        probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
        top_k_scores, top_k_indices = torch.topk(probs, top_k, sorted=True)
        top_k_tokens = self.tokenizer.convert_ids_to_tokens(top_k_indices.tolist())
        return top_k_tokens, top_k_scores.tolist()

    def mask_predict_ppl(self, text):
        # Tokenize the input text
        tokenized_text = self.tokenizer.encode_plus(text, return_tensors="pt")
        input_ids = tokenized_text["input_ids"].to(self.device)

        # Randomly select a token to mask
        masked_index = torch.randint(1, input_ids.shape[1] - 1, (1,)).item()
        input_ids[0, masked_index] = self.tokenizer.mask_token_id

        # Generate predictions
        with torch.no_grad():
            outputs = self.model(input_ids)
            predictions = outputs.logits[0, masked_index]
            predicted_token_id = torch.argmax(predictions).item()
            predicted_token = self.tokenizer.decode(predicted_token_id)

        # Calculate perplexity
        perplexity = 1 / torch.exp(predictions[predicted_token_id])

        return perplexity.item(), predicted_token

    def get_embedding_from_vocab(self, inputs):
        """Get embeddings from the MaskedLM"""
        if isinstance(inputs, str):
            ids = self.tokenizer(inputs, return_tensors="pt")["input_ids"][0][1:-1]
        else:
            ids = self.tokenizer.convert_tokens_to_ids(inputs)
        embeddings = self.model.get_input_embeddings()
        id_emb = embeddings(torch.tensor(ids).to(self.device))
        print("Embedding shape: ", embeddings.weight.shape, "Input shape: ", id_emb.shape)
        return id_emb

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)


if __name__ == '__main__':
    model = BERTModel()
    # predict masked tokens
    results = model.predict_masked("Paris is the [MASK] of France.")
    print("Predicted masked tokens: ", results)
    # test calculate perplexity
    # results = model.mask_predict_ppl("Paris is the capital of France.")
    # print(results)
    # test get embeddings
    # embeddings = model.get_embedding_from_vocab(["Salvasd"])
    # print("Embedding1: ", embeddings)
