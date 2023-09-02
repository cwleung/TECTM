from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from attention.base import BaseTransformer


class BERTModel(BaseTransformer):
    """BERT model for masked token prediction"""

    def __init__(self):
        super(BERTModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
        self.model = AutoModelForMaskedLM.from_pretrained('bert-large-uncased').to(self.device)

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
        """Get embeddings from the vocabulary except the special tokens"""
        if isinstance(inputs, str):
            inputs = self.tokenizer(inputs, return_tensors="pt")["input_ids"][0][1:-1]
        else:
            ids = self.tokenizer.convert_tokens_to_ids(inputs)
        embeddings = self.model.get_input_embeddings()
        return embeddings(torch.tensor(ids).to(self.device))

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

    def generate_next_token(self, text: str, top_k=5):
        """Generate next token given a text sequence, return top k predictions and scores"""
        tokenized_text = self.tokenizer.tokenize(text, return_tensors="pt")
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]
        probs = torch.nn.functional.softmax(predictions[0, -1], dim=-1)
        top_k_scores, top_k_indices = torch.topk(probs, top_k, sorted=True)
        top_k_tokens = self.tokenizer.convert_ids_to_tokens(top_k_indices.tolist())
        return top_k_tokens, top_k_scores.tolist()


    def visualize_embeddings_tsne(self, inputs):
        """Visualize embeddings using t-SNE"""
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        embeddings = self.get_embedding_from_vocab(inputs)
        embeddings = embeddings.detach().cpu().numpy()
        tsne = TSNE(n_components=2, random_state=0)
        embeddings_2d = tsne.fit_transform(embeddings)
        plt.figure(figsize=(6, 6))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        for i, word in enumerate(inputs):
            plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]))
        plt.show()


if __name__ == '__main__':
    model = BERTModel()
    # test calculate perplexity
    results = model.mask_predict_ppl("Paris is the capital of France.")
    print(results)
    # test get embeddings
    embeddings = model.get_embedding_from_vocab(["Paris", "France"])
    print(embeddings.shape)
    # test visualize embeddings
    model.visualize_embeddings_tsne("""
    When Meredith Tabbone read an article about an Italian village selling homes for the price of a slice of pizza, she jumped at the idea. The Chicago native’s great-grandfather was from Sambuca, a charming village in the southern region of Sicily, which in 2019 was auctioning abandoned homes starting at 1 euro. She placed an impromptu bid on a grainy, black-and-white image of a dilapidated home for 5,555 euros (about $6,355 at the time). I did no research on any of this,” the 44-year-old financial analyst said. “I assumed a lot of people would put in bids since it was in major news outlets. I was shocked when I won the bid.” The only condition: Tabbone had to renovate the property within three years.""")
