
import torch
from torch import nn

if __name__ == '__main__':
    n, d, m = 3, 5, 7
    embedding = nn.Embedding(n, d, max_norm=True)
    W = torch.randn((m, d), requires_grad=True)
    idx = torch.tensor([1, 2])
    a = embedding.weight.clone() @ W.t()  # weight must be cloned for this to be differentiable
    b = embedding(idx) @ W.t()  # modifies weight in-place
    out = (a.unsqueeze(0) + b.unsqueeze(1))
    loss = out.sigmoid().prod()
    loss.backward()


embedding = nn.Embedding(10, 3)
embedding = nn.Embedding(10, 3)
# a batch of 2 samples of 4 indices each
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
embedding(input)