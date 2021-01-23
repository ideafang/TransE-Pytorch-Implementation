import numpy as np
import torch
import torch.nn.functional as F

'''
Thanks LYuhang's repo : https://github.com/LYuhang/Trans-Implementation
'''

class TransE(torch.nn.Module):
    def __init__(self, n_entities, n_relations, embDim, margin=1.0, L=2):
        super(TransE, self).__init__()
        assert (L == 1 or L == 2)
        self.model = "TransE"
        self.margin = margin
        self.L = L

        self.e_emb = torch.nn.Embedding(num_embeddings=n_entities, embedding_dim=embDim)
        self.r_emb = torch.nn.Embedding(num_embeddings=n_relations, embedding_dim=embDim)
        self.pdis = torch.nn.PairwiseDistance(L)

    def normal_emb(self):
        emb_weight = self.e_emb.weight.detach().cpu().numpy()
        emb_weight = emb_weight / np.sqrt(np.sum(np.square(emb_weight), axis=1, keepdims=True))
        self.e_emb.weight.data.copy_(torch.from_numpy(emb_weight))

    def scoreOp(self, triples):
        h, r, t = torch.chunk(triples, 3, 1)
        head = torch.squeeze(self.e_emb(h), 1)  # (batch_size, 1, embDim)
        relation = torch.squeeze(self.r_emb(r), 1)  # (batch_size, 1, embDim)
        tail = torch.squeeze(self.e_emb(t), 1)  # (batch_size, 1, embDim)

        output = self.pdis(head+relation, tail)  # (batch_size, 1)
        return output

    def forward(self, pos_triples, neg_triples):
        n_data = pos_triples.size()[0]

        pos_score = self.scoreOp(pos_triples)
        neg_score = self.scoreOp(neg_triples)

        return torch.sum(F.relu(input=self.margin + pos_score - neg_score)) / n_data

    def get_emb_weights(self):
        return {'e_emb': self.e_emb.weight.detach().cpu().numpy(),
                'r_emb': self.r_emb.weight.detach().cpu().numpy()}

