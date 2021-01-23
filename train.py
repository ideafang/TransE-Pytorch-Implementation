from dataloader import get_dataset, TrainDataset, EvalDataset
from evaluation import Eval_MR
from torch.utils.data import DataLoader
from model import TransE
import torch

import pickle

GPU = True
EPOCHS_PER_SEED = 5
LR = 0.01
LR_DECAY_EPOCH = 5


def adjust_learning_rate(optim, decay):
    for param_group in optim.param_groups:
        param_group['lr'] *= decay


class Train:
    def __init__(self, data_name):
        self.dataset = get_dataset(data_name)
        self.n_entities = self.dataset.n_entities
        self.n_relations = self.dataset.n_relations

    def prepareData(self):
        print("Perpare dataloader")
        self.train = TrainDataset(self.dataset)
        self.trainloader = None
        self.valid = EvalDataset(self.dataset)
        self.validloader = DataLoader(self.valid, batch_size=self.valid.n_triples, shuffle=False)

    def prepareModel(self):
        print("Perpare model")
        self.model = TransE(self.n_entities, self.n_relations, embDim=100)
        if GPU:
            self.model.cuda()

    def saveModel(self):
        pickle.dump(self.model.get_emb_weights(), open('emb_weight.pkl', 'wb'))

    def fit(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=LR)
        minLoss = float("inf")
        bestMR = float("inf")
        GlobalEpoch = 0
        for seed in range(100):
            print(f"# Using seed: {seed}")
            self.train.regenerate_neg_samples(seed=seed)
            self.trainloader = DataLoader(self.train, batch_size=1024, shuffle=True, num_workers=4)
            for epoch in range(EPOCHS_PER_SEED):
                GlobalEpoch += 1
                for sample in self.trainloader:
                    if GPU:
                        pos_triples = torch.LongTensor(sample['pos_triples']).cuda()
                        neg_triples = torch.LongTensor(sample['neg_triples']).cuda()
                    else:
                        pos_triples = torch.LongTensor(sample['pos_triples'])
                        neg_triples = torch.LongTensor(sample['neg_triples'])

                    self.model.normal_emb()

                    loss = self.model(pos_triples, neg_triples)
                    if GPU:
                        lossVal = loss.cpu().item()
                    else:
                        lossVal = loss.item()

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    if minLoss > lossVal:
                        minLoss = lossVal
                MR = Eval_MR(self.validloader, "L2", **self.model.get_emb_weights())
                if MR < bestMR:
                    bestMR = MR
                    print('save embedding weight')
                    self.saveModel()
                print(f"Epoch: {epoch + 1}, Total_Train: {GlobalEpoch}, Loss: {lossVal}, minLoss: {minLoss},"
                      f"MR: {MR}, bestMR: {bestMR}")
                if GlobalEpoch % LR_DECAY_EPOCH == 0:
                    adjust_learning_rate(optim, 0.96)


if __name__ == '__main__':
    train = Train('FB15k-237')
    train.prepareData()
    train.prepareModel()
    train.fit()
