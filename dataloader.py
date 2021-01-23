import os
import numpy as np
from torch.utils.data import Dataset


class KGDataset:
    def __init__(self, entity_path, relation_path,
                 train_path, valid_path=None, test_path=None, format=None,
                 delimiter='\t', skip_first_line=False):
        if format is None:
            format = [0, 1, 2]
        self.delimiter = delimiter
        self.entity2id, self.n_entities = self.read_entity(entity_path)
        self.relation2id, self.n_relations = self.read_relation(relation_path)
        self.train = self.read_triple(train_path, "train", skip_first_line, format)
        if valid_path is not None:
            self.valid = self.read_triple(valid_path, "valid", skip_first_line, format)
        else:
            self.valid = None
        if test_path is not None:
            self.test = self.read_triple(test_path, "test", skip_first_line, format)
        else:
            self.test = None

    def read_entity(self, entity_path):
        with open(entity_path) as f:
            entity2id = {}
            for line in f:
                eid, entity = line.strip().split(self.delimiter)
                entity2id[entity] = int(eid)
        return entity2id, len(entity2id)

    def read_relation(self, relation_path):
        with open(relation_path) as f:
            relation2id = {}
            for line in f:
                rid, relation = line.strip().split(self.delimiter)
                relation2id[relation] = int(rid)
        return relation2id, len(relation2id)

    def read_triple(self, path, mode, skip_first_line=False, format=None):
        if format is None:
            format = [0, 1, 2]
        if path is None:
            return None

        print(f"Reading {mode} triples...")
        heads = []
        tails = []
        rels = []
        with open(path) as f:
            if skip_first_line:
                _ = f.readline()
            for line in f:
                triple = line.strip().split(self.delimiter)
                h, r, t = triple[format[0]], triple[format[1]], triple[format[2]]
                heads.append(self.entity2id[h])
                rels.append(self.relation2id[r])
                tails.append(self.entity2id[t])

        heads = np.array(heads, dtype=np.int64)
        tails = np.array(tails, dtype=np.int64)
        rels = np.array(rels, dtype=np.int64)
        print(f"Finished. Read {len(heads)} {mode} triples.")

        return (heads, rels, tails)


class KGFB15k237(KGDataset):
    def __init__(self, path, name='FB15k-237'):
        self.name = name
        if not os.path.exists(os.path.join(path, name)):
            assert False, f"File not found in {path}"
        self.path = os.path.join(path, name)
        super(KGFB15k237, self).__init__(os.path.join(self.path, 'entities.dict'),
                                        os.path.join(self.path, 'relations.dict'),
                                        os.path.join(self.path, 'train.txt'),
                                        os.path.join(self.path, 'valid.txt'),
                                        os.path.join(self.path, 'test.txt'))


def get_dataset(data_name):
    if data_name == 'FB15k-237':
        dataset = KGFB15k237('dataset')
    else:
        assert False, f"Unknown dataset {data_name}"
    return dataset


class TrainDataset(Dataset):
    def __init__(self, dataset):
        self.pos_samples = np.array(dataset.train).T
        self.n_entities = dataset.n_entities
        self.n_pos_samples = self.pos_samples.shape[0]
        self.neg_samples, self.n_neg_samples = self.generate_neg_samples()
        self.n_data = self.n_pos_samples + self.n_neg_samples
        # self.data, self.label, self.n_data = self.format_data(pos_samples, neg_samples)
        print(f"|Train|: {self.n_data}")

    def __len__(self):
        return self.n_pos_samples

    def __getitem__(self, item):
        sample = {'pos_triples': self.pos_samples[item], 'neg_triples': self.neg_samples[item]}
        return sample

    def generate_neg_samples(self, neg_rate=1, seed=601):
        """
        更换头节点或尾节点生成负样本
        :param neg_rate: 负样本与正样本的比值，默认为1
        :return: 负样本, 负样本数量
        """
        n_neg_samples = self.n_pos_samples * neg_rate
        neg_samples = np.tile(self.pos_samples, (neg_rate, 1))
        np.random.seed(seed)
        values = np.random.randint(self.n_entities, size=n_neg_samples)
        for i in range(int(n_neg_samples/2)):
            neg_samples[i][0] = values[i]
        for i in range(int(n_neg_samples/2), n_neg_samples):
            neg_samples[i][2] = values[i]
        return neg_samples, n_neg_samples

    def regenerate_neg_samples(self, seed):
        neg_samples = np.tile(self.pos_samples, (1, 1))
        np.random.seed(seed)
        values = np.random.randint(self.n_entities, size=self.n_neg_samples)
        for i in range(int(self.n_neg_samples / 2)):
            neg_samples[i][0] = values[i]
        for i in range(int(self.n_neg_samples / 2), self.n_neg_samples):
            neg_samples[i][2] = values[i]
        self.neg_samples = neg_samples

    # def format_data(self, pos_samples, neg_samples):
    #     data = np.concatenate((pos_samples, neg_samples))
    #     n_data = self.n_pos_samples + self.n_neg_samples
    #     label = np.zeros(shape=(n_data, 1), dtype=np.float)
    #     label[: self.n_pos_samples][0] = 1.0
    #     idx = np.random.permutation(n_data)
    #     data[:] = data[idx]
    #     label[:] = label[idx]
    #     return data, label, n_data


class EvalDataset(Dataset):
    def __init__(self, dataset):
        self.triples = np.array(dataset.valid).T
        self.n_triples = self.triples.shape[0]
        print('|Valid|:', self.n_triples)

    def __len__(self):
        return self.n_triples

    def __getitem__(self, item):
        return self.triples[item]