import numpy as np
from torch.utils.data import dataloader


def calSimilarity(exptail, e_emb, simMeasure):
    if simMeasure == "L2":
        simScore = []
        for exp_e in exptail:
            score = np.linalg.norm(exp_e[np.newaxis, :] - e_emb, ord=2, axis=1, keepdims=False)
            simScore.append(score)
        return np.array(simScore)
    elif simMeasure == "L1":
        simScore = []
        for exp_e in exptail:
            score = np.linalg.norm(exp_e[np.newaxis, :] - e_emb, ord=1, axis=1, keepdims=False)
            simScore.append(score)
        return np.array(simScore)
    else:
        assert False, f"Simlarity method {simMeasure} is not support!"


def calRank(simScore, tail):
    realScore = simScore[np.arange(tail.shape[0]), tail].reshape((-1, 1))
    judMatrix = simScore - realScore
    judMatrix[judMatrix > 0] = 0
    judMatrix[judMatrix < 0] = 1
    judMatrix = np.sum(judMatrix, axis=1)
    return judMatrix


def Eval_MR(evalloader: dataloader, simMeasure, **kwargs):
    R = 0
    N = 0
    for triples in evalloader:
        triples = triples.numpy()
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        h = np.take(kwargs['e_emb'], indices=h, axis=0)
        r = np.take(kwargs['r_emb'], indices=r, axis=0)
        simScore = calSimilarity(h+r, kwargs['e_emb'], simMeasure)
        ranks = calRank(simScore, t)
        R += np.sum(ranks)
        N += ranks.shape[0]
    return R / N

