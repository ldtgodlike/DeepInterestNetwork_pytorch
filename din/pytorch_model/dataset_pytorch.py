import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np

class DataInput:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        t = self.data[index]
        return {"uid": t[0], "iid": t[2], "y": t[3], "hist": t[1], "sl": len(t[1])}

def collate_train1(batch):
    uids = torch.cat([torch.tensor(d["uid"]) for d in batch], dim = 0)
    iids = torch.cat([torch.tensor(d["iid"]) for d in batch], dim = 0)
    ys = torch.cat([torch.tensor(d["y"]) for d in batch], dim = 0)
    max_sl = max([len(d["hist"]) for d in batch])
    hists = np.zeros([len(batch), max_sl], np.int64)
    for i, d in enumerate(batch):
        hists[i][:len(d["hist"])] = d["hist"]
    hists = torch.from_numpy(hists)
    sl = torch.cat([torch.tensor(d["sl"]) for d in batch], dim = 0)
    return uids, iids, ys, hists, sl

def collate_train(batch):
    uids = torch.tensor([torch.tensor(d["uid"]) for d in batch])
    iids = torch.tensor([torch.tensor(d["iid"]) for d in batch])
    ys = torch.tensor([torch.tensor(d["y"]) for d in batch])
    max_sl = max([len(d["hist"]) for d in batch])
    hists = np.zeros([len(batch), max_sl], np.int64)
    for i, d in enumerate(batch):
        hists[i][:len(d["hist"])] = d["hist"]
    hists = torch.from_numpy(hists)
    sl = torch.tensor([torch.tensor(d["sl"]) for d in batch])
    return uids, iids, ys, hists, sl


class DataInputTest:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        t = self.data[index]
        return {"uid": t[0], "iid": t[2], "hist": t[1], "sl": len(t[1])}

def collate_test(batch):
    uids = torch.tensor([torch.tensor(d["uid"]) for d in batch])
    iids = torch.tensor([torch.tensor(d["iid"][0]) for d in batch])
    jids = torch.tensor([torch.tensor(d["iid"][1]) for d in batch])
    max_sl = max([len(d["hist"]) for d in batch])
    hists = np.zeros([len(batch), max_sl], np.int64)
    for i, d in enumerate(batch):
        hists[i][:len(d["hist"])] = d["hist"]
    hists = torch.from_numpy(hists)
    sl = torch.tensor([torch.tensor(d["sl"]) for d in batch])
    return uids, iids, jids, hists, sl


if __name__ == "__main__":

    with open('../dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f)

    dataset_train = DataInput(train_set)
    dataset_test = DataInputTest(test_set)

    dataLoader_train = DataLoader(dataset_train, collate_fn=collate_train, shuffle=False, batch_size=1)
    dataLoader_test = DataLoader(dataset_test, collate_fn=collate_test, shuffle=False, batch_size=1)

    i = 0
    for item in dataLoader_train:
        if i > 5:
            break
        print(item)
        i += 1

    print("="*100)

    i = 0
    for item in dataLoader_test:
        if i > 5:
            break
        print(item)
        i += 1