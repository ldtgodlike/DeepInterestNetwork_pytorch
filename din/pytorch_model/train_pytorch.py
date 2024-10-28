from dataset_pytorch import DataInput, DataInputTest, collate_train, collate_test, DataLoader
from model_pytorch import DIN
import pickle
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

batch_size = 4
predict_batch_size = 4
predict_users_num = 32
predict_ads_num = 100
train_epochs = 50
accumulation_steps = 8

with open('../dataset.pkl', 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count = pickle.load(f)

def load_dataset():

    dataset_train = DataInput(train_set)
    dataset_test = DataInputTest(test_set[:20000])

    print(len(dataset_train), len(dataset_test))

    dataLoader_train = DataLoader(dataset_train, collate_fn=collate_train, shuffle=False, batch_size=batch_size, num_workers= 0)
    dataLoader_test = DataLoader(dataset_test, collate_fn=collate_test, shuffle=False, batch_size=predict_batch_size)

    return dataLoader_train, dataLoader_test


def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """
    # sort by pred value, from small to big
    with torch.no_grad():
        arr = sorted(raw_arr, key=lambda d:d[2])

        auc = 0.0
        fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
        for record in arr:
            fp2 += record[0] # noclick
            tp2 += record[1] # click
            auc += (fp2 - fp1) * (tp2 + tp1)
            fp1, tp1 = fp2, tp2

        # if all nonclick or click, disgard
        threshold = len(arr) - 1e-3
        if tp2 > threshold or fp2 > threshold:
            return -0.5

        if tp2 * fp2 > 0.0:  # normal auc
            return (1.0 - auc / (2.0 * tp2 * fp2))
        else:
            return None


def _auc_arr(score):
    with torch.no_grad():
        score_p = score[:,0]
        score_n = score[:,1]
        #print "============== p ============="
        #print score_p
        #print "============== n ============="
        #print score_n
        score_arr = []
        for s in score_p.tolist():
            score_arr.append([0, 1, s])
        for s in score_n.tolist():
            score_arr.append([1, 0, s])
        return score_arr

def test_step(model,dataLoader_test, device):
    global cate_list
    model.eval()
    auc_sum = 0.0
    score_arr = []
    test_nums = 0
    for uids, iids, jids, hists, sl in tqdm(dataLoader_test):
        iids = iids.to(device)
        jids = jids.to(device)
        hists = hists.to(device)
        sl = sl.to(device)
        test_nums += 1
        with torch.no_grad():
            logits, logits_sub, x, i_b, j_b, din_i, din_j = model(iids, jids, hists, sl, cate_list)
            auc_, score_ = model.eval_step(x, i_b, j_b, din_i, din_j)
            score_arr += _auc_arr(score_)
            auc_sum += auc_ #* len(uids)
    test_gauc = auc_sum / test_nums
    Auc = calc_auc(score_arr)
    model.train()

    print('Eval_GAUC: % .4f, Eval_AUC: % .4f' % (test_gauc, Auc))


best_auc = 0.0

def main():
    global cate_list
    device = "cuda:0"
    model = DIN(user_count, item_count, cate_count, predict_ads_num, hidden_units=128).to(device)
    crition = nn.BCELoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr = 1)

    cate_list = torch.Tensor(cate_list).long().to(device)
    dataLoader_train, dataLoader_test = load_dataset()

    #print(model)
    for epoch in range(train_epochs):
        model.train()
        train_loop = tqdm(dataLoader_train, desc="train")
        i = 0
        loss_batch = 0.0
        for uids, iids, jids, hists, sl in train_loop:
            i += 1
            iids = iids.to(device)
            jids = jids.to(device)
            hists = hists.to(device)
            sl = sl.to(device)
            logits, logits_sub, x, i_b, j_b, _, _ = model(iids, jids, hists, sl, cate_list)
            loss = F.binary_cross_entropy(logits, jids.view(-1).float(), reduction='mean')
            loss_batch += loss
            loss = loss / accumulation_steps
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)

            if i % accumulation_steps == 0:
                train_loop.set_description(f'Epoch [{epoch + 1}] / {train_epochs}')
                train_loop.set_postfix({'loss': '{0:1.2f}'.format(loss_batch.item() / accumulation_steps)})
                loss_batch = 0.0
                optimizer.step()
                optimizer.zero_grad()
            if i % 5000 == 0:
                test_step(model,dataLoader_test,device)



if __name__ == "__main__":
    main()


