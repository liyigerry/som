# sklearn, condusion matrix, mcc and auc
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
# build dataset
from rdkit import Chem
import networkx as nx
import pickle
import numpy as np
from torch_geometric.utils import from_networkx
# torch
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
# tensorboard
from torch.utils.tensorboard import SummaryWriter
# random
import random

train_writer = SummaryWriter("./runs/sage/train")
val_writer = SummaryWriter("./runs/sage/val")

raw = pickle.load(open('../../raw_database/graph_datasetmm.pkl', 'rb'))
dataset = []
for mol, label in raw:
    mol = from_networkx(mol)
    mol.feature = mol.feature.float()
    label = torch.tensor(label).float()
    dataset.append((mol,label))

random.seed('42')
random.shuffle(dataset)

total = len(dataset)
ratio = 0.8
training_set = dataset[:int(total * 0.8)]
test_set = dataset[int(total * 0.8):]

# evaluation
def top2(output, label):
    preds = torch.sigmoid(output)
    _, indices = torch.topk(preds, 2)
    pos_index = []
    for i in range(label.shape[0]):
        if label[i] == 1:
            pos_index.append(i)
    # print(pos_index)      
    for li in pos_index:
        if li in indices:
            return True
    return False
    
def MCC(output, label):
    tn,fp,fn,tp=confusion_matrix(label, output).ravel()
    # print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    up = (tp * tn) - (fp * fn)
    down = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return up / down

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(47, 1024)
        self.conv2 = SAGEConv(1024, 1024)
        self.conv3 = SAGEConv(1024, 1024)
        self.linear1 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()


    
    def forward(self, mol):
        res = self.conv1(mol.feature, mol.edge_index)
        res = self.relu(res)
        res = self.conv2(res, mol.edge_index)
        res = self.relu(res)
        res = self.conv3(res, mol.edge_index)
        res = self.relu(res)
        res = self.linear1(res)
        # res = self.relu(res)
        # # res = self.drop1(res)
        # res = self.linear2(res)
        return res

def train(args, model, device, training_set, optimizer, criterion, epoch, record):
    model.train()
    total_loss = 0
    all_pred = []
    all_pred_raw = []
    all_labels = []
    top2n = 0
    for mol, target in training_set:
        mol, target = mol.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(mol)
        # squeeze
        output = torch.squeeze(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # tracking
        top2n += top2(output, target)
        total_loss += loss.item()
        all_pred.append(np.rint(torch.sigmoid(output).cpu().detach().numpy()))
        all_pred_raw.append(torch.sigmoid(output).cpu().detach().numpy())
        all_labels.append(target.cpu().detach().numpy())
    all_pred = np.concatenate(all_pred).ravel()
    all_pred_raw = np.concatenate(all_pred_raw).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    mcc = MCC(all_pred, all_labels)
    record['train_loss'].append(total_loss / len(training_set))
    record['train_acc'].append(accuracy_score(all_labels, all_pred))
    record['train_top2'].append(top2n / len(training_set))
    record['train_auc'].append(roc_auc_score(all_labels, all_pred_raw))
    record['train_mcc'].append(mcc)
    # loss_record['train'].append(total_loss / len(training_set))
    # print(f'Train Epoch: {epoch}, Ave Loss: {total_loss / len(training_set)} ACC: {accuracy_score(all_labels, all_pred)} Top2: {top2n / len(training_set)} AUC: {roc_auc_score(all_labels, all_pred_raw)} MCC: {mcc}')


def val(args, model, device, val_set, optimizer, criterion, epoch, record):
    model.eval()
    total_loss = 0
    all_pred = []
    all_pred_raw = []
    all_labels = []
    top2n = 0
    for mol, target in val_set:
        mol, target = mol.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(mol)
        # squeeze
        output = torch.squeeze(output)
        loss = criterion(output, target)
        # tracking
        top2n += top2(output, target)
        total_loss += loss.item()
        all_pred.append(np.rint(torch.sigmoid(output).cpu().detach().numpy()))
        all_pred_raw.append(torch.sigmoid(output).cpu().detach().numpy())
        all_labels.append(target.cpu().detach().numpy())
    all_pred = np.concatenate(all_pred).ravel()
    all_pred_raw = np.concatenate(all_pred_raw).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    mcc = MCC(all_pred, all_labels)
    # val_writer.add_scalar('Ave Loss', total_loss / len(val_set), epoch)
    # val_writer.add_scalar('ACC', accuracy_score(all_labels, all_pred), epoch)
    # val_writer.add_scalar('Top2', top2n / len(val_set), epoch)
    # val_writer.add_scalar('AUC', roc_auc_score(all_labels, all_pred_raw), epoch)
    # val_writer.add_scalar('MCC', mcc, epoch)
    record['val_loss'].append(total_loss / len(val_set))
    record['val_acc'].append(accuracy_score(all_labels, all_pred))
    record['val_top2'].append(top2n / len(val_set))
    record['val_auc'].append(roc_auc_score(all_labels, all_pred_raw))
    record['val_mcc'].append(mcc)
    # loss_record['dev'].append(total_loss / len(val_set))
    # print(f'Val Epoch: {epoch}, Ave Loss: {total_loss / len(val_set)} ACC: {accuracy_score(all_labels, all_pred)} Top2: {top2n / len(val_set)} AUC: {roc_auc_score(all_labels, all_pred_raw)} MCC: {mcc}')
    return top2n / len(val_set)


def test(model, device, test_set, record):
    model.eval()
    all_pred = []
    all_pred_raw = []
    all_labels = []
    top2n = 0
    with torch.no_grad():
        for mol, target in test_set:
            mol, target = mol.to(device), target.to(device)
            output = model(mol)
            # squeeze
            output = torch.squeeze(output)
            # tracking
            top2n += top2(output, target)
            all_pred.append(np.rint(torch.sigmoid(output).cpu().detach().numpy()))
            all_pred_raw.append(torch.sigmoid(output).cpu().detach().numpy())
            all_labels.append(target.cpu().detach().numpy())
    all_pred = np.concatenate(all_pred).ravel()
    all_pred_raw = np.concatenate(all_pred_raw).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    mcc = MCC(all_pred, all_labels)
    record['test_acc'].append(accuracy_score(all_labels, all_pred))
    record['test_top2'].append(top2n / len(test_set))
    record['test_auc'].append(roc_auc_score(all_labels, all_pred_raw))
    record['test_mcc'].append(mcc)
    print(f'ACC: {accuracy_score(all_labels, all_pred)} Top2: {top2n / len(test_set)} AUC: {roc_auc_score(all_labels, all_pred_raw)} MCC: {mcc}') 


def main(args, training_set, validation_set, seed, i):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args['seed'])
    model = Model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])
    criterion = nn.BCEWithLogitsLoss(torch.tensor(args['pos_weight']).to(device))
    max_top2 = 0
    record = {
    'train_loss':[],
    'val_loss': [],
    'train_acc': [],
    'val_acc': [],
    'train_top2':[],
    'val_top2':[],
    'train_auc':[],
    'val_auc':[],
    'train_mcc':[],
    'val_mcc':[],
    'test_acc':[],
    'test_top2':[],
    'test_auc':[],
    'test_mcc':[]
}
    for epoch in range(1, args['epoch'] + 1):
        train(args, model, device, training_set, optimizer, criterion, epoch, record)
        top2acc = val(args, model, device, validation_set, optimizer, criterion, epoch, record)
        random.shuffle(training_set)
        if top2acc > max_top2:
            max_top2 = top2acc
            print('Saving model (epoch = {:4d}, top2acc = {:.4f})'
                .format(epoch, max_top2))
            torch.save(model.state_dict(), args['save_path'] + '_' + str(seed) + '_' + str(i))
    model = Model().to(device)
    model.load_state_dict(torch.load(args['save_path'] + '_' + str(seed) + '_' + str(i)))
    test(model, device, test_set, record)
    return record


def crossvalidation(args, trainingSet, k, seed):
    # split training set to k fold
    random.seed(seed)
    random.shuffle(trainingSet)
    fold0 = trainingSet[:108]
    fold1 = trainingSet[108:216]
    fold2 = trainingSet[216:324]
    fold3 = trainingSet[324:432]
    fold4 = trainingSet[432:]
    folds = [fold0, fold1, fold2, fold3, fold4]
    records = []
    for i in range(5):
        val_set = folds[i]
        _tr_set = folds[0:i] + folds[i+1:]
        tr_set = [y for x in _tr_set for y in x]
        random.shuffle(tr_set)
        record = main(args, tr_set, val_set, seed, i)
        records.append(record)
    return records

def ntimecrossvalidation(n=5):
    seeds = [10,20,30,40,50]
    all_records = []
    for seed in seeds:
        records = crossvalidation(args, training_set, 5, seed)
        all_records.append(records)
    pickle.dump(all_records, open('./all_records_cdk.pkl', 'wb'))

args = {
    'lr': 0.01,
    'epoch': 100,
    'seed': 42,
    'save_path': './cdkmodel/model',
    'momentum':0.9,
    'weight_decay': 1e-7,
    'pos_weight': 3
}

import time
starttime = time.time()
ntimecrossvalidation()
endtime = (starttime - time.time())
with open('./time.txt', 'a') as f:
    f.write('cdk spend time ' + str(endtime) + 'h\n')

