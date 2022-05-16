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

identity = {
    'C':[1,0,0,0,0,0,0,0,0,0],
    'N':[0,1,0,0,0,0,0,0,0,0],
    'O':[0,0,1,0,0,0,0,0,0,0],
    'F':[0,0,0,1,0,0,0,0,0,0],
    'P':[0,0,0,0,1,0,0,0,0,0],
    'S':[0,0,0,0,0,1,0,0,0,0],
    'Cl':[0,0,0,0,0,0,1,0,0,0],
    'Br':[0,0,0,0,0,0,0,1,0,0],
    'I':[0,0,0,0,0,0,0,0,1,0],
    'other':[0,0,0,0,0,0,0,0,0,1],
}

zero_five = {
    0:[1,0,0,0,0,0],
    1:[0,1,0,0,0,0],
    2:[0,0,1,0,0,0],
    3:[0,0,0,1,0,0],
    4:[0,0,0,0,1,0],
    5:[0,0,0,0,0,1]
}

num_H = {
    0:[1,0,0,0,0],
    1:[0,1,0,0,0],
    2:[0,0,1,0,0],
    3:[0,0,0,1,0],
    4:[0,0,0,0,1]
}

def mol2graph(mol):
    # mol = Chem.MolFromSmiles(smiles)
    # mol = add_atom_index(mol)
    # graph
    g = nx.Graph()
    for atom in mol.GetAtoms():
        # atom number
        idx = atom.GetIdx()
        # print(idx)
        feature = []
        # identity one-hot 10
        feature.extend(identity.get(atom.GetSymbol(),[0,0,0,0,0,0,0,0,0,1]))
        # degree of atom one-hot 6
        # feature.extend(zero_five[atom.GetDegree()])
        # number of hydrogen atoms attached one-hot 5
        # feature.extend(num_H[atom.GetNumImplicitHs()])
        # implicit valence electrons one-hot 6
        # feature.extend(zero_five[atom.GetImplicitValence()])
        # aromatic 0 or 1
        # if atom.GetIsAromatic():
        #     feature.append(1)
        # else:
        #     feature.append(0)
        # total feature 28d
        g.add_node(idx, feature=feature)
    # add edge
    bonds_info = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    # add self_loop
    for atom in mol.GetAtoms():
        bonds_info.append((atom.GetIdx(), atom.GetIdx()))
    g.add_edges_from(bonds_info)
    # print(g.nodes.data)
    return g


def mol2y(mol):
    _y = []
    som = ['PRIMARY_SOM_1A2', 'PRIMARY_SOM_2A6','PRIMARY_SOM_2B6','PRIMARY_SOM_2C8','PRIMARY_SOM_2C9','PRIMARY_SOM_2C19','PRIMARY_SOM_2D6','PRIMARY_SOM_2E1','PRIMARY_SOM_3A4',
           'SECONDARY_SOM_1A2', 'SECONDARY_SOM_2A6','SECONDARY_SOM_2B6','SECONDARY_SOM_2C8','SECONDARY_SOM_2C9','SECONDARY_SOM_2C19','SECONDARY_SOM_2D6','SECONDARY_SOM_2E1','SECONDARY_SOM_3A4',
           'TERTIARY_SOM_1A2', 'TERTIARY_SOM_2A6','TERTIARY_SOM_2B6','TERTIARY_SOM_2C8','TERTIARY_SOM_2C9','TERTIARY_SOM_2C19','TERTIARY_SOM_2D6','TERTIARY_SOM_2E1','TERTIARY_SOM_3A4'
          ]
    result = []
    for k in som:
        try:
            _res = mol.GetProp(k)
            if ' ' in _res:
                res = _res.split(' ')
                for s in res:
                    result.append(int(s))
                # res = [int(temp) for temp in res]
            else:
                # res = [int(_res)]
                result.append(int(_res))
        except:
            pass

    for data in result:
        _y.append(data)
    _y = list(set(_y))

    y = np.zeros(len(mol.GetAtoms()))
    for i in _y:
        y[i-1] = 1
    return y

mols = Chem.SDMolSupplier('../../raw_database/merged.sdf')
dataset = []
for mol in mols:
    g = mol2graph(mol)
    y = mol2y(mol)
    graph = from_networkx(g)
    graph.feature = graph.feature.float()
    label = torch.tensor(y, dtype=torch.float)
    dataset.append((graph, label))

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
        self.conv1 = SAGEConv(10, 1024)
        self.conv2 = SAGEConv(1024, 1024)
        self.conv3 = SAGEConv(1024, 1024)
        self.linear1 = nn.Linear(1024, 1)
        # self.linear2 = nn.Linear(224, 1)
        self.relu = nn.ReLU()
        # self.ln1 = nn.LayerNorm(1024)
        # self.ln2 = nn.LayerNorm(112)
        # self.bn3 = nn.BatchNorm1d(112)
        # self.drop1 = nn.Dropout(p=0.3)

    
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
    # train_writer.add_scalar('Ave Loss', total_loss / len(training_set), epoch)
    # train_writer.add_scalar('ACC', accuracy_score(all_labels, all_pred), epoch)
    # train_writer.add_scalar('Top2', top2n / len(training_set), epoch)
    # train_writer.add_scalar('AUC', roc_auc_score(all_labels, all_pred_raw), epoch)
    # train_writer.add_scalar('MCC', mcc, epoch)
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
    splits = [0, 109, 218, 327, 436, 545]
    fold0 = trainingSet[:109]
    fold1 = trainingSet[109:218]
    fold2 = trainingSet[218:327]
    fold3 = trainingSet[327:436]
    fold4 = trainingSet[436:]
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
    pickle.dump(all_records, open('./all_records_base.pkl', 'wb'))

args = {
    'lr': 0.01,
    'epoch': 100,
    'seed': 42,
    'save_path': './basemodel/model',
    'momentum':0.9,
    'weight_decay': 1e-7,
    'pos_weight': 3
}

import time
starttime = time.time()
ntimecrossvalidation()
endtime = (starttime - time.time()) / 3600
with open('./time.txt', 'a') as f:
    f.write('base spend time ' + str(endtime) + 'h\n')

