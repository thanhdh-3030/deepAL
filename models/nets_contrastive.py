import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
from copy import deepcopy
from tqdm import tqdm
import torch.nn.init as init

class Net:
    def __init__(self, net, params, device,):
        self.net = net
        self.params = params
        self.device = device
        self.queue=torch.randn(self.params['num_class'],self.queue_len,self.proj_dim) # (C,M,D)
        self.queue=torch.nn.functional.normalize(self.queue,p=2,dim=2)
        self.queue_ptr=torch.zeros(self.params['num_class'],dtype=torch.long) # (C,)
    def train(self, data):
        n_epoch = self.params['n_epoch']

        dim = data.X.shape[1:]
        self.clf = self.net(dim = dim, pretrained = self.params['pretrained'], num_classes = self.params['num_class']).to(self.device)
        self.clf.train()
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
        elif self.params['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
        else:
            raise NotImplementedError

        loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])
        for epoch in tqdm(range(1, n_epoch+1), ncols=100):
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out, e1 = self.clf(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds
    def _dequeue_and_enqueue(self,keys,labels,
                             category,bs):
        if category not in labels:
            return
        keys=keys[list(labels).index(category)]
        ptr=int(self.queue_ptr[category])
        self.queue[category,:,ptr]=keys
        self.queue_ptr[category]=(ptr+bs)%self.queue_len
    def _compute_positive_contrastive_loss(self,keys,appeared_categories):
        """ Calculate contrastive loss enfoces the embeddings of same class
            to be close and different class far away.
        """
        contrast_loss=0
        for cls_ind in appeared_categories:
            query=keys[list(appeared_categories).index(cls_ind)] # (1,D)
            positive_keys= self.queue[cls_ind].clone().detach() # (M,D)
            all_ids=[i for i in range (2)] # all classes
            neg_ids=all_ids.copy().remove(cls_ind)
            negative_keys=self.queue[neg_ids] # 
        return 
    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs
    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs
    
    def get_model(self):
        return self.clf

 
 
  