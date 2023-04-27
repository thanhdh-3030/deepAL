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
from models.nt_xent_loss import NTXentLoss
class ContrastNet:
    def __init__(self, net, params, device,):
        self.net = net
        self.params = params
        self.device = device
        self.queue=torch.randn(self.params['embedding_dim'],self.params['memory_size'],self.params['embedding_dim']) # (K,D)
        self.queue=F.normalize(self.queue,p=2,dim=0)
        self.queue_ptr=torch.zeros(1, dtype=torch.long) # (1,)
    def train(self, data):
        n_epoch = self.params['n_epoch']
        dim = data.X.shape[1:]
        self.clf = self.net.to(self.device)
        self.clf.train()
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
        elif self.params['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
        else:
            raise NotImplementedError

        loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])
        for epoch in tqdm(range(1, int(n_epoch/2)+1), ncols=100):
            for batch_idx, (x1,x2, y, idxs) in enumerate(loader):
                x1,x2, y = x1.to(self.device),x2.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out, e1 = self.clf(x1)
                _, e2 = self.clf(x2)
                # normalize embedding 
                e1=F.normalize(e1,dim=1)
                e2=F.normalize(e2,dim=1)
                e2=e2.detach()
                contrast_loss=self._compute_unlabel_contrastive_loss(e1,e2)
                # contrast_criterion=NTXentLoss(device=self.device,batch_size=x1.shape[0],temperature=0.1,use_cosine_similarity=False)
                # contrast_loss=contrast_criterion(e1,e2)
                ce_loss = F.cross_entropy(out, y)
                total_loss=self.params['contrast_weight']*contrast_loss+ce_loss
                total_loss.backward()
                optimizer.step()
        for epoch in tqdm(range(1, int(n_epoch/2)+1), ncols=100):
            for batch_idx, (x,_, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out, e1 = self.clf(x)
                ce_loss = F.cross_entropy(out, y)
                ce_loss.backward()
                optimizer.step()
    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x,x1, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr
    # def _compute_positive_contrastive_loss(self,keys,appeared_categories):
    #     """ Calculate contrastive loss enfoces the embeddings of same class
    #         to be close and different class far away.
    #     """
    #     contrast_loss=0
    #     for cls_ind in appeared_categories:
    #         query=keys[list(appeared_categories).index(cls_ind)] # (1,D)
    #         positive_keys= self.queue[cls_ind].clone().detach() # (M,D)
    #         all_ids=[i for i in range (2)] # all classes
    #         neg_ids=all_ids.copy().remove(cls_ind)
    #         negative_keys=self.queue[neg_ids] # 
    #     return 
    def _compute_unlabel_contrastive_loss(self,query,positive_key):
        """ Calculates the contrastive loss for self-supervised learning.
        Args:
            query (torch.Tensor): Tensor with query samples (e.g. embeddings of the input).
                (N,D) where D is embedding dim.
            positive_keys (torch.Tensor): Tensor with positive samples (e.g. embeddings of augmented input).
                (N,D).
        Returns:
            torch.Tensor: value of contrastive loss.
        """
        # normalize query features and positive key
        query=F.normalize(query,dim=1)
        positive_key=F.normalize(positive_key,dim=1)
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [query, positive_key]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [query, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.params['temperature']
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        contrast_loss=F.cross_entropy(logits,labels)
        return contrast_loss
    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x,x1, y, idxs in loader:
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
    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x,x1, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs
    def get_model(self):
        return self.clf

 
 
  