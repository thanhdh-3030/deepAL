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
from models.resnet import ResNet18

class OnebitNet:
    def __init__(self, params, device,handler):
        # self.net = net
        self.params = params
        self.device = device
        self.queue=torch.randn(self.params['embedding_dim'],self.params['memory_size']).cuda() # (K,D)
        self.queue=F.normalize(self.queue,dim=0)
        self.queue_ptr=torch.zeros(1, dtype=torch.long) # (1,)
        self.clf_query=ResNet18().to(self.device)
        self.clf_key=ResNet18().to(self.device)
        self.handler=handler
        # Freeze clf key
        for param_q, param_k in zip(self.clf_query.parameters(), self.clf_key.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    def train(self,X_labeled, Y_labeled,X_unlabeled, Y_unlabeled):
        n_epoch = self.params['n_epoch']
        # dim = labeled_data.X.shape[1:]
        # self.clf = self.net.to(self.device)
        # self.clf.train()
        self.clf_query.train()
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.clf_query.parameters(), **self.params['optimizer_args'])
        elif self.params['optimizer'] == 'SGD':
            # optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
                optimizer = optim.SGD(self.clf_query.parameters(), **self.params['optimizer_args'])
        else:
            raise NotImplementedError

        loader_tr = DataLoader(self.handler(X_labeled, Y_labeled,X_unlabeled, Y_unlabeled,
											transform = self.params['transform_train']), shuffle= True, **self.params['loader_tr_args'])
        for epoch in tqdm(range(1, int(n_epoch)+1), ncols=100):
        # for epoch in tqdm(range(1, int(n_epoch/2)+1), ncols=100):
            for batch_idx, (idxs,x_labeled, y_labeled,x_unlabeled1,x_unlabeled2,_ ) in enumerate(loader_tr):
                x_labeled, y_labeled,x_unlabeled1,x_unlabeled2 = x_labeled.to(self.device),y_labeled.to(self.device), x_unlabeled1.to(self.device),x_unlabeled2.to(self.device)
                optimizer.zero_grad()
                out, _ = self.clf_query(x_labeled)                
                ce_loss = F.cross_entropy(out, y_labeled) # compute supervise loss

                _, query = self.clf_query(x_unlabeled1)
                query=F.normalize(query,dim=1)  # normalize embedding
                with torch.no_grad():  # no gradient to keys
                    self._momentum_update_key_encoder()  # update the key encoder
                    _, key = self.clf_key(x_unlabeled2)
                    key=F.normalize(key,dim=1)
                contrast_loss=self._compute_unlabel_contrastive_loss(query,key) # compute contrastive loss

                total_loss=ce_loss + self.params['contrast_weight']*contrast_loss
                # ce_loss.backward()
                total_loss.backward()
                optimizer.step()

                # update memory bank
                # update when queue size is divisible by batch size
                if key.shape[0]==self.params['loader_tr_args']['batch_size']: 
                    self._dequeue_and_enqueue(key)
        # for epoch in tqdm(range(1, int(n_epoch/2)+1), ncols=100):
        #     for batch_idx, (idxs,x_labeled, y_labeled,_,_,_ ) in enumerate(loader_tr):
        #         x_labeled, y_labeled = x_labeled.to(self.device),y_labeled.to(self.device)
        #         optimizer.zero_grad()
        #         out, e1 = self.clf_query(x_labeled)
        #         ce_loss = F.cross_entropy(out, y_labeled)
        #         ce_loss.backward()
        #         optimizer.step()
    def predict(self, data):
        self.clf_query.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x,x1, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf_query(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.params['memory_size'] % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.params['memory_size']  # move pointer
        self.queue_ptr[0] = ptr
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.clf_query.parameters(), self.clf_key.parameters()):
            param_k.data = param_k.data * self.params['moco_momentum'] + param_q.data * (1.0 - self.params['moco_momentum'])

    def _compute_unlabel_contrastive_loss(self,query,positive_key):
        """ Calculates the unlabel contrastive loss for self-supervised learning.
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
        self.clf_query.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x,x1, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf_query(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs
    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf_query.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf_query(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs
    def predict_prob_dropout(self, data, n_drop=10):
        self.clf_query.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x,x1, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf_query(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs
    def get_model(self):
        return self.clf_query

 
 
  