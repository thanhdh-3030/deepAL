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
        self.unlabel_queue=torch.randn(self.params['embedding_dim'],self.params['memory_size']).cuda() # (K,D)
        self.unlabel_queue=F.normalize(self.unlabel_queue,dim=0)
        self.unlabel_queue_ptr=torch.zeros(1, dtype=torch.long) # (1,)
        self.label_queue=torch.randn(self.params['num_class'],self.params['embedding_dim'],self.params['memory_size']).cuda() # (K,D)
        self.label_queue=F.normalize(self.unlabel_queue,dim=1)
        self.label_queue_ptr=torch.zeros((self.params['num_class'],1), dtype=torch.long) # (1,)
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
                out, query = self.clf(x1)
                prob = F.softmax(out,dim=1)
                pred=out.max(1)[1]
                onebit_mask=(pred==y)
                with torch.no_grad():  # no gradient to keys
                    _, key = self.clf(x2)
                key=key.detach()
                # normalize embedding
                query=F.normalize(query,dim=1)
                key=F.normalize(key,dim=1)
                contrast_loss=self._compute_unlabel_contrastive_loss(query,key)
                # contrast_criterion=NTXentLoss(device=self.device,batch_size=x1.shape[0],temperature=0.1,use_cosine_similarity=False)
                # contrast_loss=contrast_criterion(query,key)
                ce_loss = F.cross_entropy(out, y)
                total_loss=self.params['contrast_weight']*contrast_loss+ce_loss
                total_loss.backward()
                optimizer.step()

                # update memory bank
                # update when queue size is divisible by batch size
                if self.params['memory_size']%key.shape[0]==0: 
                    self._dequeue_and_enqueue(key)
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
        ptr = int(self.unlabel_queue_ptr)
        print(self.queue_ptr)
        print(batch_size)
        assert self.params['memory_size'] % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.unlabel_queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.params['memory_size']  # move pointer
        self.unlabel_queue_ptr[0] = ptr

    def _compute_unlabel_contrastive_loss(self,query,positive_key):
        # get negative keys form unlabel memory bank
        negative_keys=self.unlabel_queue.clone().detach()
        contrast_loss=self._compute_contrast_loss(query,positive_key,negative_keys)
        return contrast_loss
    
    def _compute_positive_contrastive_loss(self,query,keys_label,positive_key):
        """ Calculate contrastive loss enfoces the embeddings of same class
            to be close and different class far away.
        """
        # calculate augment postitive contrastive loss
        negative_keys=self.unlabel_queue.clone().detach()
        aug_contrast_loss=self._compute_contrast_loss(query,positive_key,negative_keys)

        # calculate same class positive contrastive loss
        same_positive_key=self.label_queue[keys_label][:,0] # get first embedding in label memory bank
        same_contrast_loss=self._compute_contrast_loss(query,same_positive_key,negative_keys)

        positive_contrast_loss=(aug_contrast_loss+same_contrast_loss)/2
        return positive_contrast_loss
    
    def _compute_negative_contrastive_loss(self,query,keys_label,positive_key):
        # get negative keys
        diff_class_negative_keys=self.label_queue.clone().detach()[keys_label][:,:128] # negative keys have label model predict wrongly
        diff_img_negative_keys=self.unlabel_queue.clone().detach()[:,:128] # negative keys formed by other images
        negative_keys=torch.cat([diff_class_negative_keys,diff_img_negative_keys],dim=1)
        contrast_loss=self._compute_contrast_loss(query,positive_key,negative_keys)
        return contrast_loss    

    def _compute_contrast_loss(self,query,positive_key,negative_keys):
        """ Calculates the unlabel contrastive loss for self-supervised learning.
        Args:
            query (torch.Tensor): Tensor with query samples (e.g. embeddings of the input).
                (N,D) where D is embedding dim.
            positive_keys (torch.Tensor): Tensor with positive samples (e.g. embeddings of augmented input).
                (N,D).
            negative_keys (torch.Tensor): Tensor with negative samples (e.g. embeddings of augmented input).
                (M,D).
        Returns:
            torch.Tensor: value of contrastive loss.
        """
        # normalize query features and positive key
        query=F.normalize(query,dim=1)
        positive_key=F.normalize(positive_key,dim=1)
        negative_keys=F.normalize(negative_keys,dim=1)
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [query, positive_key]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [query, negative_keys])
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

 
 
  