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
        self.unlabel_queue=torch.randn(self.params['embedding_dim'],self.params['memory_size']).cuda() # (K,D)
        self.unlabel_queue=F.normalize(self.unlabel_queue,dim=0)
        self.unlabel_queue_ptr=torch.zeros(1, dtype=torch.long) # (1,)
        self.label_queue=torch.randn(self.params['num_class'],self.params['embedding_dim'],self.params['memory_size']).cuda() # (K,D)
        self.label_queue=F.normalize(self.label_queue,dim=1)
        self.label_queue_ptr=torch.zeros((self.params['num_class'],1), dtype=torch.long) # (1,)
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
        # for epoch in tqdm(range(1, int(n_epoch)+1), ncols=100):
        for epoch in tqdm(range(1, int(n_epoch/100)+1), ncols=100):
            for batch_idx, (idxs,x_labeled,_, y_labeled,x_unlabeled1,x_unlabeled2,_ ) in enumerate(loader_tr):
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
                    self._dequeue_and_enqueue(key,y_labeled)
        for epoch in tqdm(range(1, int(n_epoch/100)+1), ncols=100):
            for batch_idx, (idxs,x_labeled1,x_labeled2, y_labeled,_,_,_ ) in enumerate(loader_tr):
                x_labeled1,x_labeled2, y_labeled = x_labeled1.to(self.device),x_labeled2.to(self.device),y_labeled.to(self.device)
                optimizer.zero_grad()
                out, feat = self.clf_query(x_labeled1)
                feat=F.normalize(feat,dim=1)  # normalize embedding
                with torch.no_grad():  # no gradient to keys
                    self._momentum_update_key_encoder()  # update the key encoder
                    _, key = self.clf_key(x_labeled2)
                    key=F.normalize(key,dim=1)
                pred=out.max(1)[1]
                onebit_mask=(pred==y_labeled)
                labels_onehot=F.one_hot(y_labeled,num_classes=self.params['num_class'])
                ce_loss,postive_ctr_loss=self._compute_yes_query_loss(out,feat,y_labeled,key,labels_onehot,onebit_mask)
                ce_loss,neg_ctr_loss=self._compute_yes_query_loss(out,feat,y_labeled,key,labels_onehot,onebit_mask)
                total_loss=(ce_loss+ce_loss)+self.params['contrast_weight']*(postive_ctr_loss+neg_ctr_loss)
                total_loss.backward()
                optimizer.step()
                # update memory bank
                # update when queue size is divisible by batch size
                if key.shape[0]==self.params['loader_tr_args']['batch_size']:
                    self._dequeue_and_enqueue(key,y_labeled) 
    def _compute_yes_query_loss(self,out,feature,label,feature_aug,labels_onehot,onebit_mask):
        out=out[onebit_mask]
        feature=feature[onebit_mask]
        feature_aug=feature_aug[onebit_mask]
        labels_onehot=labels_onehot[onebit_mask]
        label=label[onebit_mask]
        ce_loss = F.cross_entropy(out, label) # Cross Entropy loss
        postive_ctr_loss = torch.tensor(0.0)
        for i in range(self.params['num_class']):
            cls_mask=labels_onehot[:,i]
            query=feature[cls_mask]
            if(len(query)==0): continue
            aug_positive_key=feature_aug[cls_mask]
            postive_ctr_loss = postive_ctr_loss +self._compute_positive_contrastive_loss(query,i,aug_positive_key)
        return ce_loss,postive_ctr_loss
    def _compute_no_query_loss(self,pred,feature,label,feature_aug,labels_onehot,onebit_mask):
        pred=pred[~onebit_mask]
        feature=feature[~onebit_mask]
        feature_aug=feature_aug[~onebit_mask]
        labels_onehot=labels_onehot[~onebit_mask]
        label=label[onebit_mask]
        neg_loss = F.cross_entropy(pred, label) # Cross Entropy loss
        neg_ctr_loss = torch.tensor(0.0)
        for i in range(self.params['num_class']):
            cls_mask=labels_onehot[:,i]
            query=feature[cls_mask]
            if(len(query)==0): continue
            aug_positive_key=feature_aug[cls_mask]
            neg_ctr_loss = neg_ctr_loss +self._compute_negative_contrastive_loss(query,i,aug_positive_key)
        return neg_loss,neg_ctr_loss
    def _compute_unlabel_contrastive_loss(self,query,positive_key):
        # get negative keys form unlabel memory bank
        negative_keys=self.unlabel_queue.clone().detach()
        contrast_loss=self._compute_contrast_loss(query,positive_key,negative_keys)
        return contrast_loss

    def _compute_positive_contrastive_loss(self,query,query_category,aug_positive_key):
        """ Calculate contrastive loss enfoces the embeddings of same class
            to be close and different class far away.
        """
        # calculate augment postitive contrastive loss
        negative_keys=self.unlabel_queue.clone().detach()
        aug_contrast_loss=self._compute_contrast_loss(query,aug_positive_key,negative_keys)

        # calculate same class positive contrastive loss
        class_positive_key=self.label_queue.clone().detach()[query_category,:,:len(query)].T
        same_contrast_loss=self._compute_contrast_loss(query,class_positive_key,negative_keys)

        positive_contrast_loss=(aug_contrast_loss+same_contrast_loss)/2
        return positive_contrast_loss

    def _compute_negative_contrastive_loss(self,query,query_category,positive_key):
        # get negative keys
        diff_class_negative_keys=self.label_queue.clone().detach()[query_category,:,:128] # negative keys have label model predict wrongly
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
    def _dequeue_and_enqueue(self, keys,labels):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        ptr = int(self.unlabel_queue_ptr)
        assert self.params['memory_size'] % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        # update unlabel_queue
        self.unlabel_queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.params['memory_size']  # move pointer
        self.unlabel_queue_ptr[0] = ptr
        # update label_queue
        onehot_labels=F.one_hot(labels,num_classes=self.params['num_class'])
        for i in range(self.params['num_class']):
            cls_mask=onehot_labels[:,i]
            cls_keys=keys[cls_mask]
            if(len(cls_keys)==0):continue
            this_ptr=int(self.label_queue_ptr[i])
            self.label_queue[i,:,this_ptr:this_ptr+cls_keys.shape[0]]=cls_keys.T
            self.label_queue_ptr[i]=(this_ptr+cls_keys.shape[0])% self.params['memory_size']
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.clf_query.parameters(), self.clf_key.parameters()):
            param_k.data = param_k.data * self.params['moco_momentum'] + param_q.data * (1.0 - self.params['moco_momentum'])

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

 
 
  