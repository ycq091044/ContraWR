import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MoCo(torch.nn.modules.loss._Loss):
    def __init__(self, device, T=0.5):
        """
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive, queue):
        
        # L2 normalize
        emb_anchor = torch.mm(torch.diag(torch.sum(torch.pow(emb_anchor, 2), axis=1) ** (-0.5)), emb_anchor)
        emb_positive = torch.mm(torch.diag(torch.sum(torch.pow(emb_positive, 2), axis=1) ** (-0.5)), emb_positive)
        queue = torch.mm(torch.diag(torch.sum(torch.pow(queue, 2), axis=1) ** (-0.5)), queue)

        # positive logits: Nx1, negative logits: NxK
        l_pos = torch.einsum('nc,nc->n', [emb_anchor, emb_positive]).unsqueeze(-1)
        l_neg = torch.einsum('nc,kc->nk', [emb_anchor, queue])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class BYOL(torch.nn.modules.loss._Loss):
    """

    """
    def __init__(self, device, T=0.5):
        """
        T: softmax temperature (default: 0.07)
        """
        super(BYOL, self).__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive):

        # L2 normalize
        emb_anchor = torch.mm(torch.diag(torch.sum(torch.pow(emb_anchor, 2), axis=1) ** (-0.5)), emb_anchor)
        emb_positive = torch.mm(torch.diag(torch.sum(torch.pow(emb_positive, 2), axis=1) ** (-0.5)), emb_positive)

        # positive logits: Nxk, negative logits: NxK
        l_pos = torch.einsum('nc,nc->n', [emb_anchor, emb_positive]).unsqueeze(-1)
        l_neg = torch.mm(emb_anchor, emb_positive.t())

        loss = - l_pos.sum()
                
        return loss


class SimSiam(torch.nn.modules.loss._Loss):

    def __init__(self, device, T=0.5):
        """
        T: softmax temperature (default: 0.07)
        """
        super(SimSiam, self).__init__()
        self.T = T
        self.device = device

    def forward(self, p1, p2, z1, z2):

        # L2 normalize
        p1 = F.normalize(p1, p=2, dim=1)
        p2 = F.normalize(p2, p=2, dim=1)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # mutual prediction
        l_pos1 = torch.einsum('nc,nc->n', [p1, z2.detach()]).unsqueeze(-1)
        l_pos2 = torch.einsum('nc,nc->n', [p2, z1.detach()]).unsqueeze(-1)

        loss = - (l_pos1.sum() + l_pos2.sum())
                
        return loss


class OurLoss(torch.nn.modules.loss._Loss):

    def __init__(self, device, margin=0.5, sigma=2.0, T=2.0):
        """
        T: softmax temperature (default: 0.07)
        """
        super(OurLoss, self).__init__()
        self.T = T
        self.device = device
        self.margin = margin
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigma = sigma


    def forward(self, emb_anchor, emb_positive):
        
        # L2 normalize, Nxk, Nxk
        emb_anchor = F.normalize(emb_anchor, p=2, dim=1)
        emb_positive = F.normalize(emb_positive, p=2, dim=1)

        # compute instance-aware world representation, Nx1
        sim = torch.mm(emb_anchor, emb_positive.t()) / self.T
        weight = self.softmax(sim)
        neg = torch.mm(weight, emb_positive)

        # representation similarity of pos/neg pairs
        l_pos = torch.exp(-torch.sum(torch.pow(emb_anchor - emb_positive, 2), dim=1) / (2 * self.sigma ** 2))
        l_neg = torch.exp(-torch.sum(torch.pow(emb_anchor - neg, 2), dim=1) / (2 * self.sigma ** 2))

        zero_matrix = torch.zeros(l_pos.shape).to(self.device)
        loss = torch.max(zero_matrix, l_neg - l_pos + self.margin).mean()
        
        return loss


class SimCLR(torch.nn.modules.loss._Loss):

    def __init__(self, device, T=1.0):
        """
        T: softmax temperature (default: 0.07)
        """
        super(SimCLR, self).__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive):
        
        # L2 normalize
        emb_anchor = torch.mm(torch.diag(torch.sum(torch.pow(emb_anchor, 2), axis=1) ** (-0.5)), emb_anchor)
        emb_positive = torch.mm(torch.diag(torch.sum(torch.pow(emb_positive, 2), axis=1) ** (-0.5)), emb_positive)
        N = emb_anchor.shape[0]
        emb_total = torch.cat([emb_anchor, emb_positive], dim=0)

        # representation similarity matrix, NxN
        logits = torch.mm(emb_total, emb_total.t())
        logits[torch.arange(2*N), torch.arange(2*N)] =  -1e10
        logits /= self.T

        # cross entropy
        labels = torch.LongTensor(torch.cat([torch.arange(N, 2*N), torch.arange(N)])).to(self.device)
        loss = F.cross_entropy(logits, labels)
                
        return loss