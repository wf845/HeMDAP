import numpy
import scipy as sp
import torch
from matplotlib import pyplot as plt
from load_data import *
from params import *
from utils import *
from module import contrast
import pandas as pd
from module import Hemdap
from module import Contrast
import warnings
import datetime
import pickle as pkl
import os
import random
import os
import gc

from torch import nn, optim
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.metrics import precision_score
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
from numpy import *

import copy
import torch

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(device)
else:
    device = torch.device("cpu")

args = model_params()
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class Config(object):
    def __init__(self):
        # self.validation = 5
        self.epoch = 200
        self.fold = 5
        self.seed = 4
        self.dim = 128
        self.lr = 0.0045

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, positive_seq, negative_seq):
        positive_loss = nn.BCELoss()(positive_seq, torch.ones_like(positive_seq))

        negative_loss = nn.BCELoss()(negative_seq, torch.zeros_like(negative_seq))

        loss = positive_loss + negative_loss

        return loss



class LinkPrediction(nn.Module):
    def __init__(self, ft_in):
        super(LinkPrediction, self).__init__()
        self.fc = nn.Linear(ft_in, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq):
        seq = seq.float()
        logits = self.fc(seq)
        predictions = self.sigmoid(logits)
        return predictions



class LinkPredictionTrainer:
    def __init__(self, model, optimizer, my_loss):
        self.model = model
        self.optimizer = optimizer
        self.my_loss = my_loss

    def train_step(self, train_data, mirna_features, disease_features):
        self.optimizer.zero_grad()
        np.random.seed(4)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        model.train()
        one_index, zero_index1 = train_data['md_train'][0].cuda(), train_data['md_train'][1].cuda()

        positive = []
        negative = []

        for i in range(len(one_index)):
            current_pair = one_index[i]
            miRNA_index, disease_index = current_pair[0], current_pair[1]
            current_miRNA_feat_one = disease_features[disease_index]
            current_disease_feat_one = mirna_features[miRNA_index]
            feats_one = torch.cat((current_miRNA_feat_one, current_disease_feat_one), dim=0)
            feats_one = feats_one.to('cuda:0')
            positive_probs = self.model(feats_one)
            positive.append(positive_probs)

        for j in range(len(zero_index1)):
            current_negative_pair = zero_index1[j]
            miRNA_index, disease_index = current_negative_pair[0], current_negative_pair[1]
            current_miRNA_feat_zero = disease_features[disease_index]
            current_disease_feat_zero = mirna_features[miRNA_index]
            feats_zero = torch.cat((current_miRNA_feat_zero, current_disease_feat_zero), dim=0)
            feats_zero = feats_zero.to('cuda:0')
            negative_probs = self.model(feats_zero)
            negative.append(negative_probs)

        positive = torch.stack(positive)
        negative = torch.stack(negative)
        loss = self.my_loss(positive, negative)
        loss.backward()
        self.optimizer.step()

        return loss



if __name__ == "__main__":

    D = np.genfromtxt(r"./data/md-matrix.txt")
    nei_index1, feats1, mps1, pos1 = load_m()
    nei_index2, feats2, mps2, pos2 = load_d()
    feats_dim_list1 = [i.shape[1] for i in feats1]
    feats_dim_list2 = [i.shape[1] for i in feats2]
    P1 = int(len(mps1))
    P2 = int(len(mps2))
    print("seed ", args.seed)
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", P1, P2)

    model = Hemdap(args.hidden_dim, feats_dim_list1, feats_dim_list2, args.feat_drop, args.attn_drop,
                 P1, P2, args.sample_rate,args.sample_rate1, args.nei_num, args.tau, args.lam,args.gamma)
    LOSS = Contrast(64, args.tau, args.lam)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        LOSS.cuda()
        feats1 = [feat.cuda() for feat in feats1]
        feats2 = [feat.cuda() for feat in feats2]
        mps1 = [mp.cuda() for mp in mps1]
        mps2 = [mp.cuda() for mp in mps2]
        pos1 = pos1.cuda()
        pos2 = pos2.cuda()

    cnt_wait = 0
    best = 1e9
    best_t = 0

    starttime = datetime.datetime.now()
    for epoch in range(args.nb_epochs):
        model.train()
        optimiser.zero_grad()
        z_mp1, z_sc1, z_mp2, z_sc2 = model(feats1, feats2, mps1, mps2, nei_index1, nei_index2)
        loss = LOSS(z_mp1, z_sc1, pos1, z_mp2, z_sc2, pos2, D)

        print("loss ", loss.data.cpu())
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
        else:
            cnt_wait += 1
        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        loss.backward()
        optimiser.step()

    print('Loading {}th epoch'.format(best_t))
    model.eval()

    embed1, embed2 = model.get_embeds(feats1, feats2, mps1, mps2, nei_index1, nei_index2)
    # print(embed1.shape),print(embed2.shape)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")

    opt = Config()

    mirna_features = embed1
    disease_features = embed2

    [row, col] = np.shape(D)
    indexn = np.argwhere(D == 0)
    Index_zeroRow = indexn[:, 0]
    Index_zeroCol = indexn[:, 1]
   
    indexp = np.argwhere(D == 1)
    Index_PositiveRow = indexp[:, 0]
    Index_PositiveCol = indexp[:, 1]
    totalassociation = np.size(Index_PositiveRow)
    fold = int(totalassociation / opt.fold)
    zero_length = np.size(Index_zeroRow)
    cv_num = opt.fold
   
    varauc = []
    AAuc_list1 = []

    varf1_score = []
    f1_score_list1 = []
    varprecision = []
    precision_list1 = []
    varrecall = []
    recall_list1 = []
    varaupr = []
    aupr_list1 = []

    Auc_per = []
    f1_score_per = []
    precision_per = []
    recall_per = []
    aupr_per = []

    np.random.seed(opt.seed)
    p = np.random.permutation(totalassociation)
    model = LinkPrediction(ft_in=opt.dim)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    model = model.cuda()
    my_loss = MyLoss()
    trainer = LinkPredictionTrainer(model, optimizer, my_loss)

    for f in range(1, cv_num + 1):
        optimiser.zero_grad()
        print("cross_validation:", '%01d' % (f))
        if f == cv_num:
            testset = p[((f - 1) * fold): totalassociation + 1]
        else:
            testset = p[((f - 1) * fold): f * fold]

        # test pos and neg
        all_f = np.random.permutation(np.size(Index_zeroRow))
        test_p = list(testset)

        if f == 1:
            test_f = all_f[5*len(test_p): 6*len(test_p)]
        else:
            test_f = all_f[len(test_p): 2*len(test_p)]

        difference_set_f = list(set(all_f).difference(set(test_f)))
        train_p = list(set(p).difference(set(testset)))
        train_f = difference_set_f
        X = copy.deepcopy(D)
        Xn = copy.deepcopy(X)

        zero_index = []
        for ii in range(len(train_f)):
            zero_index.append([Index_zeroRow[train_f[ii]], Index_zeroCol[train_f[ii]]])
        true_list = zeros((len(test_p) + len(test_f), 1))
        # exclude the testset during training.
        for ii in range(len(test_p)):
            Xn[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 2
            true_list[ii, 0] = 1
        for ii in range(len(test_f)):
            Xn[Index_zeroRow[test_f[ii]], Index_zeroCol[test_f[ii]]] = 3
        D1 = copy.deepcopy(Xn)
        print(D1)

        train_data = prepare_data(opt, D1)

        gc.collect()
        torch.cuda.empty_cache()


        cnt_wait1 = 0
        best1 = 1e9
        best_t2 = 0

        for epoch in range(1, opt.epoch + 1):
            loss = trainer.train_step(train_data, mirna_features, disease_features)
            print("loss ", loss.data.cpu())
            if loss < best1:
                # print(loss)
                best1 = loss
                best_t2 = epoch
                cnt_wait1 = 0
            else:
                cnt_wait1 += 1
            if cnt_wait == args.patience:
                print('Early stopping!')
                break
            loss.backward()
            optimizer.step()

        # Set the model to evaluation mode.
        model.eval()

        test_length_p = len(test_p)

        result_list = zeros((test_length_p + len(test_f), 1))
        for i in range(test_length_p):
            miRNA_feats_one = mirna_features[Index_PositiveRow[testset[i]]]
            disease_feats_one = disease_features[Index_PositiveCol[testset[i]]]
            miRNA_feats_one = torch.tensor(miRNA_feats_one).to('cuda:0').float()
            disease_feats_one = torch.tensor(disease_feats_one).to('cuda:0').float()
            feats_one = torch.cat((miRNA_feats_one, disease_feats_one), dim=0)
            result_list[i, 0] = model(feats_one)
        for i in range(len(test_f)):
            miRNA_feats_zero = mirna_features[Index_zeroRow[test_f[i]]]
            disease_feats_zero = disease_features[Index_zeroCol[test_f[i]]]
            miRNA_feats_zero = torch.tensor(miRNA_feats_zero).to('cuda:0').float()
            disease_feats_zero = torch.tensor(disease_feats_zero).to('cuda:0').float()
            feats_zero = torch.cat((miRNA_feats_zero, disease_feats_zero), dim=0)
            result_list[i + test_length_p, 0] = model(feats_zero)
        test_predict = result_list
        label = true_list
        test_auc = roc_auc_score(label, test_predict)

        Auc_per.append(test_auc)
        print("//////////every-auc: " + str(test_auc))
        varauc.append(test_auc)

        ####
        max_f1_score, threshold = f1_score_binary(torch.from_numpy(label).float(),
                                                  torch.from_numpy(test_predict).float())
        f1_score_per.append(max_f1_score)
        print("//////////max_f1_score:", max_f1_score)

        precision = precision_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(),
                                     threshold)
        precision_per.append(precision)
        print("//////////precision:", precision)
        recall = recall_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        recall_per.append(recall)
        print("//////////recall:", recall)

        pr, re, thresholds = precision_recall_curve(label, test_predict)
        aupr = auc(re, pr)
        aupr_per.append(aupr)
        print("//////////aupr", aupr)

        varf1_score.append(max_f1_score)
        varprecision.append(precision)
        varrecall.append(recall)
        varaupr.append(aupr)
    vauc = np.var(varauc)

    vf1_score = np.var(varf1_score)
    vprecision = np.var(varprecision)
    vrecall = np.var(varrecall)
    vaupr = np.var(varaupr)
 

    print("sumauc = %f±%f\n" % (float(np.mean(AAuc_list1)), vauc))

    print("sumf1_score = %f±%f\n" % (float(np.mean(f1_score_list1)), vf1_score))
    print("sumprecision = %f±%f\n" % (float(np.mean(precision_list1)), vprecision))
    print("sumrecall = %f±%f\n" % (float(np.mean(recall_list1)), vrecall))
    print("sumaupr = %f±%f\n" % (float(np.mean(aupr_list1)), vaupr))
            
            
            
            
            
            
            
