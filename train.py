# coding=utf-8

from __future__ import division
from __future__ import print_function
from sklearn import metrics
import os
import time
import argparse
import numpy as np
import math
import torch
import torch.optim as optim
from utils import *
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--episodes', type=int, default=1000,
                    help='Number of episodes to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lamda', type=float, default=1.0, help='Distance coefficient')
parser.add_argument('--alpha', type=float, default=1.0, help='Loss coefficient')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--mode', default='With OOD', help='With OOD/Without OOD')
parser.add_argument('--scheme', default='Probability-based',
                    help='Distance-based/Probability-based')
parser.add_argument('--way', type=int, default=5, help='way.')
parser.add_argument('--shot', type=int, default=5, help='shot.')
parser.add_argument('--qry', type=int, help='m shot for query set', default=20)
parser.add_argument('--dataset', default='dblp', help='Dataset:Amazon_clothing/Amazon_eletronics/dblp/Cora')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


dataset = args.dataset
if args.mode=='With OOD':
    adj, features, labels, degrees, class_list_train, class_list_valid,  class_list_test, OOD_class_list, id_by_class = load_data(dataset)
    OOD_sample = []
    OOD_train_sample = []
    OOD_test_sample = []
    for i in range(10):
        sample = id_by_class[OOD_class_list[i]]
        OOD_sample.extend(sample)
    l =  len(OOD_sample)

    OOD_train_sample.extend(OOD_sample[:int(l / 2)])
    OOD_test_sample.extend(OOD_sample[int(l / 2):])
else:
    adj, features, labels, degrees, class_list_train, class_list_valid,  class_list_test, id_by_class = load_data(dataset)


encoder = GPN_Encoder(nfeat=features.shape[1],
            nhid=args.hidden,
            dropout=args.dropout)


scorer = GPN_Valuator(nfeat=features.shape[1],
            nhid=args.hidden,
            dropout=args.dropout)

optimizer_encoder = optim.Adam(encoder.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

optimizer_scorer = optim.Adam(scorer.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    encoder.cuda()
    scorer.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    degrees = degrees.cuda()

def train(class_selected, id_support, id_query, n_way, k_shot):
    encoder.train()
    scorer.train()
    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings =  support_embeddings * support_scores
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)
    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_train = F.nll_loss(output, labels_new)
    loss_train.backward()
    optimizer_encoder.step()
    optimizer_scorer.step()
    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)
    return acc_train, f1_train


def test(class_selected, id_support, id_query, n_way, k_shot):
    encoder.eval()
    scorer.eval()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)
    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)
    return acc_test, f1_test


def OOD_train_C1(id_support, id_query, OOD_sample, ID_class_selected, n_way, k_shot, m_shot):
    encoder.train()
    scorer.train()
    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]
    OOD_embeddings = embeddings[OOD_sample]
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1,keepdim=True)
    support_embeddings = support_embeddings * support_scores
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    dists_OOD = euclidean_dist(OOD_embeddings, prototype_embeddings)
    ID_max_dist = []
    for i in range(n_way):
        temp1 = prototype_embeddings[i].view([1, z_dim])
        temp2 = embeddings[id_support[k_shot*i:k_shot*(i+1)]]
        ID_dist = euclidean_dist(temp2, temp1)
        radius = ID_dist.cpu().detach().numpy().max()
        ID_max_dist.append(radius)

    reward = 0
    for i in range(m_shot):
        flag = 1
        for j in range(n_way):
            if dists_OOD[i][j] < lamda * ID_max_dist[j]:
                flag = 0
        if (flag == 1):
            reward = reward + 1
    reward = reward / m_shot
    dist_aver = 0
    for i in range(m_shot):
        for j in range(n_way):
            dist_aver = dist_aver + (dists_OOD[i][j] - ID_max_dist[j])
    dist_aver = dist_aver / (m_shot * n_way)

    loss_OOD = -(dist_aver * reward)
    loss_OOD = 0.009 * loss_OOD
    output = F.log_softmax(-dists, dim=1)
    output_OOD = F.softmax(-dists_OOD, dim=1)
    labels_new = torch.LongTensor([ID_class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_train = F.nll_loss(output, labels_new)
    loss_train_with_OOD = loss_train + 0.9 * loss_OOD
    loss_train_with_OOD.backward()
    optimizer_encoder.step()
    optimizer_scorer.step()
    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    return loss_train_with_OOD


def OOD_test_C1(id_support, id_query, OOD_sample, ID_class_selected, n_way, k_shot, m_shot):
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]
    OOD_embeddings = embeddings[OOD_sample]
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1,keepdim=True)
    support_embeddings = support_embeddings * support_scores
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    dists_OOD = euclidean_dist(OOD_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)
    output_OOD = F.softmax(-dists_OOD, dim=1)
    labels_new = torch.LongTensor([ID_class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    out = F.softmax(-dists, dim=1)
    out_max, inds_max = torch.max(output_OOD, dim=1)
    score_ID, inds = torch.max(out, dim=1)
    score_ID = score_ID.cpu().detach().numpy()
    score_OOD = 1 - out_max
    score_OOD = score_OOD.cpu().detach().numpy()
    y_score = np.hstack((score_ID, score_OOD))
    y_true = np.hstack((np.ones(n_way * m_shot), np.zeros(m_shot)))
    auROC_test = metrics.roc_auc_score(y_true, y_score)
    return auROC_test


def OOD_train_C2(id_support, id_query, OOD_sample, ID_class_selected, n_way, k_shot, m_shot):
    encoder.train()
    scorer.train()
    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]
    OOD_embeddings = embeddings[OOD_sample]
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1,keepdim=True)
    support_embeddings = support_embeddings * support_scores
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    dists_OOD = euclidean_dist(OOD_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)
    output_OOD = F.log_softmax(-dists_OOD, dim=1)
    labels_new = torch.LongTensor([ID_class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_train = F.nll_loss(output, labels_new)
    loss_OOD = []
    for i in range(n_way):
        loss_OOD[i]=F.nll_loss(output_OOD, torch.LongTensor([i for _ in range(m_shot)]))
    loss_train_with_OOD = loss_train
    for i in range(n_way):
        loss_train_with_OOD += loss_OOD[i] * 0.018
    print(loss_train)
    print(loss_train_with_OOD-loss_train)
    loss_train_with_OOD.backward()
    optimizer_encoder.step()
    optimizer_scorer.step()
    if args.cuda:
        output.cpu().detach()
        labels_new.cpu().detach()
    return loss_train_with_OOD


def OOD_test_C2(id_support, id_query, OOD_sample, ID_class_selected, n_way, k_shot, m_shot):
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]
    OOD_embeddings = embeddings[OOD_sample]
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    dists_OOD = euclidean_dist(OOD_embeddings, prototype_embeddings)
    out = F.softmax(-dists, dim=1)
    output_OOD = F.softmax(-dists_OOD, dim=1)
    out_max, inds_max = torch.max(output_OOD, dim=1)
    score_ID, inds = torch.max(out, dim=1)
    score_ID = score_ID.cpu().detach().numpy()
    score_OOD = 1-out_max
    score_OOD = score_OOD.cpu().detach().numpy()
    y_score = np.hstack((score_ID, score_OOD))
    y_true = np.hstack((np.ones(n_way * m_shot), np.zeros(m_shot)))
    auROC_test = metrics.roc_auc_score(y_true, y_score)
    return auROC_test



if __name__ == '__main__':
    n_way = args.way
    k_shot = args.shot
    m_shot = args.qry
    lamda = args.lamda
    alpha = args.alpha
    mode = args.mode
    scheme = args.scheme
    episodes = args.episodes
    meta_valid_num = 50
    meta_test_num = 50

    valid_pool = [task_generator(id_by_class, class_list_valid, n_way, k_shot, m_shot) for i in range(meta_valid_num)]
    test_pool = [task_generator(id_by_class, class_list_test, n_way, k_shot, m_shot) for i in range(meta_test_num)]

    t_total = time.time()
    meta_train_acc = []

    for episode in range(episodes):
        id_support, id_query, class_selected = task_generator(id_by_class, class_list_train, n_way, k_shot, m_shot)
        acc_train, f1_train = train(class_selected, id_support, id_query, n_way, k_shot)
        meta_train_acc.append(acc_train)
        if episode > 0 and episode % 10 == 0:
            print("-------Episode {}-------".format(episode))
            print("Meta-Train_Accuracy: {}".format(np.array(meta_train_acc).mean(axis=0)))
            meta_valid_acc = []
            meta_valid_f1 = []
            for idx in range(meta_valid_num):
                id_support, id_query, class_selected = valid_pool[idx]
                acc_valid, f1_valid = test(class_selected, id_support, id_query, n_way, k_shot)
                meta_valid_acc.append(acc_valid)
                meta_valid_f1.append(f1_valid)
            print("Meta-valid_Accuracy: {}, Meta-valid_F1: {}".format(np.array(meta_valid_acc).mean(axis=0),np.array(meta_valid_f1).mean(axis=0)))

            meta_test_acc = []
            meta_test_f1 = []
            for idx in range(meta_test_num):
                id_support, id_query, class_selected = test_pool[idx]
                acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
                meta_test_acc.append(acc_test)
                meta_test_f1.append(f1_test)
            print("Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(np.array(meta_test_acc).mean(axis=0),np.array(meta_test_f1).mean(axis=0)))

    if (mode == 'With OOD'):
        for i in range(500):
            OOD_loss_train = []
            id_support, id_query, OOD_sample, ID_class_selected = OOD_task_generator(id_by_class, class_list_valid, OOD_train_sample, n_way, k_shot, m_shot)
            if scheme == 'Distance-based':
                loss_train_with_OOD = OOD_train_C1(id_support, id_query, OOD_sample, ID_class_selected, n_way, k_shot, m_shot)
            else:
                loss_train_with_OOD = OOD_train_C2(id_support, id_query, OOD_sample, ID_class_selected, n_way, k_shot, m_shot)
            OOD_loss_train.append(loss_train_with_OOD)
            print("loss_train_with_OOD: {}".format(loss_train_with_OOD))

        for i in range(100):
            OOD_auROC_test = []
            id_support, id_query, OOD_sample, ID_class_selected = OOD_task_generator(id_by_class, class_list_test, OOD_test_sample, n_way, k_shot, m_shot)
            if (scheme == 'Distance-based'):
                auROC_test = OOD_test_C1(id_support, id_query, OOD_sample, ID_class_selected, n_way, k_shot, m_shot)
            else:
                auROC_test = OOD_test_C2(id_support, id_query, OOD_sample, ID_class_selected, n_way, k_shot, m_shot)
            OOD_auROC_test.append(auROC_test)
        print("OOD_auROC_test: {}".format(np.mean(OOD_auROC_test)))

        meta_test_acc = []
        meta_test_f1 = []
        test_pool = [task_generator(id_by_class, class_list_test, n_way, k_shot, m_shot) for i in range(meta_test_num)]
        for idx in range(meta_test_num):
            id_support, id_query, class_selected = test_pool[idx]
            acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
            meta_test_acc.append(acc_test)
            meta_test_f1.append(f1_test)
        print("Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(np.array(meta_test_acc).mean(axis=0),np.array(meta_test_f1).mean(axis=0)))

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

