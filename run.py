import sys

import torch

from model import Model
from utils import *
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import argparse
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='NLGAD')
parser.add_argument('--expid', type=int, default=3)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--num_epoch', type=int, default=700)
parser.add_argument('--select_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio_patch', type=int, default=6)
parser.add_argument('--negsamp_ratio_context', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.6, help='how much context-level involves')
args = parser.parse_args()

if __name__ == '__main__':

    print('Dataset: {}'.format(args.dataset), flush=True)
    print('Alpha: ', args.alpha)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    batch_size = args.batch_size
    subgraph_size = args.subgraph_size

    adj, features, _, _, _, _, ano_label, _, _ = load_mat(args.dataset)

    features, _ = preprocess_features(features)
    dgl_graph = adj_to_dgl_graph(adj)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis]).to(device)
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)


    all_auc = []


    for run in range(args.runs):
        seed = run + 1
        print('\n# Run:{} with random seed:{}'.format(run, seed), flush=True)
        dgl.random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio_patch, args.negsamp_ratio_context,
                      args.readout).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        b_xent_patch = nn.BCEWithLogitsLoss(reduction='none',
                                            pos_weight=torch.tensor([args.negsamp_ratio_patch]).to(device))
        b_xent_context = nn.BCEWithLogitsLoss(reduction='none',
                                            pos_weight=torch.tensor([args.negsamp_ratio_context]).to(device))

        cnt_wait = 0
        best = 1e9
        best_t = 0


        all_idx = list(range(nb_nodes))
        memorybank = []

        train_ano_score = torch.zeros((args.select_epoch, nb_nodes), dtype=torch.float).cuda()

        with tqdm(total=args.num_epoch) as pbar_training:
            pbar_training.set_description('Training')
            for epoch in range(args.num_epoch):

                if epoch <= args.select_epoch and epoch:
                    _, train_list_temp = train_ano_score[epoch - 1].topk(int(nb_nodes - (epoch / args.select_epoch) ** 2 * nb_nodes), dim=0,
                                                                  largest=False, sorted=True)
                    train_list_temp = train_list_temp.cpu().numpy()
                    train_list_temp = train_list_temp.tolist()
                    memorybank.append(train_list_temp)

                if epoch == args.select_epoch:
                    train_ano_score = train_ano_score.cpu().detach().numpy()
                    scaler = MinMaxScaler()
                    train_ano_score = scaler.fit_transform(train_ano_score.T).T
                    train_ano_score = torch.DoubleTensor(train_ano_score).cuda()
                    for idx in range(len(memorybank)):
                        train_ano_score[idx, memorybank[idx]] = 0
                    train_ano_score_nonzero = torch.count_nonzero(train_ano_score, dim=0)
                    train_ano_score = torch.sum(train_ano_score, dim=0)
                    train_ano_score = train_ano_score / train_ano_score_nonzero
                    _, train_list = train_ano_score.topk(int(0.80 * nb_nodes), dim=0, largest=False, sorted=True)
                    train_list = train_list.cpu().numpy()
                    train_list = train_list.tolist()
                    all_idx = train_list
                    print('')
                    print(len(all_idx))
                    print(len(all_idx) / nb_nodes)
                    print(np.sum(ano_label[all_idx]))
                    best =  float ('inf')
                    best_t = epoch

                model.train()
                random.shuffle(all_idx)
                total_loss = 0.
                batch_num = len(all_idx) // batch_size + 1

                subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

                for batch_idx in range(batch_num):

                    optimiser.zero_grad()

                    is_final_batch = (batch_idx == (batch_num - 1))
                    if not is_final_batch:
                        idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                    else:
                        idx = all_idx[batch_idx * batch_size:]

                    cur_batch_size = len(idx)

                    lbl_patch = torch.unsqueeze(torch.cat(
                        (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_patch))), 1).to(device)

                    lbl_context = torch.unsqueeze(torch.cat(
                        (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_context))), 1).to(device)

                    ba = []
                    bf = []
                    added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                    added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                    added_adj_zero_col[:, -1, :] = 1.
                    added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

                    for i in idx:
                        cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                        cur_feat = features[:, subgraphs[i], :]
                        ba.append(cur_adj)
                        bf.append(cur_feat)

                    ba = torch.cat(ba)
                    ba = torch.cat((ba, added_adj_zero_row), dim=1)
                    ba = torch.cat((ba, added_adj_zero_col), dim=2)
                    bf = torch.cat(bf)
                    bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                    logits_1, logits_2 = model(bf, ba)

                    # Context-level
                    loss_all_1 = b_xent_context(logits_1, lbl_context)
                    loss_1 = torch.mean(loss_all_1)

                    # Patch-level
                    loss_all_2 = b_xent_patch(logits_2, lbl_patch)
                    loss_2 = torch.mean(loss_all_2)

                    test_logits_1 = torch.sigmoid(torch.squeeze(logits_1))
                    test_logits_2 = torch.sigmoid(torch.squeeze(logits_2))

                    loss = args.alpha * loss_1 + (1 - args.alpha) * loss_2

                    if epoch < args.select_epoch:
                        if args.alpha != 1.0 and args.alpha != 0.0:
                            if args.negsamp_ratio_context == 1 and args.negsamp_ratio_patch == 1:
                                ano_score_1 = - (
                                            test_logits_1[:cur_batch_size] - test_logits_1[cur_batch_size:])
                                ano_score_2 = - (
                                            test_logits_2[:cur_batch_size] - test_logits_2[cur_batch_size:])
                            else:
                                ano_score_1 = - (
                                            test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                                        cur_batch_size, args.negsamp_ratio_context), dim=1))  # context
                                ano_score_2 = - (
                                            test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                                        cur_batch_size, args.negsamp_ratio_patch), dim=1))  # patch
                            ano_score = args.alpha * ano_score_1 + (1 - args.alpha) * ano_score_2
                        elif args.alpha == 1.0:
                            if args.negsamp_ratio_context == 1:
                                ano_score = - (
                                            test_logits_1[:cur_batch_size] - test_logits_1[cur_batch_size:])
                            else:
                                ano_score = - (
                                            test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                                        cur_batch_size, args.negsamp_ratio_context), dim=1))  # context
                        elif args.alpha == 0.0:
                            if args.negsamp_ratio_patch == 1:
                                ano_score = - (
                                            test_logits_2[:cur_batch_size] - test_logits_2[cur_batch_size:])
                            else:
                                ano_score = - (
                                            test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                                        cur_batch_size, args.negsamp_ratio_patch), dim=1))  # patch
                        train_ano_score[epoch, idx] = ano_score
                    # else:
                    loss.backward()
                    optimiser.step()

                    loss = loss.detach().cpu().numpy()
                    if not is_final_batch:
                        total_loss += loss

                mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes

                if mean_loss < best:
                    best = mean_loss
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(model.state_dict(), '{}.pkl'.format(args.dataset))
                else:
                    cnt_wait += 1


                pbar_training.set_postfix(loss=mean_loss)
                pbar_training.update(1)

        # Testing
        print('Loading {}th epoch'.format(best_t), flush=True)
        model.load_state_dict(torch.load('{}.pkl'.format(args.dataset)))
        multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
        print('Testing AUC!', flush=True)

        with tqdm(total=args.auc_test_rounds) as pbar_test:
            pbar_test.set_description('Testing')
            for round in range(args.auc_test_rounds):
                all_idx = list(range(nb_nodes))
                random.shuffle(all_idx)
                subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
                for batch_idx in range(batch_num):
                    optimiser.zero_grad()
                    is_final_batch = (batch_idx == (batch_num - 1))
                    if not is_final_batch:
                        idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                    else:
                        idx = all_idx[batch_idx * batch_size:]
                    cur_batch_size = len(idx)
                    ba = []
                    bf = []
                    added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                    added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                    added_adj_zero_col[:, -1, :] = 1.
                    added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)
                    for i in idx:
                        cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                        cur_feat = features[:, subgraphs[i], :]
                        ba.append(cur_adj)
                        bf.append(cur_feat)

                    ba = torch.cat(ba)
                    ba = torch.cat((ba, added_adj_zero_row), dim=1)
                    ba = torch.cat((ba, added_adj_zero_col), dim=2)
                    bf = torch.cat(bf)
                    bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                    with torch.no_grad():
                        test_logits_1, test_logits_2 = model(bf, ba)
                        test_logits_1 = torch.sigmoid(torch.squeeze(test_logits_1))
                        test_logits_2 = torch.sigmoid(torch.squeeze(test_logits_2))

                    if args.alpha != 1.0 and args.alpha != 0.0:
                        if args.negsamp_ratio_context == 1 and args.negsamp_ratio_patch == 1:
                            ano_score_1 = - (test_logits_1[:cur_batch_size] - test_logits_1[cur_batch_size:]).cpu().numpy()
                            ano_score_2 = - (test_logits_2[:cur_batch_size] - test_logits_2[cur_batch_size:]).cpu().numpy()
                        else:
                            ano_score_1 = - (test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                                cur_batch_size, args.negsamp_ratio_context), dim=1)).cpu().numpy()  # context
                            ano_score_2 = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                                cur_batch_size, args.negsamp_ratio_patch), dim=1)).cpu().numpy()  # patch
                        ano_score = args.alpha * ano_score_1 + (1 - args.alpha) * ano_score_2
                    elif args.alpha == 1.0:
                        if args.negsamp_ratio_context == 1:
                            ano_score = - (test_logits_1[:cur_batch_size] - test_logits_1[cur_batch_size:]).cpu().numpy()
                        else:
                            ano_score = - (test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                                    cur_batch_size, args.negsamp_ratio_context), dim=1)).cpu().numpy()  # context
                    elif args.alpha == 0.0:
                        if args.negsamp_ratio_patch == 1:
                            ano_score = - (test_logits_2[:cur_batch_size] - test_logits_2[cur_batch_size:]).cpu().numpy()
                        else:
                            ano_score = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                                    cur_batch_size, args.negsamp_ratio_patch), dim=1)).cpu().numpy()  # patch

                    multi_round_ano_score[round, idx] = ano_score

                pbar_test.update(1)

            ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)
            auc = roc_auc_score(ano_label, ano_score_final)
            all_auc.append(auc)
            print('Testing AUC:{:.4f}'.format(auc), flush=True)



    print('\n==============================')
    print(all_auc)
    print('FINAL TESTING AUC:{:.4f}'.format(np.mean(all_auc)))
    print('==============================')