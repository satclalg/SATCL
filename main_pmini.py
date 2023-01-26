import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import argparse
import time
from copy import deepcopy

from scipy.stats import wasserstein_distance


def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch == 1):
            param_group['lr'] = args.lr
        else:
            param_group['lr'] /= args.lr_factor


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, norm_feature, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)

    def forward(self, input, task_id, p, epoch):

        if p is not None:
            if epoch == 1:
                norm_project = torch.mm(p, p.transpose(1, 0))
                proj_weight = torch.mm(self.weight, norm_project)

                masked_weight = self.weight - proj_weight
            else:
                masked_weight = self.weight
        else:
            masked_weight = self.weight

        masked_weight = self.weight
        x = F.linear(input, masked_weight, self.bias)
        return x


# Define MLP model
class MLPNet(nn.Module):
    def __init__(self, n_hidden=100, n_outputs=10):
        super(MLPNet, self).__init__()
        self.act = OrderedDict()
        self.lin1 = Linear(784, n_hidden, n_hidden, bias=False)
        self.lin2 = Linear(n_hidden, n_hidden, n_hidden, bias=False)
        self.fc1 = Linear(n_hidden, n_outputs, n_outputs, bias=False)

    def forward(self, x, t, p, epoch):
        if p is None:
            self.act['Lin1'] = x
            x = self.lin1(x, None, epoch)
            x = F.relu(x)
            self.act['Lin2'] = x
            x = self.lin2(x, None, epoch)
            x = F.relu(x)
            self.act['fc1'] = x
            x = self.fc1(x, None, epoch)
        else:
            self.act['Lin1'] = x
            x = self.lin1(x, p[0], epoch)
            x = F.relu(x)
            self.act['Lin2'] = x
            x = self.lin2(x, p[1], epoch)
            x = F.relu(x)
            self.act['fc1'] = x
            x = self.fc1(x, p[2], epoch)
        return x


def contrast_cls(data, every_task_base, sim_tasks,
                 sim_scores, model, task_id, device):
    l2 = 0
    cnt = 0
    list_keys = list(model.act.keys())
    for k, (m, params) in enumerate(model.named_parameters()):

        sim = []
        # sim.append(params.flatten())

        for tt in range(task_id):
            sz = params.grad.data.size(0)
            tmp = torch.FloatTensor(every_task_base[tt][cnt]).to(device)
            norm_project = torch.mm(tmp, tmp.transpose(1, 0)).to(device)
            if "conv" in m:
                proj_weight = torch.mm(params.view(sz, -1),
                                       norm_project).view(params.size())
            else:
                proj_weight = torch.mm(params, norm_project)
            sim.append(proj_weight.data)
            dis = list(set(range(task_id)) - set(sim_tasks[cnt]))
            tt = random.sample(dis, 1)[0]
            tmp = torch.FloatTensor(every_task_base[tt][cnt]).to(device)
            norm_project = torch.mm(tmp, tmp.transpose(1, 0))
            proj_weight = torch.mm(params.view(sz, -1),
                                    norm_project).view(params.size())
            sim.append(proj_weight.data)
            if len(sim) >= 4:
                break

            sim = torch.stack(sim).view(4, -1)
            if sum(sim_scores[cnt]) != 2:
                idxs = torch.arange(0, sim.shape[0], device=device)
                y_true = idxs + 1 - idxs % 2 * 2
                similarities = F.cosine_similarity(sim.unsqueeze(1), sim.unsqueeze(0), dim=2)

                similarities = similarities - torch.eye(sim.shape[0], device=device) * 1e12
                similarities = similarities / 0.05

                loss = F.cross_entropy(similarities, y_true)
                l2 += torch.mean(loss)

            cnt += 1

    return l2


def train(args, epoch, task_id, model, device, x, y, optimizer, scheduler, criterion):
    model.train()
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0, len(r), args.batch_size_train):
        if i+args.batch_size_train <= len(r):
            b = r[i:i+args.batch_size_train]
        else:
            b = r[i:]
        data = x[b].view(-1, 28 * 28)
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()


def train_projected(args, p, model, device, x, y, optimizer, criterion, feature_mat, task_id, epoch, sim_tasks, sim_scores, every_task_base):
    model.train()
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0, len(r), args.batch_size_train):
        if i+args.batch_size_train <= len(r):
            b = r[i:i+args.batch_size_train]
        else:
            b = r[i:]
        data = x[b].view(-1, 28 * 28)
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data, p, epoch=epoch + i)
        loss = criterion(output, target)

        if len(sim_tasks) != 0:
            l2 = contrast_cls(data, every_task_base, sim_tasks,
                              sim_scores, model, task_id, device)
            loss += l2

        loss.backward()
        # Gradient Projections
        kk = 0
        for k, (m, params) in enumerate(model.named_parameters()):
            if k < 3 and len(params.size()) != 1:
                sz = params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1),
                                                               feature_mat[kk]).view(params.size())
                kk += 1
            elif (k < 3 and len(params.size()) == 1) and task_id != 0:
                params.grad.data.fill_(0)

        optimizer.step()


def test(args, model, device, x, y, criterion, id=None):
    model.eval()
    total_loss = 0
    total_num = 0
    correct = 0
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0, len(r), args.batch_size_test):
            if i + args.batch_size_test <= len(r):
                b = r[i:i + args.batch_size_test]
            else:
                b = r[i:]
            data = x[b].view(-1, 28 * 28)
            data, target = data.to(device), y[b].to(device)
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item() * len(b)
            total_num += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc


def get_representation_matrix(task_id, net, device, x, y, old_task_distribution):
    example_data = []
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    un = torch.unique(y)
    idx = 0
    for _ in range(30):
        b = []
        for i in un:
            while y[idx] != i:
                idx += 1
            b.append(idx)
        assert len(b) == 10
        tmp_data = x[b].view(-1, 28*28).to(device)
        target = y[b]
        example_data.append(tmp_data)

    example_data = torch.cat(example_data, dim=0).squeeze(1)
    example_data = example_data.view(-1, 28 * 28)
    net.eval()
    example_out = net(example_data)

    batch_list = [300, 300, 300]
    mat_list = []  # list contains representation matrix of each layer
    act_key = list(net.act.keys())

    for i in range(len(act_key)):
        bsz = batch_list[i]
        act = net.act[act_key[i]].detach().cpu().numpy()
        activation = act[0:bsz].transpose()
        mat_list.append(activation)
        old_task_distribution[task_id][i].append(
            deepcopy(activation.flatten()))

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_list)):
        print('Layer {} : {}'.format(i+1, mat_list[i].shape))
    print('-'*30)
    return mat_list


def update_GPM(task_id, model, mat_list, threshold, feature_list=[], proj=None, every_task_base=None):
    print('Threshold: ', threshold)
    if not feature_list:
        # After First Task
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
            feature_list.append(U[:, 0:r])
            proj[task_id][i] = U[:, 0:r]
            every_task_base[task_id][i] = U[:, 0:r]
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            sval_ratio = (S1**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
            every_task_base[task_id][i] = U1[:, 0:r]

            act_hat = activation - np.dot(
                np.dot(feature_list[i], feature_list[i].transpose()),
                activation)
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total

            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r != 0:
                print('Skip Updating GPM for layer: {}'.format(i + 1))
                Ui = np.hstack((feature_list[i], U[:, 0:r]))
                if Ui.shape[1] > Ui.shape[0]:
                    print('-' * 40)
                    print('Base Matrix has OOM')
                    print('-' * 40)
                    feature_list[i] = Ui[:, 0:Ui.shape[0]]
                else:
                    feature_list[i] = Ui
            if r == 0:
                proj[task_id][i] = proj[task_id-1][i]
            else:
                proj[task_id][i] = U[:, 0:r]

    print('-' * 40)
    print('Gradient Constraints Summary')
    print('-' * 40)
    for i in range(len(feature_list)):
        print('Layer {} : {}/{}'.format(i + 1, feature_list[i].shape[1],
                                        feature_list[i].shape[0]))
    print('-' * 40)
    return feature_list


def main(args):
    tstart = time.time()
    # Device Setting
    device = torch.device("cuda:{}".format(args.cuda)
                          if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Load PMNIST DATASET
    from dataloader import pmnist as pmd
    data, taskcla, inputsize = pmd.get(seed=args.seed, pc_valid=args.pc_valid)

    n_task = 10
    acc_matrix = np.zeros((10, 10))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []

    model = MLPNet(args.n_hidden, args.n_outputs).to(device)
    print('Model parameters ---')
    for k_t, (m, param) in enumerate(model.named_parameters()):
        print(k_t, m, param.shape)
    print('-' * 40)
    pre_task_distribution = [[[] for j in range(3)] for i in range(n_task)]
    old_task_distribution = [[[] for j in range(3)] for i in range(n_task)]

    task_id = 0
    print("*" * 100)
    print("Get Init Distribution.")
    for k, ncla in taskcla:
        xtrain = data[k]['train']['x']
        ytrain = data[k]['train']['y']
        _ = get_representation_matrix(
            model, device, xtrain, ytrain, task_id, pre_task_distribution)
        task_id += 1
    print("*" * 100)
    del model

    proj = {}
    every_task_base = {}
    task_id = 0
    task_list = []

    for k, ncla in taskcla:
        # specify threshold hyperparameter
        threshold = np.array([0.95, 0.99, 0.99])

        print('*' * 100)
        print('Task {:2d} ({:s})'.format(k, data[k]['name']))
        print('*' * 100)
        xtrain = data[k]['train']['x']
        ytrain = data[k]['train']['y']
        xvalid = data[k]['valid']['x']
        yvalid = data[k]['valid']['y']
        xtest = data[k]['test']['x']
        ytest = data[k]['test']['y']
        task_list.append(k)

        lr = args.lr
        best_loss = np.inf
        print('-' * 40)
        print('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print('-' * 40)

        proj[task_id] = {}
        every_task_base[task_id] = {}

        if task_id == 0:
            feature_list = []
            optimizer = optim.SGD(model.parameters(),
                                  lr=lr, momentum=args.momentum)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            for epoch in range(1, args.n_epochs + 1):
                clock0 = time.time()
                train(args, epoch, task_id, model, device,
                      xtrain, ytrain, optimizer, criterion)
                clock1 = time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,
                                       criterion)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,
                                                                                                 tr_loss, tr_acc, 1000*(clock1-clock0)), end='')
                # Validate
                valid_loss, valid_acc = test(args, model, device, xvalid,
                                             yvalid, criterion)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(
                    valid_loss, valid_acc),
                    end='')
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    patience = args.lr_patience
                    print(' *', end='')
                # else:
                #     patience -= 1
                #     if patience <= 0:
                #         lr /= args.lr_factor
                #         print(' lr={:.1e}'.format(lr), end='')
                #         if lr < args.lr_min:
                #             print()
                #             break
                #         patience = args.lr_patience
                #         adjust_learning_rate(optimizer, epoch, args)
                print()
            # Test
            print('-' * 40)
            test_loss, test_acc = test(args, model, device, xtest, ytest,
                                       criterion)
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(
                test_loss, test_acc))

            # Memory Update
            mat_list = get_representation_matrix(
                task_id, model, device, xtrain, ytrain, old_task_distribution)
            feature_list = update_GPM(
                task_id, model, mat_list, threshold, feature_list, proj, every_task_base)

        else:

            sim_tasks = []
            sim_scores = []
            # find top-2 tasks
            if task_id >= 3:
                sim_tasks = [i for i in range(3)]
                sim_scores = [i for i in range(3)]

                _ = get_representation_matrix(
                    task_id, model, device, xtrain, ytrain, old_task_distribution)
                distribution = [[[]
                                 for j in range(task_id)] for i in range(3)]

                cnt = 0
                for kk, (m, params) in enumerate(model.named_parameters()):
                    if len(params.size()) != 1 and kk < 3:
                        for tt in range(task_id):

                            a = wasserstein_distance(
                                old_task_distribution[tt][cnt][0], old_task_distribution[task_id][cnt][0])
                            distribution[cnt][tt] = a
                        cnt += 1
                print(distribution)

                cnt = 0
                for kk, (m, params) in enumerate(model.named_parameters()):
                    if len(params.size()) != 1 and kk < 3:
                        for tt in range(task_id):

                            t = wasserstein_distance(
                                pre_task_distribution[tt][cnt][0], pre_task_distribution[task_id][cnt][0])
                            if t < distribution[cnt][tt]:
                                distribution[cnt][tt] = 1000
                        cnt += 1

                print(distribution)

                for idx, ii in enumerate(distribution):
                    sim_tasks[idx] = sorted(
                        range(len(ii)), key=lambda i: ii[i])[:2]
                print(sim_tasks)


                for idx, ii in enumerate(distribution):
                    if ii.count(ii[0]) == len(ii) and ii.count(ii[0]) == 1000:
                        tmp = [1, 1]
                    else:
                        tmp = sorted(ii)[:2]
                        t = [sum(tmp) - i for i in tmp]
                        tmp = [i/sum(t) for i in t]
                    sim_scores[idx] = tmp
                print(sim_scores)

                print("*" * 40)
                print("Task {} has sim Tasks".format(task_id), end="")
                cnt = 0
                for kk, (m, params) in enumerate(model.named_parameters()):
                    if len(params.size()) != 1 and kk < 4:
                        print("Layer: {}".format(cnt))
                        print(sim_tasks[cnt], sim_scores[cnt])
                        cnt += 1
                print("*" * 40)

            optimizer = optim.SGD(model.parameters(),
                                  lr=args.lr, momentum=args.momentum)
            scheduler = None
            # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            feature_mat = []
            # Projection Matrix Precomputation
            for i in range(len(model.act)):
                Uf = torch.Tensor(
                    np.dot(feature_list[i],
                           feature_list[i].transpose())).to(device)
                print('Layer {} - Projection Matrix shape: {}'.format(
                    i + 1, Uf.shape))
                feature_mat.append(Uf)
            print('-' * 40)

            p = [None, None, None]
            if task_id >= 1:

                for i in range(3):
                    p[i] = torch.Tensor(proj[task_id-1][i]).to(device)

            for epoch in range(1, args.n_epochs + 1):

                clock0 = time.time()

                train_projected(args, p, model, device, xtrain,
                                ytrain, optimizer, criterion, feature_mat, k, epoch, sim_tasks, sim_scores, every_task_base)
                clock1 = time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,
                                       criterion)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,
                                                                                                 tr_loss, tr_acc, 1000*(clock1-clock0)), end='')
                # Validate
                valid_loss, valid_acc = test(args, model, device, xvalid,
                                             yvalid, criterion)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(
                    valid_loss, valid_acc),
                    end='')
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    patience = args.lr_patience
                    print(' *', end='')
                # else:
                #     patience -= 1
                #     if patience <= 0:
                #         lr /= args.lr_factor
                #         print(' lr={:.1e}'.format(lr), end='')
                #         if lr < args.lr_min:
                #             print()
                #             break
                #         patience = args.lr_patience
                #         adjust_learning_rate(optimizer, epoch, args)
                print()

            # Test
            test_loss, test_acc = test(args, model, device, xtest, ytest,
                                       criterion)
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(
                test_loss, test_acc))
            # Memory Update
            mat_list = get_representation_matrix(
                task_id, model, device, xtrain, ytrain, old_task_distribution)
            feature_list = update_GPM(
                task_id, model, mat_list, threshold, feature_list, proj, every_task_base)

        # save accuracy
        jj = 0
        for ii in np.array(task_list)[0:task_id + 1]:
            xtest = data[ii]['test']['x']
            ytest = data[ii]['test']['y']
            _, acc_matrix[task_id, jj] = test(args, model, device, xtest,
                                              ytest, criterion)
            jj += 1
        print('Accuracies =')
        for i_a in range(task_id + 1):
            print('\t', end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a, j_a]), end='')
            print()
        # update task id
        task_id += 1
    print('-' * 50)
    # Simulation Results
    print('Task Order : {}'.format(np.array(task_list)))
    print('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean()))
    bwt = np.mean((acc_matrix[-1] - np.diag(acc_matrix))[:-1])
    print('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time() - tstart) * 1000))
    print('-' * 50)
    # Plots
    array = acc_matrix
    df_cm = pd.DataFrame(
        array,
        index=[
            i for i in
            ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]
        ],
        columns=[
            i for i in
            ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]
        ])
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    plt.show()


if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--cuda', default=0, type=int,
                        help='default GPU device')
    parser.add_argument('--batch_size_train',
                        type=int,
                        default=10,
                        metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--batch_size_test',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs',
                        type=int,
                        default=5,
                        metavar='N',
                        help='number of training epochs/task (default: 5)')
    parser.add_argument('--seed',
                        type=int,
                        default=2,
                        metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--pc_valid',
                        default=0.1,
                        type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.0,
                        metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min',
                        type=float,
                        default=1e-5,
                        metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience',
                        type=int,
                        default=6,
                        metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor',
                        type=int,
                        default=2,
                        metavar='LRF',
                        help='lr decay factor (default: 2)')
    # Architecture
    parser.add_argument('--n_hidden',
                        type=int,
                        default=100,
                        metavar='NH',
                        help='number of hidden units in MLP (default: 100)')
    parser.add_argument('--n_outputs',
                        type=int,
                        default=10,
                        metavar='NO',
                        help='number of output units in MLP (default: 10)')
    parser.add_argument('--n_tasks',
                        type=int,
                        default=10,
                        metavar='NT',
                        help='number of tasks (default: 10)')

    args = parser.parse_args()
    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)

    main(args)
