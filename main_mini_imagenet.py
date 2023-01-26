import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random
import pdb
import argparse
import time
import math
from copy import deepcopy
from scipy.stats import wasserstein_distance


class Sequential(nn.Sequential):

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input, t, p, epoch):
        for module in self:
            input = module(input, t, p, epoch)
        return input


class Shortcut(nn.Module):
    def __init__(self, stride, in_planes, expansion, planes):
        super(Shortcut, self).__init__()
        self.identity = True
        self.shortcut = Sequential()

        if stride != 1 or in_planes != expansion*planes:
            self.identity = False
            self.conv1 = nn.Conv2d(
                in_planes, expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.bn1 = nn.ModuleList()
            for _ in range(20):
                self.bn1.append(nn.BatchNorm2d(
                    expansion*planes))

    def forward(self, x, t, p, epoch):
        if self.identity:
            out = self.shortcut(x, t, p, epoch)
        else:
            out = self.conv1(x)
            out = self.bn1[t](out)
        return out


class Conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(Conv2d, self).__init__(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=bias)

    def forward(self, input, task_id, p, epoch):
        if p is not None:
            if epoch == 1:
                sz = self.weight.grad.data.size(0)
                norm_project = torch.mm(p, p.transpose(1, 0))
                proj_weight = torch.mm(self.weight.view(sz, -1),
                                       norm_project).view(self.weight.size())
                masked_weight = self.weight - proj_weight
            else:
                masked_weight = self.weight
        else:
            masked_weight = self.weight
        return F.conv2d(input, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.ModuleList()
        for i in range(20):
            self.bn1.append(nn.BatchNorm2d(planes))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.ModuleList()
        for i in range(20):
            self.bn2.append(nn.BatchNorm2d(planes))

        self.shortcut = Sequential()
        self.shortcut = Shortcut(
            stride=stride, in_planes=in_planes, expansion=self.expansion, planes=planes)
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x, t, p, epoch):
        if p is None:
            self.count = self.count % 2
            self.act['conv_{}'.format(self.count)] = x
            self.count += 1
            out = relu(self.bn1[t](self.conv1(x, t, None, epoch)))
            self.count = self.count % 2
            self.act['conv_{}'.format(self.count)] = out
            self.count += 1
            out = self.bn2[t](self.conv2(out, t, None, epoch))
            out += self.shortcut(x, t, None, epoch)
            out = relu(out)
        else:
            self.count = self.count % 2
            self.act['conv_{}'.format(self.count)] = x
            self.count += 1
            out = relu(self.bn1[t](self.conv1(x, t, p[0], epoch)))
            self.count = self.count % 2
            self.act['conv_{}'.format(self.count)] = out
            self.count += 1
            out = self.bn2[t](self.conv2(out, t, p[1], epoch))
            out += self.shortcut(x, t, None, epoch)
            out = relu(out)
        return out


# Define ResNet18 model
def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))


def conv3x3(in_planes, out_planes, stride=1):
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    return Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                  padding=1, bias=False)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf):
        super(ResNet, self).__init__()
        self.taskcla = taskcla
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 2)
        self.bn1 = nn.ModuleList()
        for t, n in self.taskcla:
            self.bn1.append(nn.BatchNorm2d(nf * 1))
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.linear = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.linear.append(
                nn.Linear(nf * 8 * block.expansion * 9, n, bias=False))
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential(*layers)

    def forward(self, x, t, p, epoch):
        if p is None:
            bsz = x.size(0)
            self.act['conv_in'] = x.view(bsz, 3, 84, 84)
            out = relu(self.bn1[t](self.conv1(
                x.view(bsz, 3, 84, 84), t, None, epoch)))
            out = self.layer1(out, t, None, epoch)
            out = self.layer2(out, t, None, epoch)
            out = self.layer3(out, t, None, epoch)
            out = self.layer4(out, t, None, epoch)
            out = avg_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            y = []
            for t, i in self.taskcla:
                y.append(self.linear[t](out))
        else:
            bsz = x.size(0)
            self.act['conv_in'] = x.view(bsz, 3, 84, 84)
            out = relu(
                self.bn1[t](self.conv1(x.view(bsz, 3, 84, 84), t, p[0], epoch)))
            out = self.layer1[0](out, t, p[1:3], epoch)
            out = self.layer1[1](out, t, p[3:5], epoch)
            out = self.layer2[0](out, t, p[5:8], epoch)
            out = self.layer2[1](out, t, p[8:10], epoch)
            out = self.layer3[0](out, t, p[10:13], epoch)
            out = self.layer3[1](out, t, p[13:15], epoch)
            out = self.layer4[0](out, t, p[15:18], epoch)
            out = self.layer4[1](out, t, p[18:20], epoch)

            out = avg_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            y = []
            for t, i in self.taskcla:
                y.append(self.linear[t](out))
        return y


def ResNet18(taskcla, nf=32):
    return ResNet(BasicBlock, [2, 2, 2, 2], taskcla, nf)


def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch == 1):
            param_group['lr'] = args.lr
        else:
            param_group['lr'] /= args.lr_factor


def train(args, model, device, x, y, optimizer, criterion, task_id):
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
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data, task_id, None, -1)
        loss = criterion(output[task_id], target)
        loss.backward()
        optimizer.step()


def contrast_cls(every_task_base, sim_tasks, sim_scores, model, task_id, device, criterion):
    l2 = 0
    cnt = 0
    stride_list = [1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]

    ttt = 0
    for k, (m, params) in enumerate(model.named_parameters()):

        if "short" in m and "conv" in m:
            ttt += 1
            cnt += 1
            continue

        if "conv" in m and len(params.size()) == 4:
            sim = []
            # x = act_list[cnt-ttt]
            # pre = F.conv2d(x, params, None, stride_list[cnt-ttt], 2, 1, 1)

            for tt in sim_tasks[cnt-ttt]:
                sz = params.grad.data.size(0)
                tmp = torch.FloatTensor(every_task_base[tt][cnt]).to(device)
                norm_project = torch.mm(tmp, tmp.transpose(1, 0))
                proj_weight = torch.mm(params.view(sz, -1),
                                       norm_project).view(params.size())
                sim.append(proj_weight.data)

                dis = list(set(range(task_id)) - set(sim_tasks[cnt-ttt]))
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
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data, task_id, p, epoch + i)
        loss = criterion(output[task_id], target)

        if len(sim_tasks) != 0:
            l2 = contrast_cls(every_task_base, sim_tasks,
                              sim_scores, model, task_id, device, criterion)
            loss += l2

        loss.backward()
        # Gradient Projections
        kk = 0
        for k, (m, params) in enumerate(model.named_parameters()):
            if len(params.size()) == 4:
                sz = params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1),
                                                               feature_mat[kk]).view(params.size())
                kk += 1

        optimizer.step()


def test(args, model, device, x, y, criterion, task_id,):
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
            if i+args.batch_size_test <= len(r):
                b = r[i:i+args.batch_size_test]
            else:
                b = r[i:]
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data, task_id, None, -1)
            loss = criterion(output[task_id], target)
            pred = output[task_id].argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc


def get_representation_matrix(task_id, net, device, x, y, old_task_distribution):
    # Collect activations by forward pass
    net.eval()
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    b = r[0:100]  # ns=100 examples
    example_data = x[b]
    example_data = example_data.to(device)
    example_out = net(example_data, task_id, None, -1)

    act_list = []
    act_list.extend([net.act['conv_in'],
                     net.layer1[0].act['conv_0'], net.layer1[0].act['conv_1'], net.layer1[1].act['conv_0'], net.layer1[1].act['conv_1'],
                     net.layer2[0].act['conv_0'], net.layer2[0].act['conv_1'], net.layer2[1].act['conv_0'], net.layer2[1].act['conv_1'],
                     net.layer3[0].act['conv_0'], net.layer3[0].act['conv_1'], net.layer3[1].act['conv_0'], net.layer3[1].act['conv_1'],
                     net.layer4[0].act['conv_0'], net.layer4[0].act['conv_1'], net.layer4[1].act['conv_0'], net.layer4[1].act['conv_1']])

    batch_list = [10, 10, 10, 10, 10, 10, 10, 10, 50,
                  50, 50, 100, 100, 100, 100, 100, 100]  # scaled
    # network arch
    stride_list = [2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]
    map_list = [84, 42, 42, 42, 42, 42, 21,
                21, 21, 21, 11, 11, 11, 11, 6, 6, 6]
    in_channel = [3, 20, 20, 20, 20, 20, 40, 40,
                  40, 40, 80, 80, 80, 80, 160, 160, 160]

    pad = 1
    sc_list = [5, 9, 13]
    p1d = (1, 1, 1, 1)
    mat_final = []  # list containing GPM Matrices
    mat_list = []
    mat_sc_list = []
    for i in range(len(stride_list)):
        if i == 0:
            ksz = 3
        else:
            ksz = 3
        bsz = batch_list[i]
        st = stride_list[i]
        k = 0
        s = compute_conv_output_size(map_list[i], ksz, stride_list[i], pad)
        mat = np.zeros((ksz*ksz*in_channel[i], s*s*bsz))
        act = F.pad(act_list[i], p1d, "constant", 0).detach().cpu().numpy()
        for kk in range(bsz):
            for ii in range(s):
                for jj in range(s):
                    mat[:, k] = act[kk, :, st*ii:ksz+st *
                                    ii, st*jj:ksz+st*jj].reshape(-1)
                    k += 1
        mat_list.append(mat)
        # For Shortcut Connection
        if i in sc_list:
            k = 0
            s = compute_conv_output_size(map_list[i], 1, stride_list[i])
            mat = np.zeros((1*1*in_channel[i], s*s*bsz))
            act = act_list[i].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:, k] = act[kk, :, st*ii:1+st *
                                        ii, st*jj:1+st*jj].reshape(-1)
                        k += 1
            mat_sc_list.append(mat)

    ik = 0
    for i in range(len(mat_list)):
        mat_final.append(mat_list[i])
        if i in [6, 10, 14]:
            mat_final.append(mat_sc_list[ik])
            ik += 1

    for i in range(len(mat_final)):
        old_task_distribution[task_id][i].append(
            deepcopy(mat_final[i].flatten()))

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_final)):
        print('Layer {} : {}'.format(i+1, mat_final[i].shape))
    print('-'*30)
    return mat_final


def update_GPM(task_id, model, mat_list, threshold, feature_list=[], proj=None, every_task_base=None):
    print('Threshold: ', threshold)
    if not feature_list:
        # After First Task
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
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

            # Projected Representation (Eq-8)
            act_hat = activation - \
                np.dot(
                    np.dot(feature_list[i], feature_list[i].transpose()), activation)
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            accumulated_sval = (sval_total-sval_hat)/sval_total

            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r != 0:
                print('Not Skip Updating GPM for layer: {}'.format(i+1))
                # update GPM
                Ui = np.hstack((feature_list[i], U[:, 0:r]))
                if Ui.shape[1] > Ui.shape[0]:
                    feature_list[i] = Ui[:, 0:Ui.shape[0]]
                else:
                    feature_list[i] = Ui
            if r == 0:
                proj[task_id][i] = proj[task_id-1][i]
            else:
                proj[task_id][i] = U[:, 0:r]

    print('-'*40)
    print('Gradient Constraints Summary')
    print('-'*40)
    for i in range(len(feature_list)):
        print('Layer {} : {}/{}'.format(i+1,
              feature_list[i].shape[1], feature_list[i].shape[0]))
    print('-'*40)
    return feature_list


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == Conv2d:
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.kaiming_uniform_(
            m.weight, mode='fan_in', nonlinearity='relu')
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def main(args):
    tstart = time.time()
    # Device Setting
    device = torch.device("cuda:{}".format(args.cuda)
                          if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load CIFAR100 DATASET
    from dataloader import miniimagenet as data_loader
    dataloader = data_loader.DatasetGen(args)
    taskcla, inputsize = dataloader.taskcla, dataloader.inputsize

    n_task = 20
    acc_matrix = np.zeros((20, 20))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0)

    task_id = 0
    task_list = []

    pre_task_distribution = [[[] for j in range(20)] for i in range(n_task)]
    old_task_distribution = [[[] for j in range(20)] for i in range(n_task)]

    task_id = 0
    model = ResNet18(taskcla, 20).to(device)  # base filters: 20
    model.apply(init_weights)
    print("*" * 100)
    print("Get Init Distribution.")
    for k, ncla in taskcla:
        data = dataloader.get(k)
        x = data[k]['train']['x']
        y = data[k]['train']['y']
        _ = get_representation_matrix(
            task_id, model, device, x, y, pre_task_distribution)
        task_id += 1
    print("*" * 100)
    del model

    proj = {}
    every_task_base = {}
    task_id = 0
    task_list = []

    for k, ncla in taskcla:
        # specify threshold hyperparameter
        data = dataloader.get(k)
        threshold = np.array([0.985] * 20)
        print('*'*100)
        print('Task {:2d} ({:s})'.format(k, data[k]['name']))
        print('*'*100)
        xtrain = data[k]['train']['x']
        ytrain = data[k]['train']['y']
        xvalid = data[k]['valid']['x']
        yvalid = data[k]['valid']['y']
        xtest = data[k]['test']['x']
        ytest = data[k]['test']['y']
        task_list.append(k)

        lr = args.lr
        best_loss = np.inf
        print('-'*40, log)
        print('Task ID :{} | Learning Rate : {}'.format(task_id, lr), log)
        print('-'*40, log)

        proj[task_id] = {}
        every_task_base[task_id] = {}

        if task_id == 0:
            model = ResNet18(taskcla, 20).to(device)  # base filters: 20
            model.apply(init_weights)
            best_model = get_model(model)
            feature_list = []
            optimizer = optim.SGD(model.parameters(),
                                  lr=lr, momentum=args.momentum)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.n_epochs)

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0 = time.time()
                train(args, model, device, xtrain,
                      ytrain, optimizer, criterion, k)
                clock1 = time.time()
                tr_loss, tr_acc = test(
                    args, model, device, xtrain, ytrain,  criterion, k)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,
                                                                                                 tr_loss, tr_acc, 1000*(clock1-clock0)), end='')
                # Validate
                valid_loss, valid_acc = test(
                    args, model, device, xvalid, yvalid,  criterion, k)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(
                    valid_loss, valid_acc), end='')
                # Adapt lr
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = get_model(model)
                    patience = args.lr_patience
                    print(' *', end='')
                # scheduler.step()
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= args.lr_factor
                        print(' lr={:.1e}'.format(lr), end='')
                        if lr < args.lr_min:
                            print()
                            break
                        patience = args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
                print()

            # Test
            print('-'*40)
            test_loss, test_acc = test(
                args, model, device, xtest, ytest,  criterion, k)
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
                sim_tasks = [i for i in range(20)]
                sim_scores = [i for i in range(20)]

                _ = get_representation_matrix(
                    task_id, model, device, xtrain, ytrain, old_task_distribution)
                distribution = [[[]
                                 for j in range(task_id)] for i in range(20)]

                cnt = 0
                for kk, (m, params) in enumerate(model.named_parameters()):
                    if 'weight' in m and len(params.size()) == 4:
                        for tt in range(task_id):

                            distribution[cnt][tt] = wasserstein_distance(
                                old_task_distribution[tt][cnt][0], old_task_distribution[task_id][cnt][0])
                        cnt += 1

                cnt = 0
                for kk, (m, params) in enumerate(model.named_parameters()):
                    if 'weight' in m and len(params.size()) == 4:
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
                    if ii.count(ii[0]) == len(ii) and ii[0] == 1000:
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
                    if 'weight' in m and len(params.size()) == 4:
                        print("Layer: {}".format(cnt))
                        print(sim_tasks[cnt], sim_scores[cnt])
                        cnt += 1
                print("*" * 40)

            optimizer = optim.SGD(model.parameters(),
                                  lr=lr, momentum=args.momentum)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.n_epochs)
            feature_mat = []
            # Projection Matrix Precomputation
            for i in range(len(feature_list)):
                Uf = torch.Tensor(
                    np.dot(feature_list[i], feature_list[i].transpose())).to(device)
                print('Layer {} - Projection Matrix shape: {}'.format(i+1, Uf.shape))
                feature_mat.append(Uf)
            print('-'*40)

            p = [None for i in range(20)]
            if task_id >= 1:

                for i in range(20):
                    p[i] = torch.FloatTensor(proj[task_id-1][i]).to(device)

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0 = time.time()
                train_projected(args, p, model, device, xtrain,
                                ytrain, optimizer, criterion, feature_mat, k, epoch, sim_tasks, sim_scores, every_task_base)
                clock1 = time.time()
                tr_loss, tr_acc = test(
                    args, model, device, xtrain, ytrain, criterion, k)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,
                                                                                                 tr_loss, tr_acc, 1000*(clock1-clock0)), end='')
                # Validate
                valid_loss, valid_acc = test(
                    args, model, device, xvalid, yvalid, criterion, k)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(
                    valid_loss, valid_acc), end='')
                # Adapt lr
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = get_model(model)
                    patience = args.lr_patience
                    print(' *', end='')
                # scheduler.step()
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= args.lr_factor
                        print(' lr={:.1e}'.format(lr), end='')
                        if lr < args.lr_min:
                            print()
                            break
                        patience = args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
                print()

            # Test
            test_loss, test_acc = test(
                args, model, device, xtest, ytest,  criterion, k)
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(
                test_loss, test_acc))
            # Memory Update
            mat_list = get_representation_matrix(
                task_id, model, device, xtrain, ytrain, old_task_distribution)
            feature_list = update_GPM(
                task_id, model, mat_list, threshold, feature_list, proj, every_task_base)

        # save accuracy
        jj = 0
        for ii in np.array(task_list)[0:task_id+1]:
            xtest = data[ii]['test']['x']
            ytest = data[ii]['test']['y']
            _, acc_matrix[task_id, jj] = test(
                args, model, device, xtest, ytest, criterion, ii)
            jj += 1
        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t', end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a, j_a]), end='')
            print()
        # update task id
        task_id += 1

    print('-'*50, log)
    # Simulation Results
    print('Task Order : {}'.format(np.array(task_list)), log)
    print('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean()), log)
    bwt = np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1])
    print('Backward transfer: {:5.2f}%'.format(bwt), log)
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000), log)
    print('-'*50, log)
    # Plots
    array = acc_matrix
    df_cm = pd.DataFrame(array, index=[i for i in ["1", "2", "3", "4", "5", "6", "7",
                                                   "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]],
                         columns=[i for i in ["1", "2", "3", "4", "5", "6", "7",
                                              "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]])
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    plt.show()


if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='5 datasets with GPM')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=100, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=37, metavar='S',
                        help='random seed (default: 37)')
    parser.add_argument('--pc_valid', default=0.02, type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--cuda', default=0, type=int,
                        help='default GPU device')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=5, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=3, metavar='LRF',
                        help='lr decay factor (default: 2)')
    parser.add_argument('--savename', type=str, default='./model/',
                        help='save path')

    args = parser.parse_args()
    if not os.path.exists(args.savename):
        os.makedirs(args.savename)
    log = open(os.path.join(args.savename,
                            'log_seed_{}.txt'.format(args.seed)), 'w')
    print('='*100, log)
    print('Arguments =', log)
    for arg in vars(args):
        print('\t'+arg+': {}'.format(getattr(args, arg)), log)
    print('='*100, log)

    print('save path : {}'.format(args.savename), log)
    main(args)
