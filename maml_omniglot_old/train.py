import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import copy
from db import OmniglotNShot

# Runs off of the zou3519/pytorch:dynlayer branch.
# make_functional_with_buffers is purely python logic so we could lift it out.
from torch.eager_transforms import make_functional_with_buffers

r"""
Example taken from https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py
Citation is as follows

@misc{MAML_Pytorch,
  author = {Liangqu Long},
  title = {MAML-Pytorch Implementation},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dragen1860/MAML-Pytorch}},
  commit = {master}
}
"""

class CNN(nn.Module):
    def __init__(self, n_way):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 2, 0)
        self.relu1 = nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 0)
        self.relu2 = nn.ReLU(True)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 2, 0)
        self.relu3 = nn.ReLU(True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 2, 1, 0)
        self.relu4 = nn.ReLU(True)
        self.bn4 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, n_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)

        x = x.flatten(1)
        x = self.fc(x)
        return x


def run_task(update_lr, update_step, params, buffers, net,
             x_spt, y_spt, x_qry, y_qry):
    losses_q = [0 for _ in range(update_step + 1)]
    corrects = [0 for _ in range(update_step + 1)]

    # Compute loss and accuracy before first update for debugging
    with torch.no_grad():
        logits_q = net(params, buffers, (x_qry,))
        loss_q = F.cross_entropy(logits_q, y_qry)
        losses_q[0] += loss_q

        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y_qry).sum().item()
        corrects[0] = corrects[0] + correct

    new_params = params
    for k in range(0, update_step):
        # Train the model w.r.t spt input
        logits = net(new_params, buffers, (x_spt,))
        loss = F.cross_entropy(logits, y_spt)
        grads = torch.autograd.grad(loss, new_params, create_graph=True)
        new_params = [param - update_lr * param_grad
                      for param_grad, param in zip(grads, new_params)]

        # Compute loss w.r.t qry
        logits_q = net(new_params, buffers, (x_qry,))
        loss_q = F.cross_entropy(logits_q, y_qry)
        losses_q[k + 1] += loss_q

        # Compute accuracy
        with torch.no_grad():
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[k + 1] = corrects[k + 1] + correct

    return losses_q, corrects


def compute_maml_loss(args, params, buffers, net, meta_optim,
                      x_spt, y_spt, x_qry, y_qry):
    task_num = args.task_num
    querysz = x_qry.size(1)

    # losses_q[i] is the loss on step i (1 indexed)
    losses_q = [0 for _ in range(args.update_step + 1)]
    corrects = [0 for _ in range(args.update_step + 1)]

    for i in range(task_num):
        loss_q, correct = run_task(args.update_lr, args.update_step,
                                   params, buffers, net,
                                   x_spt[i], y_spt[i], x_qry[i], y_qry[i])
        for j in range(args.update_step + 1):
            losses_q[j] += loss_q[j]
            corrects[j] += correct[j]

    # sum over all losses on query set across all tasks
    loss_q = losses_q[-1] / task_num

    # optimize theta parameters
    meta_optim.zero_grad()
    loss_q.backward()
    meta_optim.step()

    accs = np.array(corrects) / (querysz * task_num)
    return accs, loss_q.item()


def finetune(args, params, buffers, net, x_spt, y_spt, x_qry, y_qry):
    # deepcopy the params and buffers to avoid updating BN buffers
    params = [nn.Parameter(p.detach()) for p in params]
    buffers = copy.deepcopy(buffers)
    querysz = x_qry.size(0)

    loss_q, corrects = run_task(args.update_lr, args.update_step_test,
                                params, buffers, net,
                                x_spt, y_spt, x_qry, y_qry)
    accs = np.array(corrects) / querysz
    return loss_q[-1], accs


def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    device = 'cuda'

    model = CNN(args.n_way).to(device)

    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(model)
    print('Total trainable tensors:', num)
    params, buffers, net, _, _ = make_functional_with_buffers(model)
    meta_optim = optim.Adam(params, lr=args.meta_lr)

    db_train = OmniglotNShot('omniglot',
                             batchsz=args.task_num,
                             n_way=args.n_way,
                             k_shot=args.k_spt,
                             k_query=args.k_qry,
                             imgsz=args.imgsz)

    for epoch in range(args.epoch):
        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = tuple(torch.from_numpy(arr).to(device)
                                           for arr in [x_spt, y_spt, x_qry, y_qry])

        accs, loss = compute_maml_loss(args, params, buffers, net, meta_optim,
                                       x_spt, y_spt, x_qry, y_qry)
        if epoch % 50 == 0:
            print('step:', epoch, '\ttraining acc:', accs, '\tloss:', loss)

        if epoch % 500 == 0:
            accs = []
            for _ in range(1000//args.task_num):
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = tuple(torch.from_numpy(arr).to(device)
                                                   for arr in [x_spt, y_spt, x_qry, y_qry])
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    _, test_acc = finetune(args, params, buffers, net,
                                           x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append(test_acc)

            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main(args)
