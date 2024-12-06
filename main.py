
from __future__ import print_function
from tqdm import tqdm
import argparse
import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


from celeba_data import Celebadataset
from utils import accuracy, mkdir_p
from tensorboardX import SummaryWriter
from model import Resnet50_celeba


import statistics

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', default='./Celeba_aligned', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=1500, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=1500, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--learning_rate', default=0.05  , type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--turning-point', type=int, default=100,
                    help='epoch number from linear to exponential decay mode')
parser.add_argument('--gamma', type=float, default=0.6,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-1, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='output', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--groups', type=int, default=3, help='ShuffleNet model groups')
# Miscs
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

# parser.add_argument('--pretrained', dest='pretrained', default = True,
#                     help='use pre-trained model')

best_acc = 0
def main():

    global args, best_acc
    args = parser.parse_args()


    # Use CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Random seed
    if args.manual_seed is None: # args.manual_seed = None
        args.manual_seed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manual_seed)


    # create model
    model = Resnet50_celeba().to(device)
    model = model.to(device)
    # model = torch.nn.DataParallel(model)  # device_ids=device

    # pretrained_dict = torch.load('')
    # model.load_state_dict(pretrained_dict['model_state_dict'])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    all_layers = []
    for name, param in model.named_parameters():
        all_layers.append(name)
    layers_to_update = all_layers[161:]


    params_to_update = []
    for name, param in model.named_parameters():
        if name in layers_to_update:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False

    optimizer = torch.optim.SGD(params_to_update, lr=args.learning_rate)#, momentum=args.momentum, weight_decay=args.weight_decay
    # optimizer.load_state_dict(pretrained_dict['optimizer_state_dict'])


    # optimizer = torch.optim.SGD(
    #     [
    #         {"params": params_to_update[161:], "lr": 1e-2},
    #         {"params": params_to_update[0:161]},
    #     ],
    #     lr=args.learning_rate,
    # )



    lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 10, gamma=args.gamma)


    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    train_dataset= Celebadataset(
        args.data,
        'train_attr_list.txt')




    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers)



    val_loader = torch.utils.data.DataLoader(
        Celebadataset(args.data, 'val_attr_list.txt'),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers)



    writer = SummaryWriter(os.path.join(args.checkpoint, 'logs'))

    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr_sch.get_last_lr()[0]))

        # train for one epoch
        train_loss_, train_acc_ = train(train_loader, model, criterion, optimizer, device, epoch)

        # evaluate on validation set
        train_loss, train_acc_avg, t_moving_accuracy_male, t_moving_accuracy_young  = validate(train_loader, model, criterion, device)
        val_loss, val_acc_avg, v_moving_accuracy_male, v_moving_accuracy_young = validate(val_loader, model, criterion, device)

        print('epoch{} train_loss:{}, train_acc_avg:{}, t_male:{}, t_young:{}, val_loss:{}, val_acc_avg:{}, v_male:{}, v_young:{}'.format(epoch + 1, train_loss, train_acc_avg, t_moving_accuracy_male, t_moving_accuracy_young,  val_loss, val_acc_avg, v_moving_accuracy_male, v_moving_accuracy_young))

        # tensorboardX
        writer.add_scalar('learning rate', lr_sch.get_last_lr()[0], epoch + 1)
        writer.add_scalars('loss', {'train loss': train_loss, 'validation loss': val_loss}, epoch + 1)
        writer.add_scalars('accuracy', {'train accuracy': train_acc_avg, 'validation accuracy': val_acc_avg}, epoch + 1)


        if (epoch + 1) % 1 == 0:
            if val_acc_avg > best_acc:
                best_acc = val_acc_avg
                tmp = 'best_model'
            else:
                tmp = 'tmpp'

            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'Val_loss': val_loss,
                        'train_acc':train_acc_avg,
                        'val_acc':val_acc_avg}, os.path.join('./',args.checkpoint, 'weights_%s_epoch%d.pth' % (tmp,epoch)))

        lr_sch.step()

    writer.close()



def Average(lst):
    return sum(lst) / len(lst)

def train(train_loader, model, criterion, optimizer, device, epoch):



    # switch to train mode
    model.train()

    end = time.time()
    running_loss = 0
    moving_accuracy = 0
    for i, (input, target,_) in enumerate(tqdm(train_loader)):
        input = input.to(device)
        target = target.to(device)

        # compute output
        output = model(input)
        # measure accuracy and record loss
        loss = []
        prec1 = []
        for j in range(len(output)):

            loss.append(criterion(output[j], target[:, j]))
            prec1.append(accuracy(output[j], target[:, j]))

        loss_Avg = Average(loss)
        loss_sum = sum(loss)
        prec1_Avg = Average(torch.FloatTensor(prec1))
        running_loss = running_loss + loss_Avg
        moving_accuracy = moving_accuracy + prec1_Avg
        print('loss_avg:{}, acc_avg:{}, male:{}, young:{}'.format(loss_Avg, prec1_Avg.item(), prec1[0], prec1[1]))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()


    running_loss_avg = running_loss / len(train_loader)
    moving_accuracy = moving_accuracy / len(train_loader)

    return (running_loss_avg, moving_accuracy)


def validate(val_loader, model, criterion, device):

    running_loss = 0
    moving_accuracy_avg = 0
    moving_accuracy_male = 0
    moving_accuracy_young = 0

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target,_) in enumerate(tqdm(val_loader)):
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            # measure accuracy and record loss
            loss = []
            prec1 = []
            for j in range(len(output)):
                loss.append(criterion(output[j], target[:, j]))
                prec1.append(accuracy(output[j], target[:, j]))

            loss_Avg = Average(loss)
            prec1_Avg = Average(torch.FloatTensor(prec1))
            running_loss = running_loss + loss_Avg
            moving_accuracy_avg = moving_accuracy_avg + prec1_Avg
            moving_accuracy_male = moving_accuracy_male + prec1[0][0]
            moving_accuracy_young = moving_accuracy_young + prec1[1][0]

            print('loss_avg:{}, acc_avg:{}, male:{}, young:{}'.format(loss_Avg, prec1_Avg, prec1[0], prec1[1]))


        running_loss_avg = running_loss / len(val_loader)
        moving_accuracy_avg = moving_accuracy_avg / len(val_loader)

        moving_accuracy_male = moving_accuracy_male / len(val_loader)
        moving_accuracy_young = moving_accuracy_young / len(val_loader)


    return (running_loss_avg, moving_accuracy_avg, moving_accuracy_male, moving_accuracy_young)




if __name__ == '__main__':
    main()

