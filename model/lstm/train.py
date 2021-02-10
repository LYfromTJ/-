import argparse
import configparser
import os
import time

import torch
import torch.nn as nn
import wandb

import dataset
from model import LSTM
from utils import save_checkpoint

wandb.init(project="TST")

parser = argparse.ArgumentParser(description='TST2020')

parser.add_argument('--config', '-c', metavar='CONFIG', type=str,
                    help='path to confg file')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')

parser.add_argument('--batch_size', '-bs', metavar='BATCHSIZE', type=int,
                    help='batch size', default=4)

parser.add_argument('--gpu', metavar='GPU', type=str,
                    help='GPU id to use.', default="0")

parser.add_argument('--task', metavar='TASKID', type=str, default=wandb.run.name,
                    help='Task id of this run.')


def create_dir_not_exist(path):
    for length in range(1, len(path.split(os.path.sep))):
        check_path = os.path.sep.join(path.split(os.path.sep)[:(length + 1)])
        if not os.path.exists(check_path):
            os.mkdir(check_path)
            print(f'Created Dir: {check_path}')


class FullModel(nn.Module):
    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, targets, *inputs):
        outputs = self.model(*inputs)
        # print(outputs.shape)
        loss = self.loss(outputs, targets)
        return torch.unsqueeze(loss, 0), outputs


def DataParallel_withLoss(model, loss, **kwargs):
    model = FullModel(model, loss)
    if 'device_ids' in kwargs.keys():
        device_ids = kwargs['device_ids']
    else:
        device_ids = None
    if 'output_device' in kwargs.keys():
        output_device = kwargs['output_device']
    else:
        output_device = None
    if 'cuda' in kwargs.keys():
        cudaID = kwargs['cuda']
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda(cudaID)
    else:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda()
    return model


def adjust_learning_rate(optimizer, epoch):
    args.lr = args.original_lr
    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    wandb.log({'Learning Rate': args.lr})


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    global args, best_prec1
    best_prec1 = 1e6
    args = parser.parse_args()
    args.original_lr = 1e-6
    args.lr = 1e-6
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 5000
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30
    args.feature_size = 100
    args.lSeq=5
    wandb.config.update(args)
    wandb.run.name = f"Default_{wandb.run.name}" if (args.task == wandb.run.name) else f"{args.task}_{wandb.run.name}"

    conf = configparser.ConfigParser()
    conf.read(args.config)
    # print(conf)
    TRAIN_DIR = conf.get("lstm", "train")
    VALID_DIR = conf.get("lstm", "valid")
    TEST_DIR = conf.get("lstm", "test")
    LOG_DIR = conf.get("lstm", "log")
    create_dir_not_exist(LOG_DIR)
    # TODO: train_list to train_file
    train_list = [os.path.join(TRAIN_DIR, item) for item in os.listdir(TRAIN_DIR)]
    val_list = [os.path.join(VALID_DIR, item) for item in os.listdir(VALID_DIR)]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(int(args.seed))
    model = LSTM(args.feature_size, args.feature_size, args.feature_size)
    model = model.cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay)
    model = DataParallel_withLoss(model, criterion)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_list, model, criterion, optimizer, epoch)
        prec1 = validate(val_list, model, criterion, epoch)
        with open(os.path.join(LOG_DIR, args.task + ".txt"), "a") as f:
            f.write("epoch " + str(epoch) + "  MSELoss: " + str(float(prec1)))
            f.write("\n")
        wandb.save(os.path.join(LOG_DIR, args.task + ".txt"))
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MSELoss {MSELoss:.3f} '.format(MSELoss=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.task, epoch=epoch, path=os.path.join(LOG_DIR, args.task))


def train(train_list, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loader = torch.utils.data.DataLoader(dataset.fileDataset(train_list, args.lSeq), num_workers=args.workers,
                                               batch_size=args.batch_size, shuffle=True)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    model.train()
    end = time.time()
    for i, (seq, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        seq = seq.cuda()
        seq = seq.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor).cuda()
        # print(img.shape)
        # print(target.shape)
        loss, _ = model(target, seq)
        loss = loss.sum()
        losses.update(loss.item(), seq.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {lossval:.4f} ({lossavg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, lossval=losses.val / args.batch_size, lossavg=losses.avg / args.batch_size))
    wandb.log({'Train Loss': losses.avg / args.batch_size})


def validate(val_list, model, criterion, epoch):
    print('begin test')
    test_loader = torch.utils.data.DataLoader(dataset.fileDataset(val_list, args.lSeq),
                                              batch_size=args.batch_size, shuffle=False)
    model.eval()
    MSELoss = 0
    with torch.no_grad():
        for i, (seq, target) in enumerate(test_loader):
            seq = seq.cuda()
            seq = seq.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor).cuda()
            _, output = model(target, seq)
            MSELoss += criterion(output.data, target)
    MSELoss /= len(test_loader)
    print(' * MSELoss {MSELoss:.3f} '.format(MSELoss=MSELoss))
    wandb.log({'epoch': epoch, 'Val Loss': MSELoss})
    return MSELoss
