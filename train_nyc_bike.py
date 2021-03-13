import os
import sys
import argparse

import h5py
import numpy as np
from sklearn import metrics
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
sys.path.append('../../')
from models.STPWNet import PWNet

import random

parse = argparse.ArgumentParser()
parse.add_argument('-close_size', type=int, default=3)
parse.add_argument('-period_size', type=int, default=0)
parse.add_argument('-trend_size', type=int, default=0)
parse.add_argument('-test_rate', type=float, default=0.2)

parse.set_defaults(crop=False)
parse.add_argument('-train', dest='train', action='store_true')
parse.add_argument('-no-train', dest='train', action='store_false')
parse.set_defaults(train=True)
parse.add_argument('-loss', type=str, default='l2', help='l1 | l2')
parse.add_argument('-lr', type=float, default=0.001)
parse.add_argument('-batch_size', type=int, default=32, help='batch size')
parse.add_argument('-epoch', type=int, default=100, help='epochs')

parse.add_argument('-save_dir', type=str, default='results')

opt = parse.parse_args()

def train_epoch():
    total_loss = 0
    model.train()
    data = train_loader

    if (opt.period_size > 0) & (opt.close_size > 0) & (opt.trend_size > 0):
        for idx, (c, p, t, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            input_var = [Variable(_.float()).cuda() for _ in [c, p, t]]
            target_var = Variable(target.float(), requires_grad=False).cuda()
            pred = model(input_var)
            loss = criterion(pred, target_var)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    elif (opt.close_size > 0) & (opt.period_size > 0):
        for idx, (c, p, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            input_var = [Variable(_.float()).cuda() for _ in [c, p]]
            target_var = Variable(target.float(), requires_grad=False).cuda()
            pred = model(input_var)
            loss = criterion(pred, target_var)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    elif opt.close_size > 0:
        for idx, (c, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            x = [Variable(c.float()).cuda()]
            y = Variable(target.float(), requires_grad=False).cuda()
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

    return total_loss


def valid_epoch():
    total_loss = 0
    model.eval()
    data = valid_loader
    if (opt.period_size > 0) & (opt.close_size > 0) & (opt.trend_size > 0):
        for idx, (c, p, t, target) in enumerate(data):
            input_var = [Variable(_.float()).cuda() for _ in [c, p, t]]
            target_var = Variable(target.float(), requires_grad=False).cuda()
            pred = model(input_var)
            loss = criterion(pred, target_var)
            total_loss += loss.item()
    elif (opt.close_size > 0) & (opt.period_size > 0):
        for idx, (c, p, target) in enumerate(data):
            input_var = [Variable(_.float()).cuda() for _ in [c, p]]
            target_var = Variable(target.float(), requires_grad=False).cuda()
            pred = model(input_var)
            loss = criterion(pred, target_var)
            total_loss += loss.item()

    elif opt.close_size > 0:
        for idx, (c, target) in enumerate(data):
            x = [Variable(c.float()).cuda()]
            y = Variable(target.float(), requires_grad=False).cuda()
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()

    return total_loss

def train():
    best_valid_loss = 1.0
    train_loss, valid_loss = [], []
    for i in range(opt.epoch):
        print('epoch ',i)
        train_loss.append(train_epoch())
        valid_loss.append(valid_epoch())

        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]

            torch.save({'epoch': i, 'model': model, 'train_loss': train_loss,
                        'valid_loss': valid_loss}, '.model')
            torch.save(optimizer, '.optim')

    print('train and val loss =', train_loss[-1],valid_loss[-1])


def predict(test_type='test'):
    predictions = []
    ground_truth = []
    loss = []
    best_model = torch.load('.model').get('model')

    if test_type == 'train':
        data = train_loader
    elif test_type == 'test':
        data = test_loader
    elif test_type == 'valid':
        data = valid_loader

    if (opt.period_size > 0) & (opt.close_size > 0) & (opt.trend_size > 0):
        for idx, (c, p, t, target) in enumerate(data):
            input_var = [Variable(_.float()).cuda() for _ in [c, p, t]]
            target_var = Variable(target.float(), requires_grad=False).cuda()
            pred = best_model(input_var)
            predictions.append(pred.data.cpu().numpy())
            ground_truth.append(target.numpy())
            loss.append(criterion(pred, target_var).item())
    elif (opt.close_size > 0) & (opt.period_size > 0):
        print('--> test: close size & period size',opt.close_size,opt.period_size)
        for idx, (c, p, target) in enumerate(data):
            input_var = [Variable(_.float()).cuda() for _ in [c, p]]
            target_var = Variable(target.float(), requires_grad=False).cuda()
            pred = best_model(input_var)
            predictions.append(pred.data.cpu().numpy())
            ground_truth.append(target.numpy())
            loss.append(criterion(pred, target_var).item())
    elif opt.close_size > 0:
        for idx, (c, target) in enumerate(data):
            input_var = [Variable(c.float()).cuda()]
            target_var = Variable(target.float(), requires_grad=False).cuda()
            pred = best_model(input_var)
            predictions.append(pred.data.cpu().numpy())
            ground_truth.append(target.numpy())
            loss.append(criterion(pred, target_var).item())

    final_predict = np.concatenate(predictions) * mmn[1]+mmn[0]
    ground_truth = np.concatenate(ground_truth) *  mmn[1]+mmn[0]
    print('final prediction shape:', final_predict.shape, ground_truth.shape)

    np.save('final_predict.npy',final_predict)
    np.save('ground_truth.npy',ground_truth)

    print('final prediction and ground truth shape: {} {}'.format(final_predict.shape, ground_truth.shape))
    print('FINAL RMSE:{:0.2f}'.format(
        metrics.mean_squared_error(ground_truth.ravel(), final_predict.ravel()) ** 0.5))
    print('FINAL MAE:{:0.2f}'.format(
        metrics.mean_absolute_error(ground_truth.ravel(), final_predict.ravel())))
    print('FINAL R2:{:0.2f}'.format(
        metrics.r2_score(ground_truth.ravel(), final_predict.ravel())))
    print('FINAL Variance:{:0.2f}'.format(
        metrics.explained_variance_score(ground_truth.ravel(), final_predict.ravel())))



def train_valid_split(dataloader, test_size=0.2, shuffle=True, random_seed=0):
    length=len(dataloader)
    indices = list(range(0, length))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if type(test_size) is float:
        split = int(np.floor(test_size * length))
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or float'.format(str))
    return indices[split:], indices[:split]


def create_dataset(data, close_len=3, output_len=1,period_len=0, test_rate=0, shuffle=False, norm=True):
    '''test length will be set 30 or 60,
    shuffle equal false is mean using last 30 days for test ...'''
    time_intervel = max(close_len, period_len * 30)
    print(close_len,output_len)
    X = []
    Y = []
    for i in range(time_intervel, len(data) - output_len + 1):
        X.append(data[i - time_intervel:i])
        Y.append(data[i + output_len-1:i+output_len])

    X = np.array(X)
    Y = np.array(Y)
    X = np.reshape(X,(X.shape[0],-1,X.shape[-2],X.shape[-1]))
    if output_len>=1:
        Y=np.squeeze(Y,1)
    if shuffle:
        index = [i for i in range(len(X))]
        random.shuffle(index)
        X = X[index]
        Y = Y[index]

    test_len=int(test_rate*len(X))
    x_train, y_train, x_test, y_test = X[:-test_len], Y[:-test_len], X[-test_len:], Y[-test_len:]

    mmn_list = []
    if norm:
        max_value = np.max(x_train)
        min_value = np.min(x_train)
        max_sub_min = max_value - min_value

        x_train = (x_train- min_value) / max_sub_min
        y_train = (y_train - min_value) / max_sub_min
        x_test = (x_test - min_value) / max_sub_min
        y_test = (y_test - min_value) / max_sub_min

        mmn_list.append(min_value)
        mmn_list.append(max_sub_min)

        return [x_train], [y_train], [x_test], [y_test], mmn_list

    return [x_train], [y_train], [x_test], [y_test], mmn_list

if __name__ == '__main__':
    f = h5py.File('data/Bike_NYC14_M16x8_T60_NewEnd.h5')
    data = f['data']
    x_train, y_train, x_test, y_test, mmn = create_dataset(data,close_len=3,output_len=1,test_rate=0.2)
    x_train+=y_train
    x_test+=y_test
    train_data = list(zip(*x_train))
    test_data = list(zip(*x_test))
    train_idx, valid_idx = train_valid_split(train_data, 0.1, shuffle=True)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=train_sampler,
                              num_workers=8, pin_memory=True)
    valid_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=valid_sampler,
                              num_workers=2, pin_memory=True)

    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)

    # get data channels
    channels = [opt.close_size*2,
                opt.period_size*2,
                opt.trend_size*2]
    model = PWNet(6,2).cuda()

    optimizer = optim.Adam(model.parameters(), opt.lr)

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    if not os.path.isdir(opt.save_dir):
        raise Exception('%s is not a dir' % opt.save_dir)

    if opt.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    elif opt.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    if opt.train:
        print('Training...')
        train()

