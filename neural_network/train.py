import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable

import numpy as np
from ml import nets
from ml import evaluate
from ml import util
import gc
import time

# all inputs are numpy arrays
# X_train/X_test is a float64 2D array
# y_train/y_test is a int64 1D array with value equal to classification type
# converts these inputs to dataloaders, for use in PyTorch training


def load_datasets(X_train, y_train, X_val, y_val,
                  parameters, X_test=None, y_test=None):
    batch_size = parameters.get('batch_size', 200)
    print_test = parameters.get('print_test', False) and (X_test is not None)

    # calculate output dim
    y_combined = np.concatenate((y_train, y_val))
    if y_test is not None:
        y_combined = np.concatenate((y_combined, y_test))
    output_dim = len(np.unique(y_combined))
    print('output_dim is', output_dim)
    parameters['input_dim'] = X_train.shape[1]
    parameters['output_dim'] = output_dim

    # datasets
    # convert np matrices into torch Variables, and then feed them into a
    # dataloader
    X_train, y_train = torch.from_numpy(
        X_train).float(), torch.from_numpy(y_train)
    X_val, y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    test_dataloader = None

    if X_test is not None:
        X_test, y_test = torch.from_numpy(
            X_test).float(), torch.from_numpy(y_test)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, shuffle=True, batch_size=batch_size, num_workers=0)

    return train_dataloader, val_dataloader, test_dataloader


# trains a NN with parameters specified in parameters, with the dataloaders
# test_dataloader can be done
def train(train_dataloader, val_dataloader, test_dataloader,
          parameters, models_directory='../models', suffix=''):
    print('Starting training at ' + util.get_time())
    print(', '.join(['{}={!r}'.format(k, v)
                     for k, v in sorted(parameters.items())]))
    # batch_size is determined in the dataloader, so the variable is
    # irrelevant here
    batch_size = parameters.get('batch_size', 200)
    num_epochs = parameters.get('num_epochs', 100)
    hidden_sizes = parameters.get('hidden_sizes', [200])
    learning_rate = parameters.get('learning_rate', 0.0005)
    weight_decay = parameters.get('weight_decay', 0)
    dropout = parameters.get('dropout', 0.0)
    patience = parameters.get('patience', 10)
    threshold = parameters.get('threshold', 1e-3)
    input_dim = parameters['input_dim']
    output_dim = parameters['output_dim']

    # output_period: output training loss every x batches
    output_period = parameters.get('output_period', 0)
    model_prefix = parameters.get('model_prefix', None)
    only_train = parameters.get('only_train', False)
    save_model = parameters.get('save_model', False)
    test_best = parameters.get('test_best', False)
    print_test = parameters.get(
        'print_test', False) and (
        test_dataloader is not None)

    # we want to print out test accuracies to a separate file
    test_file = None
    if print_test:
        test_file = open('test{}.txt'.format(suffix), 'a')
        test_file.write('\n\n')
        test_file.write('Starting at ' + util.get_time() + '\n')
        test_file.write(', '.join(['{}={!r}'.format(k, v)
                                   for k, v in sorted(parameters.items())]) + '\n\n')

    # nets and optimizers
    criterion = nn.CrossEntropyLoss().cuda()
    net = nets.AdvancedNet(
        input_dim,
        hidden_sizes,
        output_dim,
        dropout=dropout).cuda()
    optimizer = optim.Adam(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)
    # ReduceLROnPlateau reduces learning rate by factor of 10 once val loss
    # has plateaued
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience, threshold=threshold)

    num_train_batches = len(train_dataloader)
    epoch = 1
    best_epoch, best_acc = 0, 0
    train_acc = [0]

    print('starting training')
    while epoch <= num_epochs:
        running_loss = 0.0
        epoch_acc = 0.0

        net.train()
        print(
            'epoch: %d, lr: %.1e' %
            (epoch,
             optimizer.param_groups[0]['lr']) +
            '    ' +
            util.get_time())
        for batch_num, (inputs, labels) in enumerate(train_dataloader, 1):
            optimizer.zero_grad()
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # output is 2D array (logsoftmax output), so we flatten it to a 1D to get the max index for each example
            # and then calculate accuracy off that
            max_index = outputs.max(dim=1)[1]
            epoch_acc += np.sum(max_index.data.cpu().numpy()
                                == labels.data.cpu().numpy()) / inputs.size()[0]

            # output every output_period batches
            if output_period:
                if batch_num % output_period == 0:
                    print('[%d:%.2f] loss: %.3f' % (
                        epoch, batch_num * 1.0 / num_train_batches,
                        running_loss / output_period))
                    running_loss = 0.0
                    gc.collect()

        # save model after every epoch in models/ folder
        if save_model:
            torch.save(
                net.state_dict(),
                models_directory +
                '/' +
                model_prefix +
                ".%d" %
                epoch)

        # print training/val accuracy
        epoch_acc = epoch_acc / num_train_batches
        train_acc.append(epoch_acc)
        print('train acc: %.4f' % (epoch_acc))
        if only_train:
            scheduler.step(loss)
        else:
            val_accuracy, total_loss = evaluate.eval_error(
                net, val_dataloader, criterion)
            print('val acc: %.4f, loss: %.4f' % (val_accuracy, total_loss))
            # remember: feed val loss into scheduler
            scheduler.step(total_loss)
            if val_accuracy > best_acc:
                best_epoch, best_acc = epoch, val_accuracy
            print()

            # write test accuracy
            if print_test:
                test_accuracy, total_loss = evaluate.eval_error(
                    net, test_dataloader, criterion)
                test_file.write(
                    'epoch: %d' %
                    (epoch) +
                    '    ' +
                    util.get_time() +
                    '\n')
                test_file.write('train acc: %.4f' % (epoch_acc) + '\n')
                test_file.write('val acc: %.4f' % (val_accuracy) + '\n')
                test_file.write('test acc: %.4f' % (test_accuracy) + '\n')
                test_file.write('loss: %.4f' % (test_accuracy) + '\n')

            gc.collect()
        # perform early stopping here if our learning rate is below a threshold
        # because small lr means little change in accuracy anyways
        if optimizer.param_groups[0]['lr'] < (0.9 * 0.01 * learning_rate):
            print('Low LR reached, finishing training early')
            break
        epoch += 1

    print('best epoch: %d' % best_epoch)
    print('best val accuracy: %.4f' % best_acc)
    print('train accuracy at that epoch: %.4f' % train_acc[best_epoch])
    print('ending at', time.ctime())

    if test_best:
        net.load_state_dict(
            torch.load(
                models_directory +
                '/' +
                model_prefix +
                '.' +
                str(best_epoch)))
        best_test_accuracy, total_loss = evaluate.eval_error(
            net, test_dataloader, criterion)
        test_file.write('*****\n')
        test_file.write(
            'best test acc: %.4f, loss: %.4f' %
            (best_test_accuracy, total_loss) + '\n')
        test_file.write('*****\n')
        print('best test acc: %.4f, loss: %.4f' %
              (best_test_accuracy, total_loss))

    if print_test:
        test_file.write('\n')
        test_file.close()

    print('\n\n\n')
