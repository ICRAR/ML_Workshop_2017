# -*- coding: utf-8 -*-
#
#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2017
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#    MA 02111-1307  USA
#

from __future__ import print_function

import argparse
import logging
import math
import os
from os import makedirs
from os.path import exists
from timeit import default_timer

import h5py
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.utils.data as data
from astropy.utils.console import human_time
from statsmodels.robust import scale
from torch.autograd import Variable
from torch.utils.data import Dataset

H5_VERSION = '2017_10_20_001'
NUMBER_CHANNELS = 1
NUMBER_OF_CLASSES = 2
URL_ROOT = 'http://ict.icrar.org/store/staff/kevin/rfi'
HIDDEN_LAYERS = 200


LOGGER = logging.getLogger(__name__)


class H5Exception(Exception):
    pass


class RfiData(object):
    def __init__(self, **kwargs):
        self._sequence_length = kwargs['sequence_length']
        self._num_processes = kwargs['num_processes']
        self._using_gpu = kwargs['using_gpu']
        output_file = os.path.join(kwargs['data_path'], kwargs['data_file'])
        with h5py.File(output_file, 'r') as h5_file:
            data_group = h5_file['data']

            # Move the data into memory
            self._data_channel_0 = np.copy(data_group['data_channel_0'])
            self._labels = np.copy(data_group['labels'])

            length_data = len(self._labels) - kwargs['sequence_length']
            split_point1 = int(length_data * kwargs['training_percentage'] / 100.)
            split_point2 = int(length_data * (kwargs['training_percentage'] + kwargs['validation_percentage']) / 100.)
            perm0 = np.arange(length_data)
            np.random.shuffle(perm0)

            self._train_sequence = perm0[:split_point1]
            self._validation_sequence = perm0[split_point1:split_point2]
            self._test_sequence = perm0[split_point2:]

    def get_rfi_dataset(self, data_type, rank=None, short_run_size=None):
        if data_type not in ['training', 'validation', 'test']:
            raise ValueError("data_type must be one of: 'training', 'validation', 'test'")

        if data_type == 'training':
            sequence = self._train_sequence
        elif data_type == 'validation':
            sequence = self._validation_sequence
        else:
            sequence = self._test_sequence

        if self._using_gpu or rank is None:
            if short_run_size is not None:
                sequence = sequence[0:short_run_size]
        else:
            section_length = len(sequence) / self._num_processes
            start = rank * section_length
            if rank == self._num_processes - 1:
                if short_run_size is not None:
                    sequence = sequence[start:start + short_run_size]
                else:
                    sequence = sequence[start:]
            else:
                if short_run_size is not None:
                    sequence = sequence[start:start + short_run_size]
                else:
                    sequence = sequence[start:start + section_length]

        return RfiDataset(sequence, self._data_channel_0, self._labels, self._sequence_length)


class RfiDataset(Dataset):
    def __init__(self, selection_order, x_data, y_data, sequence_length):
        self._x_data = x_data
        self._y_data = y_data
        self._selection_order = selection_order
        self._length = len(selection_order)
        self._sequence_length = sequence_length
        self._actual_node = self._sequence_length / 2
        self._median = np.median(x_data)
        self._median_absolute_deviation = scale.mad(x_data, c=1)
        self._mean = np.mean(x_data)
        LOGGER.debug('Length: {}'.format(self._length))

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        selection_index = self._selection_order[index]
        x_data = self._x_data[selection_index:selection_index + self._sequence_length]
        local_median = np.median(x_data)
        local_median_absolute_deviation = scale.mad(x_data, c=1)
        local_mean = np.mean(x_data)
        # x_data_last = x_data[self._actual_node]

        data = [self._median, self._median_absolute_deviation, self._mean, local_median, local_median_absolute_deviation, local_mean]
        for item in x_data:
            data.append(item)
            data.append(item - self._mean)
            data.append(item - self._median)
            data.append(item - self._median_absolute_deviation)
            data.append(item - local_mean)
            data.append(item - local_median)
            data.append(item - local_median_absolute_deviation)

        # return np.reshape(x_data, (NUMBER_CHANNELS, -1)), values, self._y_data[selection_index + self._actual_node]
        return np.array(data), self._y_data[selection_index + self._actual_node]


def process_files(filename, rfi_label):
    """ Process a file and return the data and the labels """
    files_to_process = []
    for ending in ['.txt', '_loc.txt']:
        complete_filename = filename + ending
        files_to_process.append(complete_filename)

    # Load the files into numpy
    LOGGER.info('Loading: {}'.format(files_to_process[0]))
    data_frame = pd.read_csv(files_to_process[0], header=None, delimiter=' ')
    data = data_frame.values.flatten()

    LOGGER.info('Loading: {}'.format(files_to_process[1]))
    data_frame = pd.read_csv(files_to_process[1], header=None, delimiter=' ')
    labels = data_frame.values.flatten()

    # Check the lengths match
    assert len(data) == len(labels), 'The line counts do not match for: {0}'.format(filename)

    # If substitute of the label is needed
    if rfi_label != 1:
        labels[labels == 1] = rfi_label

    return data, labels


def build_data(**kwargs):
    """ Read data """
    output_file = os.path.join(kwargs['data_path'], kwargs['data_file'])
    if os.path.exists(output_file):
        with h5py.File(output_file, 'r') as h5_file:
            # Everything matches
            if 'version' in h5_file.attrs and h5_file.attrs['version'] == H5_VERSION:
                # All good nothing to do
                return

    # Open the output files
    with Timer('Processing input files'):
        data1, labels1 = process_files(URL_ROOT + '/impulsive_broadband_simulation_random_5p', 1)
        data2, labels2 = process_files(URL_ROOT + '/impulsive_broadband_simulation_random_10p', 1)
        data3, labels3 = process_files(URL_ROOT + '/repetitive_rfi_timeseries', 1)
        data4, labels4 = process_files(URL_ROOT + '/repetitive_rfi_random_timeseries', 1)
        # data0, labels0 = process_files(URL_ROOT + '/impulsive_broadband_simulation_random_norfi', 0)

    # Concatenate
    with Timer('Concatenating data'):
        labels = np.concatenate((labels1, labels2, labels3, labels4))
        data = np.concatenate((data1, data2, data3, data4))

    # Standardise and one hot
    with Timer('Standardise & One hot'):
        labels = one_hot(labels, NUMBER_OF_CLASSES)
        # data = normalize(data)

    with Timer('Saving to {0}'.format(output_file)):
        if not exists(kwargs['data_path']):
            makedirs(kwargs['data_path'])
        with h5py.File(output_file, 'w') as h5_file:
            h5_file.attrs['number_channels'] = NUMBER_CHANNELS
            h5_file.attrs['number_classes'] = NUMBER_OF_CLASSES
            h5_file.attrs['version'] = H5_VERSION

            # If the data is standardised standardise the training data and then use the mean and std values to
            # standardise the validation and training

            data_group = h5_file.create_group('data')
            data_group.attrs['length_data'] = len(data)
            data_group.create_dataset('data_channel_0', data=data, compression='gzip')
            data_group.create_dataset('labels', data=labels, compression='gzip')


def normalize(all_data):
    """ normalize data """
    min_value = np.min(all_data)
    max_value = np.max(all_data)
    return (all_data - min_value) / (max_value - min_value)


def standardize(all_data):
    """ Standardize data """
    return (all_data - np.mean(all_data)) / np.std(all_data)


def one_hot(labels, number_class):
    """ One-hot encoding """
    expansion = np.eye(number_class)
    y = expansion[:, labels].T
    assert y.shape[1] == number_class, "Wrong number of labels!"

    return y


class Timer(object):
    def __init__(self, name=None):
        self.name = '' if name is None else name
        self.timer = default_timer

    def __enter__(self):
        LOGGER.info('{}, Starting timer'.format(self.name))
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs
        LOGGER.info('{}, Elapsed time: {}'.format(self.name, human_time(self.elapsed)))


class Histogram(object):
    def __init__(self, data, bins=10, title=None, number_range=None, histogram_type='bars'):
        self.bins = bins
        self.title = title
        self.type = histogram_type
        self.histogram = np.histogram(np.array(data), bins=self.bins, range=number_range)
        if histogram_type == 'numbers':
            total = len(data)
            self.percentages = [bin_value * 100.0 / total for bin_value in self.histogram[0]]

    def horizontal(self, height=4, character='|'):
        if self.title is not None:
            his = "{0}\n\n".format(self.title)
        else:
            his = ""

        if self.type == 'bars':
            bars = self.histogram[0] / float(max(self.histogram[0])) * height
            for reversed_height in reversed(range(1, height+1)):
                if reversed_height == height:
                    line = '{0} '.format(max(self.histogram[0]))
                else:
                    line = ' '*(len(str(max(self.histogram[0]))) + 1)
                for c in bars:
                    if c >= math.ceil(reversed_height):
                        line += character
                    else:
                        line += ' '
                line += '\n'
                his += line
            his += '{0:.2f}'.format(self.histogram[1][0]) + ' ' * self.bins + '{0:.2f}'.format(self.histogram[1][-1]) + '\n'
        else:
            his += ' ' * 4
            his += ''.join(['| {0:^8.2f}%'.format(n) for n in self.percentages])
            his += '|\n'
            his += ' ' * 4
            his += ''.join(['| {0:^8} '.format(n) for n in self.histogram[0]])
            his += '|\n'
            his += ' ' * 4
            his += '|----------'*len(self.histogram[0])
            his += '|\n'
            his += ''.join(['| {0:^8.2f} '.format(n) for n in self.histogram[1]])
            his += '|\n'
        return his

    def vertical(self, height=20, character='|'):
        if self.title is not None:
            his = "{0}\n\n".format(self.title)
        else:
            his = ""

        if self.type == 'bars':
            xl = ['{0:.2f}'.format(n) for n in self.histogram[1]]
            lxl = [len(l) for l in xl]
            bars = self.histogram[0] / float(max(self.histogram[0])) * height
            bars = np.rint(bars).astype('uint32')
            his += ' '*(max(bars)+2+max(lxl))+'{0}\n'.format(max(self.histogram[0]))
            for i, c in enumerate(bars):
                line = xl[i] + ' '*(max(lxl)-lxl[i])+': ' + character*c+'\n'
                his += line
        else:
            for item1, item2, item3 in zip(self.histogram[0], self.histogram[1], self.percentages):
                line = '{0:>6.2f} | {1:>5} | {2:>6.2f}%\n'.format(item2, item1, item3)
                his += line
        return his


class GmrtLinear(nn.Module):
    def __init__(self, keep_probability, sequence_length):
        super(GmrtLinear, self).__init__()
        self.keep_probability = keep_probability
        self.input_layer_length = 6 + (sequence_length * 7)

        self.fc1 = nn.Linear(self.input_layer_length, HIDDEN_LAYERS).double()
        self.fc2 = nn.Linear(HIDDEN_LAYERS + self.input_layer_length, HIDDEN_LAYERS).double()
        self.fc3 = nn.Linear(HIDDEN_LAYERS, HIDDEN_LAYERS).double()
        self.fc4 = nn.Linear(HIDDEN_LAYERS, HIDDEN_LAYERS).double()
        self.fc5 = nn.Linear(HIDDEN_LAYERS, HIDDEN_LAYERS).double()
        self.fc6 = nn.Linear(HIDDEN_LAYERS, NUMBER_OF_CLASSES).double()

    def forward(self, input_data_values):
        x = functional.leaky_relu(self.fc1(input_data_values))
        x = functional.leaky_relu(self.fc2(torch.cat((x, input_data_values), dim=1)))
        x = functional.dropout(x, p=self.keep_probability, training=self.training)
        x = functional.leaky_relu(self.fc3(x))
        x = functional.leaky_relu(self.fc4(x))
        x = functional.dropout(x, p=self.keep_probability, training=self.training)
        x = functional.leaky_relu(self.fc5(x))
        x = functional.leaky_relu(self.fc6(x))

        x = functional.softmax(x)
        return x


def train(model, rfi_data, rank=0, **kwargs):
    # This is needed to "trick" numpy into using different seeds for different processes
    if kwargs['seed'] is not None:
        np.random.seed(kwargs['seed'] + rank)
    else:
        np.random.seed()

    train_loader = data.DataLoader(
        rfi_data.get_rfi_dataset('training', rank=rank, short_run_size=kwargs['short_run']),
        batch_size=kwargs['batch_size'],
        num_workers=3,
        pin_memory=kwargs['using_gpu'],
    )
    test_loader = data.DataLoader(
        rfi_data.get_rfi_dataset('validation', rank=rank, short_run_size=kwargs['short_run']),
        batch_size=kwargs['batch_size'],
        num_workers=3,
        pin_memory=kwargs['using_gpu'],
    )

    optimizer = optim.SGD(model.parameters(), lr=kwargs['learning_rate'], momentum=kwargs['momentum'])
    for epoch in range(1, kwargs['epochs'] + 1):
        # Adjust the learning rate
        adjust_learning_rate(optimizer, epoch, kwargs['learning_rate_decay'], kwargs['start_learning_rate_decay'], kwargs['learning_rate'])
        train_epoch(epoch, model, train_loader, optimizer, kwargs['log_interval'])
        test_epoch(model, test_loader, kwargs['log_interval'])


def train_epoch(epoch, model, data_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (x_data_raw, target) in enumerate(data_loader):
        # x_data_ts = Variable(x_data_ts)
        x_data_raw = Variable(x_data_raw)
        target = Variable(target)
        optimizer.zero_grad()
        output = model(x_data_raw)
        if type(output.data) == torch.cuda.DoubleTensor:
            output = output.cpu()
        loss = functional.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0 and batch_idx > 1:
            LOGGER.info('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
                epoch,
                batch_idx * len(x_data_raw),
                len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss.data[0])
            )


def build_histogram(output, target_column, histogram_data):
    for values, column in zip(output.data.numpy(), target_column.numpy()):
        histogram_data['all'].append(values[column])
        histogram_data[column].append(values[column])


def test_epoch(model, data_loader, log_interval):
    model.eval()
    test_loss = 0
    correct = 0
    histogram_data = {key: [] for key in range(NUMBER_OF_CLASSES)}
    histogram_data['all'] = []
    for batch_index, (x_data_raw, target) in enumerate(data_loader):
        # x_data_ts = Variable(x_data_ts, volatile=True)
        x_data_raw = Variable(x_data_raw, volatile=True)
        target = Variable(target)
        output = model(x_data_raw)
        if type(output.data) == torch.cuda.DoubleTensor:
            output = output.cpu()
        test_loss += functional.binary_cross_entropy(output, target).data[0]
        pred = output.data.max(1)[1]
        target_column = target.data.max(1)[1]
        correct += pred.eq(target_column).sum()
        build_histogram(output, target_column, histogram_data)

        if batch_index % log_interval == 0 and batch_index > 1:
            LOGGER.info('Test iteration: {}, Correct count: {}'.format(batch_index, correct))

    test_loss /= len(data_loader.dataset)
    LOGGER.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(data_loader.dataset),
        100. * correct / len(data_loader.dataset))
    )
    for key, value in histogram_data.items():
        histogram = Histogram(
            value,
            title='Percentage of Correctly Predicted: {}'.format(key),
            bins=10,
            number_range=(0.0, 1.0),
            histogram_type='numbers'
        )
        LOGGER.info(histogram.horizontal())


def adjust_learning_rate(optimizer, epoch, learning_rate_decay, start_learning_rate_decay, learning_rate):
    """ Sets the learning rate to the initial LR decayed  """
    lr_decay = learning_rate_decay ** max(epoch + 1 - start_learning_rate_decay, 0.0)
    new_learning_rate = learning_rate * lr_decay
    LOGGER.info('New learning rate: {}'.format(new_learning_rate))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = new_learning_rate


def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(process)d:%(levelname)s:%(name)s:%(message)s')
    parser = argparse.ArgumentParser(description='GMRT CNN Training')
    parser.add_argument('--batch_size', type=int, default=20000, metavar='N', help='input batch size for training (default: 20000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 5)')
    parser.add_argument('--learning_rate', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--keep_probability', type=float, default=0.6, metavar='K', help='Dropout keep probability (default: 0.6)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--num_processes', type=int, default=4, metavar='N', help='how many training processes to use (default: 4)')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='use the GPU if it is available')
    parser.add_argument('--data_path', default='./data', help='the path to the data file')
    parser.add_argument('--data_file', default='data.h5', help='the name of the data file')
    parser.add_argument('--sequence_length', type=int, default=10, help='how many elements in a sequence')
    parser.add_argument('--validation_percentage', type=int, default=10, help='amount of data used for validation')
    parser.add_argument('--training_percentage', type=int, default=80, help='amount of data used for training')
    parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--learning_rate_decay', type=float, default=0.8, metavar='LRD', help='the initial learning rate decay rate')
    parser.add_argument('--start_learning_rate_decay', type=int, default=5, help='the epoch to start applying the LRD')
    parser.add_argument('--short_run', type=int, default=None, help='use a short run of the test data')
    parser.add_argument('--save', type=str,  default=None, help='path to save the final model')
    parser.add_argument('--dataset', type=int,  default=1, help='the dataset you wish to run')

    kwargs = vars(parser.parse_args())
    LOGGER.debug(kwargs)

    # If the have specified a seed get a random
    if kwargs['seed'] is not None:
        np.random.seed(kwargs['seed'])
    else:
        np.random.seed()

    if kwargs['use_gpu'] and torch.cuda.is_available():
        LOGGER.info('Using cuda devices: {}'.format(torch.cuda.device_count()))
        kwargs['cuda_device_count'] = torch.cuda.device_count()
        kwargs['using_gpu'] = True
    else:
        LOGGER.info('Using CPU')
        kwargs['cuda_device_count'] = 0
        kwargs['using_gpu'] = False

    # Do this first so all the data is built before we go parallel and get race conditions
    with Timer('Checking/Building data file'):
        build_data(**kwargs)

    rfi_data = RfiData(**kwargs)

    if kwargs['using_gpu']:
        # The DataParallel will distribute the model to all the available GPUs
        model = nn.DataParallel(GmrtLinear(kwargs['keep_probability'], kwargs['sequence_length'])).cuda()

        # Train
        train(model, rfi_data, **kwargs)

    else:
        # This uses the HOGWILD! approach to lock free SGD
        model = GmrtLinear(kwargs['keep_probability'], kwargs['sequence_length'])
        model.share_memory()  # gradients are allocated lazily, so they are not shared here

        processes = []
        for rank in range(kwargs['num_processes']):
            p = mp.Process(target=train, args=(model, rfi_data, rank), kwargs=kwargs)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    with Timer('Reading final test data'):
        test_loader = data.DataLoader(
            rfi_data.get_rfi_dataset('test', short_run_size=kwargs['short_run']),
            batch_size=kwargs['batch_size'],
            num_workers=3,
            pin_memory=kwargs['using_gpu'],
        )
    with Timer('Final test'):
        test_epoch(model, test_loader, kwargs['log_interval'])

    if kwargs['save'] is not None:
        with Timer('Saving model'):
            with open(kwargs['save'], 'wb') as save_file:
                torch.save(model.state_dict(), save_file)


if __name__ == '__main__':
    main()
