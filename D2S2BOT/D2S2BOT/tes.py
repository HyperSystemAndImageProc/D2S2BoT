import os

import numpy as np
import time
import collections
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

# import torchsummary
from torch import optim
import torch
from sklearn import metrics, preprocessing
import datetime
import scipy.io as sio  # mat
import sys
from sklearn.decomposition import PCA
sys.path.append('../global_module/')
import global_module.network as network
import global_module.train as train
from operator import truediv

from global_module.generate_pic import aa_and_each_accuracy, sampling, load_dataset, generate_png, generate_iter


from Utils import record



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')


def __init__(self, path='./'):
    # save the path
    self.path = path
    if not os.path.exists(path):
        os.mkdir(path)


def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX


# for Monte Carlo runs

seeds = [1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241]
ensemble = 1

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')

print('-----Importing Dataset-----')

global Dataset  # SV,IN,BS
dataset = 'IN'
# Dataset = dataset.upper()
data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT = load_dataset(dataset)
data_hsi = applyPCA(data_hsi,30)

print(data_hsi.shape)
image_x, image_y, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )
CLASSES_NUM = max(gt)
print('The class numbers of the HSI data is:', CLASSES_NUM)

print('-----Importing Setting Parameters-----')
ITER = 1
PATCH_LENGTH = 5

# lr, num_epochs, batch_size = 0.1, 1, 32
lr, num_epochs, batch_size = 0.001, 50, 32
loss = torch.nn.CrossEntropyLoss()
PATCH = PATCH_LENGTH*2 +1
img_rows = 2 * PATCH_LENGTH + 1
img_cols = 2 * PATCH_LENGTH + 1
img_channels = data_hsi.shape[2]
INPUT_DIMENSION = data_hsi.shape[2]
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
VAL_SIZE = int(TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE

KAPPA = []
OA = []
AA = []

TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))
data = preprocessing.scale(data)
data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
whole_data = data_
padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)

net = network.D2S2BoT(band = BAND,classes = CLASSES_NUM,patch=11,layers=3)

optimizer = optim.Adam(net.parameters(), lr=lr, amsgrad=False)  # , weight_decay=0.0001)
time_1 = int(time.time())
np.random.seed(seeds[0])
train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
_, total_indices = sampling(1, gt)

def tes(device, net, test_loader):
    count = 0
    # 模型测试
    newnet =net.cuda()
    #SV
    # newnet.load_state_dict(torch.load('./SV/SV03_08_16_100.9922.pt'))
    #BS
    # newnet.load_state_dict(torch.load('./BS/BS03_08_17_050.9662.pt'))
    #IN
    newnet.load_state_dict(torch.load('./IN/IN03_07_15_350.9566.pt'))
    newnet.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = newnet(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


TRAIN_SIZE = len(train_indices)
print('Train size: ', TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
print('Test size: ', TEST_SIZE)
VAL_SIZE = int(TRAIN_SIZE)
print('Validation size: ', VAL_SIZE)

print('-----Selecting Small Pieces from the Original Cube Data-----')

train_iter, valida_iter, test_iter, all_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices,
                                                             TOTAL_SIZE, total_indices, VAL_SIZE,
                                                             whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION,
                                                             batch_size, gt)
net =network.D2S2BoT(BAND,patch=11,classes=CLASSES_NUM,layers=3)


y_pred_test , y_test = tes(0,net,all_iter)





overall_acc_fdssc = metrics.accuracy_score(y_pred_test, y_test)
confusion_matrix_fdssc = metrics.confusion_matrix(y_pred_test, y_test)
each_acc_fdssc, average_acc_fdssc = aa_and_each_accuracy(confusion_matrix_fdssc)
kappa = metrics.cohen_kappa_score(y_pred_test, y_test)


KAPPA.append(kappa)
OA.append(overall_acc_fdssc)
AA.append(average_acc_fdssc)

ELEMENT_ACC[ :] = each_acc_fdssc

print("-------- Training Finished-----------")
print('OA:', OA)
print('AA:', AA)
print('Kappa:', KAPPA)
record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                     dataset + '/' + day_str + dataset + str(
                         VALIDATION_SPLIT) +  '.txt')


dataacc = dataset + str(round(overall_acc_fdssc, 4))

newnet = net.cuda()
generate_png(all_iter, net, gt_hsi,dataset, dataacc, device, total_indices,overall_acc_fdssc)
