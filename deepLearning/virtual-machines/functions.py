import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import torch
import torchio as tio
from torch.utils.data import DataLoader
from scipy import stats
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR


def find_coordinates_to_extract_views(subject_image): # return [x,y,z] array of where to look for the views as well as maxs (best 2D projections)
    
    data = subject_image.data[0]
    max = [0,0,0]
    index = [0,0,0]
    shape = data.shape

    for i in range(shape[0]):
        if data[i,:,:].sum() > max[0]:
            max[0] = data[i,:,:].sum()
            index[0] = i
    
    for i in range(shape[1]):
        if data[:,i,:].sum() > max[1]:
            max[1] = data[:,i,:].sum()
            index[1] = i            
    
    for i in range(shape[2]):
        if data[:,:,i].sum() > max[2]:
            max[2] = data[:,:,i].sum()
            index[2] = i
    
    return index

def get_view(subject_image, view, mode = 0): # view = 0,1,2 for sagittal, coronal, axial respectively. Allows work on torch objects

    if mode == 0: # mode 0 for actual best 2D projections, mode 1 for the ones used by torchio (middle points views)
        index = find_coordinates_to_extract_views(subject_image)
        if view == 0:
            return subject_image.data[0, index[0],:,:]
        elif view == 1:
            return subject_image.data[0, :, index[1],:]
        else:
            return subject_image.data[0, :, :, index[2]]
    else:
        if view == 0:
            return subject_image.data[0, subject_image.data.shape[1]//2,:,:]
        elif view == 1:
            return subject_image.data[0, :, subject_image.data.shape[2]//2,:]
        else:
            return subject_image.data[0, :, :, subject_image.data.shape[3]//2]

def show_view(subject_image, view, mode = 0): # view = 0,1,2 for sagittal, coronal, axial respectively. Streching done for sagittal and coronal views
    
    if mode == 0: # mode 0 for actual best 2D projections, mode 1 for the ones used by torchio (middle points views)
        index = find_coordinates_to_extract_views(subject_image)
        if view == 0:
            plt.imshow(subject_image.data[0, index[0],:,:], aspect = 0.20, cmap = 'gray')
        elif view == 1:
            plt.imshow(subject_image.data[0, :, index[1],:], aspect = 0.20, cmap = 'gray')
        else:
            plt.imshow(subject_image.data[0, :, :, index[2]], cmap = 'gray')
    else:
        if view == 0:
            plt.imshow(subject_image.data[0, subject_image.data.shape[1]//2,:,:], aspect = 0.20, cmap = 'gray')
        elif view == 1:
            plt.imshow(subject_image.data[0, :, subject_image.data.shape[2]//2,:], aspect = 0.20, cmap = 'gray')
        else:
            plt.imshow(subject_image.data[0, :, :, subject_image.data.shape[3]//2], cmap = 'gray')

def get_volume(seg_subject_view, cat): # return volume of the cat class in the selected segmented view. 0 is background, 1 is LV, 2 is RV, 3 is Myo
    data = seg_subject_view.data[0]
    if cat == 0:
        return (data == 0).sum().item()
    elif cat == 1:
        return (data == 1).sum().item()
    elif cat == 2:
        return (data == 2).sum().item()
    elif cat == 3:
        return (data == 3).sum().item()
    else:
        return -1
    
def load_training_dataset(metaDataClean): #load all 100 subjects from training dataset inside a list of torchio subjects
    subject_list = []              #need argument for csv metadata because of import. Always
    for i in range(1, 10):          # call with metaDataClean = pd.read_csv('metadata.csv', header=None).iloc
        subject = tio.Subject(
            ed=tio.ScalarImage('../Train/00' + str(i) + '/00' + str(i) + '_ED.nii'),
            es = tio.ScalarImage('../Train/00' + str(i) + '/00' + str(i) + '_ES.nii'),
            ed_seg=tio.ScalarImage('../Train/00' + str(i) + '/00' + str(i) + '_ED_seg.nii'),
            es_seg = tio.ScalarImage('../Train/00' + str(i) + '/00' + str(i) + '_ES_seg.nii'),
            diagnosis=metaDataClean[i-1][1].astype('int'),
        )
        subject_list.append(subject)

    for i in range(10, 100):
        subject = tio.Subject(
            ed=tio.ScalarImage('../Train/0' + str(i) + '/0' + str(i) + '_ED.nii'),
            es = tio.ScalarImage('../Train/0' + str(i) + '/0' + str(i) + '_ES.nii'),
            ed_seg=tio.ScalarImage('../Train/0' + str(i) + '/0' + str(i) + '_ED_seg.nii'),
            es_seg = tio.ScalarImage('../Train/0' + str(i) + '/0' + str(i) + '_ES_seg.nii'),
            diagnosis=metaDataClean[i-1][1].astype('int'),
        )
        subject_list.append(subject)

    subject_list.append(tio.Subject(
        ed=tio.ScalarImage('../Train/100/100_ED.nii'),
        es = tio.ScalarImage('../Train/100/100_ES.nii'),
        ed_seg=tio.ScalarImage('../Train/100/100_ED_seg.nii'),
        es_seg = tio.ScalarImage('../Train/100/100_ES_seg.nii'),
        diagnosis=metaDataClean[99][1].astype('int'),
    ))
    return subject_list

def load_filled_test_dataset(): #load all 50 subjects from test dataset inside a list of torchio subjects
    subject_list = []              
    for i in range(1, 10):          
        subject = tio.Subject(
            ed=tio.ScalarImage('../TestFilled/10' + str(i) + '/10' + str(i) + '_ED.nii'),
            es = tio.ScalarImage('../TestFilled/10' + str(i) + '/10' + str(i) + '_ES.nii'),
            ed_seg=tio.ScalarImage('../TestFilled/10' + str(i) + '/10' + str(i) + '_ED_seg.nii'),
            es_seg = tio.ScalarImage('../TestFilled/10' + str(i) + '/10' + str(i) + '_ES_seg.nii'),
        )
        subject_list.append(subject)

    for i in range(10, 51):
        subject = tio.Subject(
            ed=tio.ScalarImage('../TestFilled/1' + str(i) + '/1' + str(i) + '_ED.nii'),
            es = tio.ScalarImage('../TestFilled/1' + str(i) +'/1' + str(i) + '_ES.nii'),
            ed_seg=tio.ScalarImage('../TestFilled/1' + str(i) +'/1' + str(i) + '_ED_seg.nii'),
            es_seg = tio.ScalarImage('../TestFilled/1' + str(i) +'/1' + str(i) + '_ES_seg.nii'),
        )
        subject_list.append(subject)

    return subject_list



def select_row_x_and_y_from_table(table, x, y):
    return np.array([[table[i, x], table[i, y]] for i in range(table[:].shape[0])])

def compute_mean_and_covariance_matrix(data):
    mean = np.mean(data, axis = 0)
    covariance = np.cov(data.T)
    return mean, covariance

def select_lines_where_class_value_is_X(data, X):
    return data[np.where(data[:, 1] == X)]

def dens(X, Y, m, s):
    d = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            d[i, j] = stats.multivariate_normal.pdf([x, y], mean = m, cov = s)
    return d

def normalize_features_at_indices(data, mean = None, std = None, indices = [2,3,4,5,6,7,8,9]):

    if mean is None:
        mean_list = []
        std_list = []
        for i in indices:
            mean_list.append(np.mean(data[:, i]))
            std_list.append(np.std(data[:, i]))
            data[:, i] = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])
        return data, np.array(mean_list), np.array(std_list)
    else:
        j = 0
        for i in indices:
            print(i)
            data[:, i] = (data[:, i] - mean[j]) / std[j]
            j+=1
        return data

def accuracy(y, y_pred):
    return (y == y_pred).sum() / y.shape[0]


############################################################################################################
#                                          Deep Learning Functions                                       #
############################################################################################################
import torch.nn.functional as F

def loss_batch(model, loss_func, xb, yb, opt = None, metric = None):

    preds = model(xb)
    loss = loss_func(preds, yb)

    if opt is not None:
        
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None

    if metric is not None:

        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result


def evaluate(model, loss_func, x_val, y_val, metric = None):
    
    results = []
    with torch.no_grad():
        
        results.append(loss_batch(model, loss_func, x_val, y_val, metric = metric))

        losses, nums, metrics = zip(*results)

        total = np.sum(nums)

        avg_loss = np.sum(np.multiply(losses, nums)) / total

        avg_metric = None

        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total

    return avg_loss, total, avg_metric


def fit(epochs, optimizer, model, loss_func, opt, x_train, y_train, x_val, y_val, metric):
    
    lr_scheduler = CyclicLR(optimizer, base_lr=1e-7, max_lr=10, step_size_up=100, mode='triangular')

    for epoch in range(epochs):


        loss, _, _ = loss_batch(model, loss_func, x_train, y_train, opt)
        lr_scheduler.step()

        result = evaluate(model, loss_func, x_val, y_val, metric)
        val_loss, total, val_metric = result

        print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'.format(epoch+1, epochs, val_loss, metric.__name__, val_metric))

def fit2(epochs, model, loss_func, opt, x_train, y_train, x_val, y_val, metric):
    for epoch in range(epochs):

        
        loss, _, _ = loss_batch(model, loss_func, x_train, y_train, opt)

        result = evaluate(model, loss_func, x_val, y_val, metric)
        val_loss, total, val_metric = result

        print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'.format(epoch+1, epochs, val_loss, metric.__name__, val_metric))



def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.sum(preds == labels).item() / len(preds)