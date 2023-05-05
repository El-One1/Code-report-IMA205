import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import torch
import torchio as tio
from torch.utils.data import DataLoader
from scipy import stats
from scipy.ndimage import distance_transform_edt
import cv2
from scipy.ndimage import binary_fill_holes



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
            ed=tio.ScalarImage('Train/00' + str(i) + '/00' + str(i) + '_ED.nii'),
            es = tio.ScalarImage('Train/00' + str(i) + '/00' + str(i) + '_ES.nii'),
            ed_seg=tio.ScalarImage('Train/00' + str(i) + '/00' + str(i) + '_ED_seg.nii'),
            es_seg = tio.ScalarImage('Train/00' + str(i) + '/00' + str(i) + '_ES_seg.nii'),
            diagnosis=metaDataClean[i-1][1].astype('int'),
        )
        subject_list.append(subject)

    for i in range(10, 100):
        subject = tio.Subject(
            ed=tio.ScalarImage('Train/0' + str(i) + '/0' + str(i) + '_ED.nii'),
            es = tio.ScalarImage('Train/0' + str(i) + '/0' + str(i) + '_ES.nii'),
            ed_seg=tio.ScalarImage('Train/0' + str(i) + '/0' + str(i) + '_ED_seg.nii'),
            es_seg = tio.ScalarImage('Train/0' + str(i) + '/0' + str(i) + '_ES_seg.nii'),
            diagnosis=metaDataClean[i-1][1].astype('int'),
        )
        subject_list.append(subject)

    subject_list.append(tio.Subject(
        ed=tio.ScalarImage('Train/100/100_ED.nii'),
        es = tio.ScalarImage('Train/100/100_ES.nii'),
        ed_seg=tio.ScalarImage('Train/100/100_ED_seg.nii'),
        es_seg = tio.ScalarImage('Train/100/100_ES_seg.nii'),
        diagnosis=metaDataClean[99][1].astype('int'),
    ))
    return subject_list

def lv_volume(binary_image):
    mask = np.zeros(binary_image.shape)
    for i in range (binary_image.shape[2]):
        mask[:,:,i] = binary_fill_holes(binary_image[:,:,i]).astype(int)
        mask[:,:,i] -= binary_image[:,:,i]
        
    return mask


def load_filled_test_dataset(): #load all 50 subjects from test dataset inside a list of torchio subjects
    subject_list = []              
    for i in range(1, 10):          
        subject = tio.Subject(
            ed=tio.ScalarImage('TestFilled/10' + str(i) + '/10' + str(i) + '_ED.nii'),
            es = tio.ScalarImage('TestFilled/10' + str(i) + '/10' + str(i) + '_ES.nii'),
            ed_seg=tio.ScalarImage('TestFilled/10' + str(i) + '/10' + str(i) + '_ED_seg.nii'),
            es_seg = tio.ScalarImage('TestFilled/10' + str(i) + '/10' + str(i) + '_ES_seg.nii'),
        )
        subject_list.append(subject)

    for i in range(10, 51):
        subject = tio.Subject(
            ed=tio.ScalarImage('TestFilled/1' + str(i) + '/1' + str(i) + '_ED.nii'),
            es = tio.ScalarImage('TestFilled/1' + str(i) +'/1' + str(i) + '_ES.nii'),
            ed_seg=tio.ScalarImage('TestFilled/1' + str(i) +'/1' + str(i) + '_ED_seg.nii'),
            es_seg = tio.ScalarImage('TestFilled/1' + str(i) +'/1' + str(i) + '_ES_seg.nii'),
        )
        subject_list.append(subject)

    return subject_list

def load_test_dataset(): #load all 50 subjects from test dataset inside a list of torchio subjects
    subject_list = []              
    for i in range(1, 10):          
        subject = tio.Subject(
            ed=tio.ScalarImage('Test/10' + str(i) + '/10' + str(i) + '_ED.nii'),
            es = tio.ScalarImage('Test/10' + str(i) + '/10' + str(i) + '_ES.nii'),
            ed_seg=tio.ScalarImage('Test/10' + str(i) + '/10' + str(i) + '_ED_seg.nii'),
            es_seg = tio.ScalarImage('Test/10' + str(i) + '/10' + str(i) + '_ES_seg.nii'),
        )
        subject_list.append(subject)

    for i in range(10, 51):
        subject = tio.Subject(
            ed=tio.ScalarImage('Test/1' + str(i) + '/1' + str(i) + '_ED.nii'),
            es = tio.ScalarImage('Test/1' + str(i) +'/1' + str(i) + '_ES.nii'),
            ed_seg=tio.ScalarImage('Test/1' + str(i) +'/1' + str(i) + '_ED_seg.nii'),
            es_seg = tio.ScalarImage('Test/1' + str(i) +'/1' + str(i) + '_ES_seg.nii'),
        )
        subject_list.append(subject)

    return subject_list



def extract_ed_thickness(subject_list): # format max(mean), std(mean), mean(std), std(std)
    ed_thickness = np.zeros((len(subject_list), 4))

    for i in range(len(subject_list)):

        subject = subject_list[i].ed_seg.data[0]
        mask = np.where(subject == 2, True, False)
        image = np.where(mask, subject, 0)

        distances = np.zeros((image.shape[2], image.shape[0], image.shape[1]))
        
        for j in range(image.shape[2]):
            distances[j] = distance_transform_edt(image[:, :, j])

        ed_thickness[i, 0] = distances.mean(axis=0).max()
        ed_thickness[i, 1] = distances.mean(axis=0).std()
        ed_thickness[i, 2] = distances.std(axis=0).mean()
        ed_thickness[i, 3] = distances.std(axis=0).std()
    
    return ed_thickness

def extract_thickness_subject(subject_seg_data_numpy):
    interior, exterior = thickness_myocardium(subject_seg_data_numpy)
    #Sagitall axis
    #Moyenne épaisseur myocardium pour chaque slice
    mean_SA = []
    std_SA = []
    max_SA = []
    for i in range(interior.shape[3]):
        interior_indices = np.where(interior[0,:,:,i] == 1)
        exterior_indices = np.where(exterior[0,:,:,i] == 1)
        #Compute the min distance between the interior and exterior contours
        distances = []
        for j in range(len(interior_indices[0])):
            mini = np.inf
            for k in range(len(exterior_indices[0])):
                dist = np.sqrt((interior_indices[0][j]-exterior_indices[0][k])**2+(interior_indices[1][j]-exterior_indices[1][k])**2)
                if dist < mini:
                    mini = dist
            distances.append(mini)
        if len(distances) != 0:

            std_SA.append(np.std(distances))
            mean_SA.append(np.mean(distances))
            max_SA.append(np.max(distances))
    
    return np.max(mean_SA), np.std(mean_SA), np.mean(std_SA), np.std(std_SA), np.mean(mean_SA), np.max(max_SA)


def thickness_myocardium(subject_seg_data_numpy):
    mask_myocardium = np.array(subject_seg_data_numpy == 2)
    mask_left_ventricule = np.array(subject_seg_data_numpy == 3)

    interior = np.zeros(mask_myocardium.shape)
    exterior = np.zeros(mask_myocardium.shape)
    #Axial axis
    for i in range(mask_myocardium.shape[3]):
        mask = np.array(mask_myocardium[0,:,:,i])
        mask = mask.astype(np.uint8)
        
        distance_matrix = distance_transform_edt(mask)
        contours = np.where(distance_matrix==1)
        for m in range(len(contours[0])):
            if neighbour_left_ventricule(mask_left_ventricule[0,:,:,i],contours[0][m],contours[1][m]):
                interior[0, contours[0][m], contours[1][m], i] = 1
            else:
                exterior[0, contours[0][m], contours[1][m], i] = 1
    
    #Sagittal axis
    for i in range(mask_myocardium.shape[1]):
        mask = np.array(mask_myocardium[0,i,:,:])
        mask = mask.astype(np.uint8)
        
        distance_matrix = distance_transform_edt(mask)
        contours = np.where(distance_matrix==1)
        for m in range(len(contours[0])):
            if neighbour_left_ventricule(mask_left_ventricule[0,i,:,:],contours[0][m],contours[1][m]):
                interior[0, i, contours[0][m], contours[1][m]] = 1
            else:
                exterior[0, i, contours[0][m], contours[1][m]] = 1
            
    #Coronal axis
    for i in range(mask_myocardium.shape[2]):
        mask = np.array(mask_myocardium[0,:,i,:])
        mask = mask.astype(np.uint8)
        
        distance_matrix = distance_transform_edt(mask)
        contours = np.where(distance_matrix==1)
        for m in range(len(contours[0])):
            if neighbour_left_ventricule(mask_left_ventricule[0,:,i,:],contours[0][m],contours[1][m]):
                interior[0, contours[0][m], i, contours[1][m]] = 1
            else:
                exterior[0, contours[0][m], i, contours[1][m]] = 1

    return interior, exterior

def neighbour_left_ventricule(mask, i, j):
    if i != 0 and mask[i-1,j] == 1:
        return True
    if i != mask.shape[0]-1 and mask[i+1,j] == 1:
        return True
    if j != 0 and mask[i,j-1] == 1:
        return True
    if j != mask.shape[1]-1 and mask[i,j+1] == 1:
        return True
    return False

def extract_thickness_all(subject_list):

    all_thickness = np.zeros((len(subject_list), 12))

    for i in range(len(subject_list)):

        print("computing subject {}".format(i))
        es_seg = subject_list[i].es_seg.data.numpy()
        ed_seg = subject_list[i].ed_seg.data.numpy()

        es_thickness = extract_thickness_subject(es_seg)
        ed_thickness = extract_thickness_subject(ed_seg)
    
        all_thickness[i, :] = np.concatenate((ed_thickness, es_thickness))

    return all_thickness

def extract_thickness_all_test(subject_list):

    all_thickness = np.zeros((len(subject_list), 12))

    for i in range(len(subject_list)):

        print("computing subject {}".format(i))
        es_seg = subject_list[i].es_seg.data.numpy()
        ed_seg = subject_list[i].ed_seg.data.numpy()

        add_es_seg = 3*lv_volume(es_seg[0]==2).astype('uint8')
        es_seg[0] += add_es_seg

        add_ed_seg = 3*lv_volume(ed_seg[0]==2).astype('uint8')
        ed_seg[0] += add_ed_seg

        es_thickness = extract_thickness_subject(es_seg)
        ed_thickness = extract_thickness_subject(ed_seg)
    
        all_thickness[i, :] = np.concatenate((ed_thickness, es_thickness))

    return all_thickness

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


def fit(epochs, model, loss_func, opt, x_train, y_train, x_val, y_val, metric):
    for epoch in range(epochs):


        loss, _, _ = loss_batch(model, loss_func, x_train, y_train, opt)

        result = evaluate(model, loss_func, x_val, y_val, metric)
        val_loss, total, val_metric = result

        print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'.format(epoch+1, epochs, val_loss, metric.__name__, val_metric))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.sum(preds == labels).item() / len(preds)