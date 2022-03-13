import XrayData
import CephaloXrayData
import torchvision.transforms as transforms
import numpy as np
from random import randint
import torch
import model as m
import torch.optim as optim
import torch.nn as nn
from time import time
from math import pi
import matplotlib
import copy
import torch.nn.functional as F
from pyramid import pyramid, stack, pyramid_transform
import sys
import cephaloConstants

"""
List image sizes with: identify -format "%i: %wx%h\n" *.jpg
These images have 2256x2304 instead of 2260x2304 size
rm 1234.jpg 1240.jpg 134.jpg 159.jpg 188.jpg 254.jpg 435.jpg 608.jpg 609.jpg 759.jpg 769.jpg 779.jpg 938.jpg 1107.jpg
"""

def all_models():
    folds_errors = []
    for fold in range(4):
        errors = []
        for i in range(19):
            path = f"Models/big_{i}_{fold}.pt"
            errors.append(test([{'loadpath':path}],[i],fold=fold))
        all_errors = np.stack(errors)
        folds_errors.append(all_errors)
    all_folds_errors = np.stack(folds_errors)
    print(all_errors.mean())
    with open(f'results_big.npz', 'wb') as f:
        np.savez(f, all_folds_errors)

IMG_SIZE_ORIGINAL = {'width': 1935, 'height': 2400}
IMG_SIZE_ROUNDED_TO_64 = {'width': 1920, 'height': 2432}
IMG_TRANSFORM_PADDING = {'width': IMG_SIZE_ROUNDED_TO_64['width'] - IMG_SIZE_ORIGINAL['width'],
                        'height': IMG_SIZE_ROUNDED_TO_64['height']- IMG_SIZE_ORIGINAL['height']}

def rescale_point_to_original_size(point):
    middle = np.array([IMG_SIZE_ROUNDED_TO_64['width'], IMG_SIZE_ROUNDED_TO_64['height']]) / 2
    return ((point*IMG_SIZE_ROUNDED_TO_64['width'])/2) + middle

def show_landmarks(image, landmarks, ground_truth=None):
    """Show image with landmarks"""
    plt.imshow(image.permute(1, 2, 0), cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r', label="Prediction")
    if ground_truth is not None:
        plt.scatter(ground_truth[:, 0], ground_truth[:, 1], s=10, marker='.', c='b', label="Ground Truth")
    plt.figlegend('', ('Red', 'Blue'), 'center left')
    plt.pause(0.001)  # pause a bit so that plots are updated

def test_cephalo(settings, landmarks,fold=3, num_folds =4, fold_size=100):
    print("TEST")

    batchsize=2
    levels = 6
    device = 'cpu'
    output_count=len(landmarks)

    # splits, datasets, dataloaders, _ = XrayData.get_folded(landmarks,batchsize=batchsize, fold=fold, num_folds=num_folds, fold_size=fold_size)
    #
    # annos = XrayData.TransformedHeadXrayAnnos(indices=list(range(150)), landmarks=landmarks)

    splits, datasets, dataloaders, _ = CephaloXrayData.get_folded(landmarks,batchsize=batchsize, fold=fold, num_folds=num_folds, fold_size=fold_size)
    annos = CephaloXrayData.TransformedHeadXrayAnnos(indices=list(range(150)), landmarks=landmarks)

    # if avg_labels:
    #     pnts = np.stack(list(map(lambda x: (x[1] + x[2]) / 2, annos)))
    # else:
    pnts = np.stack(list(map(lambda x: x[1], annos)))

    means = torch.tensor(pnts.mean(0, keepdims=True), device=device, dtype=torch.float32)


    models = []
    for setting in settings:
        model = m.PyramidAttention(levels)
        if (device == 'cuda'):
            model.load_state_dict(torch.load(setting['loadpath']))
        else:
            model.load_state_dict(torch.load(setting['loadpath'], map_location=torch.device('cpu')))

        models.append(model)
        model.to(device)
        model.eval()

    criterion = nn.MSELoss(reduction='none')
    # Iterate over data.

    phase='val'
    data_iter = iter(dataloaders[phase])

    next_batch = data_iter.next()  # start loading the first batch

    # with pin_memory=True and async=True, this will copy data to GPU non blockingly
    if (device == 'cuda'):
        next_batch = [t.cuda(non_blocking=True) for t in next_batch]
    else:
        next_batch = [t for t in next_batch]

    start = time()
    errors = []
    predict_landmarks = []
    doc_errors = []
    print("GOT HERE")
    for i in range(len(dataloaders[phase])):
        batch = next_batch
        inputs, doc_labels = batch

        if i + 2 != len(dataloaders[phase]):
            # start copying data of next batch
            next_batch = data_iter.next()
            if (device == 'cuda'):
                next_batch = [t.cuda(non_blocking=True) for t in next_batch]
            else:
                next_batch = [t for t in next_batch]


        inputs_tensor = inputs.to(device)

        # if avg_labels:
        #     labels_tensor = torch.stack((junior_labels, senior_labels), dim=0).mean(0).to(device).to(torch.float32)
        # else:
        labels_tensor = doc_labels.to(device).to(torch.float32)

        # zero the parameter gradients

        pym = pyramid(inputs_tensor, levels)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            all_outputs = []
            for model in models:
                guess = means


                for j in range(10):
                    outputs = guess + model(pym, guess,
                                            phase == 'train')  # ,j==2 and i==0 and phase=='val' and False,rando)
                    guess = outputs.detach()

                all_outputs.append(guess)

            avg = torch.stack(all_outputs,0).mean(0)

            loss = criterion(avg, labels_tensor)

            error = loss.detach().sum(dim=2).sqrt()
            predict_landmarks.append(avg)
            errors.append(error)
            # doc_errors.append(F.mse_loss(junior_labels, senior_labels, reduction='none').sum(dim=2).sqrt())

    errors = torch.cat(errors,0).detach().cpu().numpy()/2*192
    predict_landmarks = torch.cat(predict_landmarks, 0).squeeze().detach().cpu().numpy()
    # doc_errors = torch.cat(doc_errors,0).detach().cpu().numpy()/2*192

    # doc_error = doc_errors.mean(0)
    all_error = errors.mean(0)
    error = errors.mean()
    for i in range(output_count):

        print(f"Error {i}: {all_error[i]}")

    print(f"{phase} loss: {error} in: {time() - start}s")
    return predict_landmarks, errors

if __name__=='__main__':
    import matplotlib.pyplot as plt
    predicted_landmarks = []
    folds_predictions = []
    folds_errors = []
    errors = []

    run = 0
    from time import time
    rt = time()

    pnt_tuples = cephaloConstants.filter_and_sort_isbi_to_cephalo_mapping()
    print(pnt_tuples)

    for pnt in pnt_tuples:
        (name, isbi_pnt, cephalo_pnt) = pnt

        landmarks = pnt_tuples

        settings = []
        print('-'*10)
        print(f"Test, Name: {name}, ISBI Landmark: {isbi_pnt}, Fold: {fold}", )
        path = f"Models/big_hybrid_{isbi_pnt}_{fold}.pt"
        settings.append({'loadpath': path})

        fold = 0
        num_folds = 2
        fold_size = 150
        avg_labels = True
        print("TEST")


        batchsize=1
        device = 'cpu'

        _, _, dataloaders, _ = XrayData.get_folded(landmarks,batchsize=batchsize, fold=fold, num_folds=num_folds, fold_size=fold_size)

        annos = XrayData.TransformedHeadXrayAnnos(indices=list(range(150)), landmarks=landmarks)

        if avg_labels:
            pnts = np.stack(list(map(lambda x: (x[1] + x[2]) / 2, annos)))
        else:
            pnts = np.stack(list(map(lambda x: x[1], annos)))

        means = torch.tensor(pnts.mean(0, keepdims=True), device=device, dtype=torch.float32)

        levels = 6

        output_count=len(landmarks)



        models = []
        for setting in settings:
            model = m.PyramidAttention(levels)
            model.load_state_dict(torch.load(setting['loadpath']))
            models.append(model)
            model.to(device)
            model.eval()

        criterion = nn.MSELoss(reduction='none')
        # Iterate over data.

        phase='val'
        data_iter = iter(dataloaders[phase])
        next_batch = data_iter.next()  # start loading the first batch

        # with pin_memory=True and async=True, this will copy data to GPU non blockingly
        if (device == 'cuda'):
            next_batch = [t.cuda(non_blocking=True) for t in next_batch]
        else:
            next_batch = [t for t in next_batch]

        start = time()
        errors = []
        doc_errors = []
        predict_landmarks = []
        print("GOT HERE")
        for i in range(len(dataloaders[phase])):
            batch = next_batch
            inputs, junior_labels, senior_labels = batch

            if i + 2 != len(dataloaders[phase]):
                # start copying data of next batch
                next_batch = data_iter.next()
                if (device == 'cuda'):
                    next_batch = [t.cuda(non_blocking=True) for t in next_batch]
                else:
                    next_batch = [t for t in next_batch]


            inputs_tensor = inputs.to(device)

            if avg_labels:
                labels_tensor = torch.stack((junior_labels, senior_labels), dim=0).mean(0).to(device).to(torch.float32)
            else:
                labels_tensor = junior_labels.to(device).to(torch.float32)

            # zero the parameter gradients

            pym = pyramid(inputs_tensor, levels)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                # Get model outputs and calculate loss
                all_outputs = []
                for model in models:
                    guess = means


                    for j in range(10):

                        outputs = guess + model(pym, guess,
                                                phase == 'train')  # ,j==2 and i==0 and phase=='val' and False,rando)

                        guess = outputs.detach()

                    all_outputs.append(guess)

                avg = torch.stack(all_outputs,0).mean(0)

                # show landmarks for avg
                for i in range(batchsize):
                    plt.figure()
                    print(avg[i], labels_tensor[i])
                    show_landmarks(inputs[i], rescale_point_to_original_size(avg[i].numpy()), rescale_point_to_original_size(labels_tensor[i].numpy()))
                    plt.show()

                loss = criterion(avg, labels_tensor)

                error = loss.detach().sum(dim=2).sqrt()
                predict_landmarks.append(avg)
                errors.append(error)
                doc_errors.append(F.mse_loss(junior_labels, senior_labels, reduction='none').sum(dim=2).sqrt())

        errors = torch.cat(errors,0).detach().cpu().numpy()/2*192
        predict_landmarks = torch.cat(predict_landmarks, 0).squeeze().detach().cpu().numpy()
        doc_errors = torch.cat(doc_errors,0).detach().cpu().numpy()/2*192

        doc_error = doc_errors.mean(0)
        all_error = errors.mean(0)
        error = errors.mean()
        for i in range(output_count):

            print(f"Error {i}: {all_error[i]} (doctor: {doc_error[i]}")

        print(f"{phase} loss: {error} (doctors: {doc_errors.mean()} in: {time() - start}s")
        predict_errors = errors

        predicted_landmarks.append(predict_landmarks)
        errors.append(predict_errors)
