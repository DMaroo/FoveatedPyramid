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
import pdb

IMG_SIZE_ORIGINAL = {'width': 2260, 'height': 2304}
IMG_SIZE_ROUNDED_TO_64 = {'width': 2304, 'height': 2304}
IMG_TRANSFORM_PADDING = {'width': IMG_SIZE_ROUNDED_TO_64['width'] - IMG_SIZE_ORIGINAL['width'],
                        'height': IMG_SIZE_ROUNDED_TO_64['height']- IMG_SIZE_ORIGINAL['height']}
ISBI_TO_CEPHALO_MAPPING = {
"Columella"                : {'isbi': None, 'cephalo': 0},
"Subnasale"                : {'isbi': 14, 'cephalo': 1},
"Upper lip"                : {'isbi': 12, 'cephalo': 2},
"Pogonion"                 : {'isbi': 15, 'cephalo': 3},
"Nasion"                   : {'isbi': 1, 'cephalo': 4},
"Anterior nasal spine"     : {'isbi': 17, 'cephalo': 5},
"Subspinale"               : {'isbi': 4, 'cephalo': 6},
"Point B"                  : {'isbi': None, 'cephalo': 7},
"Pogonion"                 : {'isbi': 6, 'cephalo': 8},
"Gnathion"                 : {'isbi': 8, 'cephalo': 9},
"U1 root tip"              : {'isbi': None, 'cephalo': 10},
"U1 incisal edge"          : {'isbi': None, 'cephalo': 11},
"L1 incisal edge"          : {'isbi': None, 'cephalo': 12},
"L1 root tip"              : {'isbi': None, 'cephalo': 13},
"Sella"                    : {'isbi': 0, 'cephalo': 14},
"Articulare"               : {'isbi': 18, 'cephalo': 15},
"Basion"                   : {'isbi': None, 'cephalo': 16},
"Posterior nasal spine"    : {'isbi': 16, 'cephalo': 17},
"Gonion constructed"       : {'isbi': None, 'cephalo': 18},
"Tuberositas messenterica" : {'isbi': None, 'cephalo': 19},
"Orbitale"                 : {'isbi': 2, 'cephalo': None},
"Porion"                   : {'isbi': 3, 'cephalo': None},
"Supramentale"             : {'isbi': 5, 'cephalo': None},
"Menton"                   : {'isbi': 7, 'cephalo': None},
"Gonion"                   : {'isbi': 9, 'cephalo': None},
"Incision inferis"         : {'isbi': 10, 'cephalo': None},
"Incision superius"        : {'isbi': 11, 'cephalo': None},
"Lower lip"                : {'isbi': 13, 'cephalo': None}
}

def all():
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
    # doc_errors = torch.cat(doc_errors,0).detach().cpu().numpy()/2*192

    # doc_error = doc_errors.mean(0)
    all_error = errors.mean(0)
    error = errors.mean()
    for i in range(output_count):

        print(f"Error {i}: {all_error[i]}")

    print(f"{phase} loss: {error} in: {time() - start}s")
    return predict_landmarks, errors

def test(settings, landmarks,fold=3, num_folds =4, fold_size=100, avg_labels=True):
    print("TEST")


    batchsize=2
    device = 'cuda'

    splits, datasets, dataloaders, _ = XrayData.get_folded(landmarks,batchsize=batchsize, fold=fold, num_folds=num_folds, fold_size=fold_size)

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
    print("GOT HERE")
    for i in range(len(dataloaders[phase])):
        batch = next_batch
        inputs, junior_labels, senior_labels = batch
        breakpoint()

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

            loss = criterion(avg, labels_tensor)

            error = loss.detach().sum(dim=2).sqrt()
            errors.append(error)
            doc_errors.append(F.mse_loss(junior_labels, senior_labels, reduction='none').sum(dim=2).sqrt())

    errors = torch.cat(errors,0).detach().cpu().numpy()/2*192
    doc_errors = torch.cat(doc_errors,0).detach().cpu().numpy()/2*192

    doc_error = doc_errors.mean(0)
    all_error = errors.mean(0)
    error = errors.mean()
    for i in range(output_count):

        print(f"Error {i}: {all_error[i]} (doctor: {doc_error[i]}")

    print(f"{phase} loss: {error} (doctors: {doc_errors.mean()} in: {time() - start}s")
    return errors



if __name__=='__main__':
    import matplotlib.pyplot as plt

    def show_landmarks(image, landmarks, ground_truth=None):
        """Show image with landmarks"""
        plt.imshow(image, cmap='gray')
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
        if ground_truth is not None:
            plt.scatter(ground_truth[:, 0], ground_truth[:, 1], s=10, marker='.', c='g')
        plt.pause(0.001)  # pause a bit so that plots are updated

    if len(sys.argv)>1:


        test_num = int(sys.argv[1])

        if test_num==1:
            folds_errors = []
            fold = 1

            errors = []
            run = 0
            from time import time
            rt = time()
            for i in range(19):
                settings = []
                #for run in range(1,2):
                path = f"Models/lil_hybrid_{i}_{run}.pt"
                settings.append({'loadpath': path})
                errors.append(test(settings, [i], fold=3,num_folds=4,fold_size=100))
            all_errors = np.stack(errors)
            folds_errors.append(all_errors)

            all_folds_errors = np.stack(folds_errors)
            print(all_errors.mean())
            with open(f'results_lil_hybrid_test2_{run}.npz', 'wb') as f:
                np.savez(f, all_folds_errors)
            print(time()-rt)

        elif test_num==2:
            folds_errors = []

            errors = []
            run = 0
            from time import time
            rt = time()
            for i in range(19):
                settings = []
                print('-'*10)
                print("Test, Landmark: ", i)
                path = f"Models/single_{i}.pt"
                settings.append({'loadpath': path})
                '''Use all 150 images of TestData1 as the testset'''
                errors.append(test(settings, [i], fold=1,num_folds=2,fold_size=150))
            all_errors = np.stack(errors)
            folds_errors.append(all_errors)

            all_folds_errors = np.stack(folds_errors)
            print(all_errors.mean())
            with open(f'results_lil_1.npz', 'wb') as f:
                np.savez(f, all_folds_errors)
            print(time()-rt)

        elif test_num==3:
            print("test number = 3")
            folds_errors = []
            errors = []
            run = 0
            from time import time
            rt = time()

            for fold in range(1):
                for pnt in range(1):
                    isbi_pnt = ISBI_TO_CEPHALO_MAPPING["Sella"]['isbi']
                    cephalo_pnt = ISBI_TO_CEPHALO_MAPPING["Sella"]['cephalo']

                    settings = []
                    print('-'*10)
                    print(f"Test, ISBI Landmark: {isbi_pnt}, Fold: {fold}", )
                    path = f"Models/big_hybrid_{isbi_pnt}_{fold}.pt"
                    settings.append({'loadpath': path})
                    # test on 1 image from cephalo dataset
                    predict_landmarks, predict_errors = test_cephalo(settings, [cephalo_pnt], fold=1, num_folds=2, fold_size=4)

                    xrays = CephaloXrayData.TransformedXrays(indices=[4], landmarks=[cephalo_pnt])[0]
                    middle = np.array([IMG_SIZE_ROUNDED_TO_64['width'], IMG_SIZE_ROUNDED_TO_64['height']]) / 2
                    one_predicted_point = predict_landmarks[0][0].numpy()

                    recreated_points = ((one_predicted_point*IMG_SIZE_ROUNDED_TO_64['width'])/2) + middle
                    recreated_points_gt = ((xrays[1]*IMG_SIZE_ROUNDED_TO_64['width'])/2) + middle
                    plt.figure()
                    print("ground truth:", xrays[1])
                    print("diff:", xrays[1] - recreated_points)
                    show_landmarks(xrays[0].numpy().transpose((1, 2, 0)), recreated_points, ground_truth=recreated_points_gt)
                    plt.show()
                    errors.append(predict_errors)
            all_errors = np.stack(errors)
            folds_errors.append(all_errors)

            all_folds_errors = np.stack(folds_errors)
            print(all_errors.mean())
            with open(f'results_big.npz', 'wb') as f:
                np.savez(f, all_folds_errors)
            print(f"Total time: {time()-rt}s")

        else:
            all()

    else:
        print(test([{'loadpath':"Models/test.pt"}],[11]).mean())

'''
if __name__ == '__main__':
    errors = []

    #for i in range(19):
    #    errors.append(test([], [i]))

    for i in range(17):
        errors.append(test([{'loadpath':f"Models_seed_10/single_{i}.pt",'fuckup_override':1},
                            {'loadpath': f"Models_seed_20/single_{i}.pt", 'fuckup_override': 1},
                            {'loadpath': f"Models_seed_100/single_{i}.pt", 'fuckup_override': 2},
                            {'loadpath': f"Models_seed_30/single_{i}.pt", 'fuckup_override': 0}
                            ], [i]))
    for i in range(17,19):
        errors.append(test([
            {'loadpath':f"Models_seed_10/single_{i}.pt",'fuckup_override':1},
            {'loadpath': f"Models_seed_20/single_{i}.pt", 'fuckup_override': 1},
             {'loadpath': f"Models_seed_100/single_{i}.pt", 'fuckup_override': 0},

            {'loadpath': f"Models_seed_30/single_{i}.pt", 'fuckup_override': 0},
                            ], [i]))

    all_errors = np.stack(errors)
    print(all_errors.mean())
    with open('results_ensemble.npz', 'wb') as f:
        np.savez(f, all_errors)
'''
