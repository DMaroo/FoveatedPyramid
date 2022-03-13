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
    # pdb.set_trace()
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

def test(settings, landmarks,fold=3, num_folds =4, fold_size=100, avg_labels=True):
    print("TEST")


    batchsize=2
    device = 'cpu'

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
            # for i in range(batchsize):
                # plt.figure()
                # show_landmarks(inputs[i], rescale_point_to_original_size(avg[i].numpy()), rescale_point_to_original_size(labels_tensor[i].numpy()))
                # plt.show()
            # pdb.set_trace()

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
    return predict_landmarks, errors



if __name__=='__main__':
    import matplotlib.pyplot as plt

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

        elif test_num==4:
            folds_errors = []
            errors = []
            predicted_landmarks = []
            run = 0
            from time import time
            rt = time()
            pnt_tuples = cephaloConstants.filter_and_sort_isbi_to_cephalo_mapping()[:1]

            for pnt in pnt_tuples:
                settings = []
                print('-'*10)
                (name, pnt_isbi, pnt_cephalo) = pnt
                print("Test, Landmark: ", pnt_isbi)
                path = f"Models/single_{pnt_isbi}.pt"
                settings.append({'loadpath': path})

                _, predict_errors = test_cephalo(settings, [pnt_cephalo], fold=1, num_folds=2, fold_size=150)

                errors.append(predict_errors)

            all_errors = np.stack(errors)

            folds_errors.append(all_errors)

            all_folds_errors = np.stack(folds_errors)

            print(all_errors.mean())
            with open(f'results_lil_1.npz', 'wb') as f:
                np.savez(f, all_folds_errors)
            print(time()-rt)

        elif test_num==3:
            print("test number = 3")
            predicted_landmarks = []
            folds_predictions = []
            folds_errors = []
            errors = []

            run = 0
            from time import time
            rt = time()

            pnt_tuples = cephaloConstants.filter_and_sort_isbi_to_cephalo_mapping()[:1]

            for fold in range(1):
                for pnt in pnt_tuples:
                    (name, isbi_pnt, cephalo_pnt) = pnt

                    settings = []
                    print('-'*10)
                    print(f"Test, Name: {name}, ISBI Landmark: {isbi_pnt}, Fold: {fold}", )
                    path = f"Models/big_hybrid_{isbi_pnt}_{fold}.pt"
                    settings.append({'loadpath': path})
                    # test on 1 image from cephalo dataset
                    predict_landmarks, predict_errors = test(settings, [isbi_pnt], fold=0, num_folds=2, fold_size=150)

                    predicted_landmarks.append(predict_landmarks)
                    errors.append(predict_errors)

            all_predictions = np.stack(predicted_landmarks)
            all_errors = np.stack(errors)

            folds_predictions.append(all_predictions)
            folds_errors.append(all_errors)

            all_fold_predictions = np.stack(folds_predictions)
            all_folds_errors = np.stack(folds_errors)

            print(all_fold_predictions)
            print(all_errors.mean())
            with open(f'isbi_predictions.npz', 'wb') as f:
                np.savez(f, all_fold_predictions)
            with open(f'isbi_result.npz', 'wb') as f:
                np.savez(f, all_folds_errors)
            print(f"Total time: {time()-rt}s")

        elif test_num==5:
            print("test number = 5, 1st cephalo landmark on 1st 150 images")
            predicted_landmarks = []
            folds_predictions = []
            folds_errors = []
            errors = []

            run = 0
            from time import time
            rt = time()

            pnt_tuples = cephaloConstants.cephalo_landmarks()[:15]

            for fold in range(1):
                for pnt in pnt_tuples:
                    (name, cephalo_pnt) = pnt

                    settings = []
                    print('-'*10)
                    print(f"Test, Name: {name}, ISBI Landmark: {cephalo_pnt}, Fold: {fold}", )
                    path = f"Models/big_cephalo_{cephalo_pnt}_{fold}.pt"
                    settings.append({'loadpath': path})
                    # test on 1 image from cephalo dataset
                    predict_landmarks, predict_errors = test_cephalo(settings, [cephalo_pnt], fold=0, num_folds=2, fold_size=150)

                    predicted_landmarks.append(predict_landmarks)
                    errors.append(predict_errors)

            all_predictions = np.stack(predicted_landmarks)
            all_errors = np.stack(errors)

            folds_predictions.append(all_predictions)
            folds_errors.append(all_errors)

            all_fold_predictions = np.stack(folds_predictions)
            all_folds_errors = np.stack(folds_errors)

            print(all_fold_predictions)
            print(all_errors.mean())
            with open(f'cephalo_predictions.npz', 'wb') as f:
                np.savez(f, all_fold_predictions)
            with open(f'cephalo_result.npz', 'wb') as f:
                np.savez(f, all_folds_errors)
            print(f"Total time: {time()-rt}s")

        else:
            all_models()

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
