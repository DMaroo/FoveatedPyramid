import numpy as np
from math import sqrt
import sys
import matplotlib.pyplot as plt
import CephaloXrayData

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

def show_landmarks(image, landmarks, ground_truth=None):
    """Show image with landmarks"""
    plt.imshow(image, cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r', label="Prediction")
    if ground_truth is not None:
        plt.scatter(ground_truth[:, 0], ground_truth[:, 1], s=10, marker='.', c='b', label="Ground Truth")
    # plt.figlegend('', ('Red', 'Blue'), 'center left')
    plt.pause(0.001)  # pause a bit so that plots are updated

ver = int(sys.argv[1])

if ver==0:
    predictions = [ ]
    with open('cephalo_result.npz', 'rb') as f:
        res = np.load(f)['arr_0']

    with open('cephalo_predictions.npz', 'rb') as f:
        predictions = np.load(f)['arr_0']

if ver==1:
    predictions = [ ]
    with open('cephalo_result.npz', 'rb') as f:
        res = np.load(f)['arr_0']

    with open('cephalo_predictions.npz', 'rb') as f:
        predictions = np.load(f)['arr_0']



names = [
"Sella (L1)",
"Nasion (L2)",
"Orbitale (L3)",
"Porion (L4)",
"Subspinale (L5)",
"Supramentale (L6)",
"Pogonion (L7)",
"Menton (L8)",
"Gnathion (L9)",
"Gonion (L10)",
"Incision inferius (L11)",
"Incision superius (L12)",
"Upper lip (L13)",
"Lower lip (L14)",
"Subnasale (L15)",
"Soft tissue pogonion (L16)",
"Posterior nasal spine (L17)",
"Anterior nasal spine (L18)",
"Articulare (L19)",]

str = "Landmark & PEL (mm) & SDR 2.0mm & SDR 2.5mm & SDR 3.0mm & SDR 4.0mm\\\\\n"

str+="\\hline\n"

res = res.transpose(1,0,2,3)
predictions = predictions.transpose(1,0,2,3)

print(res.shape)

def rescale_point_to_original_size(point):
    middle = np.array([IMG_SIZE_ROUNDED_TO_64['width'], IMG_SIZE_ROUNDED_TO_64['height']]) / 2
    return ((point*IMG_SIZE_ROUNDED_TO_64['width'])/2) + middle

if ver==1:
    indices = range(150)
    indicies_to_plot = indices[4:8]
    val_xrays = CephaloXrayData.TransformedXrays(indices=indicies_to_plot, landmarks=[ISBI_TO_CEPHALO_MAPPING["Sella"]['cephalo']])
    fig = plt.figure()
    for i, xray in enumerate(val_xrays):
        one_predicted_point = predictions[0][0][i]
        recreated_points = rescale_point_to_original_size(one_predicted_point)
        recreated_points_gt = rescale_point_to_original_size(xray[1])
        ax = plt.subplot(len(val_xrays)/2, 2, i+1)
        plt.tight_layout()
        ax.set_title(f"Image Index: {indicies_to_plot[i]}, Error: {res[0][0][i][0]:2.4f}")
        plot_dict = {
        'image': xray[0].numpy().transpose((1, 2, 0)),
        'landmarks': np.expand_dims(recreated_points, 0),
        'ground_truth': recreated_points_gt
        }
        plt.legend()
        show_landmarks(**plot_dict)

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'upper center')
    plt.show()

numel = res.shape[2] #numel = number_of_elements
for i, r in enumerate(res):
    rm = f"{r.mean():2.2f}"

    str += f"{names[i]} & {rm} $\\pm$ {(r.std(1)).mean():2.2f} & {((r < 2).sum(1) / numel * 100).mean():3.2f} & {((r < 2.5).sum(1) / numel * 100).mean():3.2f} & {((r < 3).sum(1) / numel * 100).mean():3.2f} & {((r < 4).sum(1) / numel * 100).mean():3.2f}\\\\\n"
str+="\hline\n"
str+=f"Average & \\textbf{{{res.mean():2.2f}}} $\\pm$ {(res.std(2)).mean():2.2f} & {((res<2).sum(2)/numel*100).mean():3.2f} & {((res<2.5).sum(2)/numel*100).mean():3.2f} & {((res<3).sum(2)/numel*100).mean():3.2f} & {((res<4).sum(2)/numel*100).mean():3.2f}\\\\\n"

print(str)
