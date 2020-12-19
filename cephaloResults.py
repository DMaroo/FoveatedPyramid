import numpy as np
from math import sqrt
import sys
import matplotlib.pyplot as plt
import CephaloXrayData
import cephaloConstants

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

if ver == 2:

    with open('results_lil_1.npz', 'rb') as f:
        res = np.load(f)['arr_0']
    predictions = res

elif ver==3:
    all_errors = []
    pnt_tuples = cephaloConstants.filter_and_sort_isbi_to_cephalo_mapping()
    for run in range(4):
        errors = []
        for pnt_tuple in pnt_tuples:
            (name, isbi_pnt, cephalo_pnt) = pnt_tuple
            with open(f'Results/big_hybrid_{isbi_pnt}_{run}.npz', 'rb') as f:
                err = np.load(f)['arr_0']
                errors.append(err)
        all_errors.append(np.stack(errors))
    res = np.stack(all_errors)
    predictions = res

elif ver==4:
    all_errors = []
    pnt_tuples = cephaloConstants.cephalo_landmarks()
    for run in range(4):
        errors = []
        for pnt_tuple in pnt_tuples:
            (name, cephalo_pnt) = pnt_tuple
            try:
                with open(f'Results/big_cephalo_{cephalo_pnt}_{run}.npz', 'rb') as f:
                    # import pdb; pdb.set_trace()
                    err = np.load(f)['arr_0']
                    errors.append(err)
            except EnvironmentError:
                print(f"Model for landmark {cephalo_pnt}, run {run}, does not exist. Exiting")
                break
        if errors:
            all_errors.append(np.stack(errors))

    res = np.stack(all_errors)
    predictions = res







names = [f"{x[0]}, Cephalo: L{x[1]}" for x in cephaloConstants.cephalo_landmarks()]
# names = [
# "Sella (L1)",
# "Nasion (L2)",
# "Orbitale (L3)",
# "Porion (L4)",
# "Subspinale (L5)",
# "Supramentale (L6)",
# "Pogonion (L7)",
# "Menton (L8)",
# "Gnathion (L9)",
# "Gonion (L10)",
# "Incision inferius (L11)",
# "Incision superius (L12)",
# "Upper lip (L13)",
# "Lower lip (L14)",
# "Subnasale (L15)",
# "Soft tissue pogonion (L16)",
# "Posterior nasal spine (L17)",
# "Anterior nasal spine (L18)",
# "Articulare (L19)",]

str = "Landmark & PEL (mm) & SDR 2.0mm & SDR 2.5mm & SDR 3.0mm & SDR 4.0mm\\\\\n"

str+="\\hline\n"

res = res.transpose(1,0,2,3)
predictions = predictions.transpose(1,0,2,3)

print(res.shape)

def rescale_point_to_original_size(point):
    middle = np.array([cephaloConstants.IMG_SIZE_ROUNDED_TO_64['width'], cephaloConstants.IMG_SIZE_ROUNDED_TO_64['height']]) / 2
    return ((point*cephaloConstants.IMG_SIZE_ROUNDED_TO_64['width'])/2) + middle

if ver==1:
    indices = range(0, 150, 9)
    indicies_to_plot = indices[4:8]
    val_xrays = CephaloXrayData.TransformedXrays(indices=indicies_to_plot, landmarks=[cephaloConstants.cephalo_landmarks()[0][1]])
    fig = plt.figure()
    for i, xray in enumerate(val_xrays):
        one_predicted_point = predictions[0][0][indicies_to_plot[i]]
        recreated_points = rescale_point_to_original_size(one_predicted_point)
        recreated_points_gt = rescale_point_to_original_size(xray[1])
        ax = plt.subplot(len(val_xrays)/2, 2, i+1)
        plt.tight_layout()
        ax.set_title(f"Image Index: {indicies_to_plot[i]}, Error: {res[0][0][indicies_to_plot[i]][0]:2.4f}")
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
# res = np.reshape(res, (11, 4, 150, 1))

for i, r in enumerate(res):
    rm = f"{r.mean():2.2f}"

    str += f"{names[i]} & {rm} $\\pm$ {(r.std(1)).mean():2.2f} & {((r < 2).sum(1) / numel * 100).mean():3.2f} & {((r < 2.5).sum(1) / numel * 100).mean():3.2f} & {((r < 3).sum(1) / numel * 100).mean():3.2f} & {((r < 4).sum(1) / numel * 100).mean():3.2f}\\\\\n"
str+="\hline\n"
str+=f"Average & \\textbf{{{res.mean():2.2f}}} $\\pm$ {(res.std(2)).mean():2.2f} & {((res<2).sum(2)/numel*100).mean():3.2f} & {((res<2.5).sum(2)/numel*100).mean():3.2f} & {((res<3).sum(2)/numel*100).mean():3.2f} & {((res<4).sum(2)/numel*100).mean():3.2f}\\\\\n"

print(str)
