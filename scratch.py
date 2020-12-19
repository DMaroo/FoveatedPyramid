import numpy as np
from math import sqrt
import sys
import cephaloConstants
import XrayData
import matplotlib.pyplot as plt

IMG_SIZE_ORIGINAL = {'width': 1935, 'height': 2400}
IMG_SIZE_ROUNDED_TO_64 = {'width': 1920, 'height': 2432}
IMG_TRANSFORM_PADDING = {'width': IMG_SIZE_ROUNDED_TO_64['width'] - IMG_SIZE_ORIGINAL['width'],
                        'height': IMG_SIZE_ROUNDED_TO_64['height']- IMG_SIZE_ORIGINAL['height']}

def show_landmarks(image, landmarks, ground_truth=None):
    """Show image with landmarks"""
    plt.imshow(image, cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r', label="Prediction")
    if ground_truth is not None:
        plt.scatter(ground_truth[:, 0], ground_truth[:, 1], s=10, marker='.', c='b', label="Ground Truth")
    # plt.figlegend('', ('Red', 'Blue'), 'center left')
    plt.pause(0.001)  # pause a bit so that plots are updated

def rescale_point_to_original_size(point):
    middle = np.array([IMG_SIZE_ROUNDED_TO_64['width'], IMG_SIZE_ROUNDED_TO_64['height']]) / 2
    return ((point*IMG_SIZE_ROUNDED_TO_64['width'])/2) + middle

predictions = [ ]
with open('isbi_result.npz', 'rb') as f:
    res = np.load(f)['arr_0']

with open('isbi_predictions.npz', 'rb') as f:
    predictions = np.load(f)['arr_0']


print(res)
print(predictions)

# import pdb; set_trace()

res = res.transpose(1,0,2,3)
predictions = predictions.transpose(1,0,2,3)

indices = range(0, 10)
indicies_to_plot = indices[4:8]
pnt_tuples = cephaloConstants.filter_and_sort_isbi_to_cephalo_mapping()[:1]
(name, isbi_pnt, cephalo_pnt) = pnt_tuples[0]
val_xrays = XrayData.TransformedXrays(indices=indicies_to_plot, landmarks=[isbi_pnt])
fig = plt.figure()
# breakpoint()
for i, xray in enumerate(val_xrays):
    one_predicted_point = predictions[0][0][indicies_to_plot[i]]
    recreated_points = rescale_point_to_original_size(one_predicted_point)
    recreated_points_gt = rescale_point_to_original_size(xray[1])
    #OG Image Plot
    ax = plt.subplot(len(val_xrays)/2, 2, i+1)
    plt.tight_layout()
    ax.set_title(f"OG Image Index: {indicies_to_plot[i]}, Error: {res[0][0][indicies_to_plot[i]][0]:2.4f}")
    plot_dict = {
    'image': xray[0].numpy().transpose((1, 2, 0)),
    'landmarks': np.expand_dims(recreated_points, 0),
    'ground_truth': recreated_points_gt
    }
    plt.legend()
    show_landmarks(**plot_dict)

    # #Tensor Image Plot
    # ax = plt.subplot(len(val_xrays), 2, i+2)
    # plt.tight_layout()
    # ax.set_title(f"Tensor Image")
    # plot_dict = {
    # 'image': xray[0].permute(1, 2, 0),
    # 'landmarks': np.expand_dims(recreated_points, 0),
    # 'ground_truth': recreated_points_gt
    # }
    # plt.legend()
    # show_landmarks(**plot_dict)

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'upper center')
plt.show()

print(res.shape)
