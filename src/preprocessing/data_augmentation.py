import numpy as np
import os
import pandas as pd
import glob
import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from imgaug.augmenters import Sequential
import sys

seq = iaa.Sequential([
    iaa.Affine(
        # rotate=iap.Normal(0.0, 20),
        translate_px=iap.RandomSign(iap.Poisson(3))
    ),
    #iaa.Crop(percent=(0, 0.2)),
    iaa.Multiply(iap.Positive(iap.Normal(0.0, 0.4)) + 0.8),
    iaa.ContrastNormalization(iap.Uniform(0.5, 1.5))
], random_order=True)


class DataAugmentator(object):
    """
    Generates new images in form of arrays for the given parameters.
    :param seq: a sequention of different augmenters applied to single augmentatiion call
    :param landmarks_num: number of image's landmarks
    :param batch_size: number of images that would be generated during augmentation for single image
    :param img_dir: directory to images to be augmented
    :param annotation_dir: directory to images' landmarks
    :param mask_dir: directory to images' masks
    :param output_dir: directory for augmented images
    """

    def __init__(self, seq: Sequential = None, landmarks_num: int = None, batch_size: int = None,
                 images_dir: str = None, annotation_dir: str = None, masks_dir: str = None, output_dir: str = None, output_annotation_dir: str = ""):
        self.seq = seq
        self.landmarks_num = landmarks_num
        self.batch_size = batch_size

        self.images_dir = images_dir
        self.annotation_dir = annotation_dir
        self.masks_dir = masks_dir
        self.output_dir = output_dir
        self.output_annotation_dir = output_annotation_dir

        self.landmarks = pd.read_csv(self.annotation_dir, delimiter=",")
        self.landmarks["filename"].sort_values()
        self.images = glob.glob(self.images_dir + "/*.jpg")
        self.images.sort()
        self.masks = glob.glob(self.masks_dir + "/*.tiff")
        self.masks.sort()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if output_annotation_dir:
            if not os.path.exists(self.output_annotation_dir):
                if not os.path.exists(os.path.dirname(self.output_annotation_dir)):
                    os.makedirs(os.path.dirname(self.output_annotation_dir))
            df = pd.DataFrame(columns=self.landmarks.columns)
            df.to_csv(self.output_annotation_dir, index=False)

    def augment_image(self, image, landmarks):
        """
           Generates a batch of augmented images with landmarks from image and its landmarks
           :param image: image which will be augmented
           :param landmarks: numpy coordinates ([x], [x, y] or [x, y, z]) of the landmark point of the image.
           :return: numpy array of the landmark image.
       """
        keypoints = []
        for k in list(range(0, 2 * self.landmarks_num - 1, 2)):
            keypoint = Keypoint(x=(float(landmarks[k])),
                                y=float(landmarks[k + 1]))
            keypoints.append(keypoint)

        keypoints_on_image = KeypointsOnImage(keypoints, shape=image.shape)

        images = [image for _ in range(self.batch_size)]
        keypoints_on_images = [keypoints_on_image for _ in range(self.batch_size)]

        images_aug, kpsois_aug = self.seq(images=images, keypoints=keypoints_on_images)

        return images_aug, kpsois_aug

    def augment_images(self):
        """
           Generates a batch of augmented images with landmarks for images specified in input folder.
           Generates landmarks for augmented images and append them to annotation file.
           If output_annotation_dir is defined, creates new annotation file with new values.
        """
        images_aug_list = []
        kpsois_aug_list = []

        print("Images: ", len(self.images))

        for image in self.images:
            # might be faster to iterate over each element in landmarks, as then you don't have to do find_image_landmarks() each time.
            image_read = ia.cv2.imread(image)
            image_landmarks = self.find_image_landmarks(os.path.basename(image))
            if image_landmarks is None:
                print(f"No image landmarks for {image}")
                continue
            image_name = image_landmarks["filename"]
            print('Augment image: {}'.format(image_name))
            images_aug, kpsois_aug = self.augment_image(image_read, image_landmarks)

            for index, (image, kpsois) in enumerate(zip(images_aug, kpsois_aug), start=1):
                ia.cv2.imwrite(os.path.join(self.output_dir, image_name[:-4] + "_" + str(index) + ".jpg"), image)
                images_aug_list.append(image)
                kpsois_aug_list.append(kpsois)


                output_annotation_path = self.output_annotation_dir if self.output_annotation_dir else self.annotation_dir

                with open(output_annotation_path, 'a', newline='') as file:
                    points_arrays = [kpsoi.xy for kpsoi in kpsois]
                    points = np.concatenate(points_arrays, axis=0)
                    row_of_points = ','.join([repr(num) for num in points])
                    row_of_points += "," + image_name[:-4] + "_" + str(index) + ".jpg" + "\n"
                    file.write(str(row_of_points))

        return images_aug_list, kpsois_aug_list

    def find_image_landmarks(self, image):
        """
        change fixes bug:
        Image path: data/2304/images/1102.jpg
        Augment image: 102.jpg
        """
        for _, landmarks in self.landmarks.iterrows():
            if landmarks["filename"] == image:
                return landmarks
        return None


if __name__ == "__main__":
    da = None
    if len(sys.argv)>1:
        test = int(sys.argv[1])
        if test == 0:
            """
            Scale original images to 512px and move them to new folder
            Create new landmarks csv file for those images
            Perform Augmentation on rescaled images and append to newly created csv
            """
            seq_scale = iaa.Sequential([
                iaa.Resize(size={"height": 512 , "width": "keep-aspect-ratio"})
            ])

            da = DataAugmentator(seq=seq_scale,
                                 landmarks_num=20,
                                 batch_size=1,
                                 images_dir="data/2304/images/",
                                 masks_dir="",
                                 annotation_dir="data/2304/cephalo_landmarks.csv",
                                 output_dir="data/512/images/",
                                 output_annotation_dir="data/512/cephalo_landmarks.csv")

            images_aug_list, kpsois_aug_list = da.augment_images()

            da = DataAugmentator(seq=seq,
                                 landmarks_num=20,
                                 batch_size=8,
                                 images_dir="data/512/images/",
                                 masks_dir="",
                                 annotation_dir="data/512/cephalo_landmarks.csv",
                                 output_dir="data/512/augmented_images/",
                                 output_annotation_dir="")

            images_aug_list, kpsois_aug_list = da.augment_images()

    else:
        da = DataAugmentator(seq=seq,
                             landmarks_num=20,
                             batch_size=20,
                             images_dir="../../data/images/",
                             masks_dir="../../data/masks/",
                             annotation_dir="../../data/cephalolandmarks.csv",
                             output_dir="../../augmented_images/")

        images_aug_list, kpsois_aug_list = da.augment_images()
