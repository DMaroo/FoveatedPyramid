# Original Source
Source code for 'Locating Cephalometric X-Ray Landmarks with Foveated Pyramid Attention' by Logan Gilmour and Nilanjan Ray.

# How to reproduce paper on GPU
1. `git clone https://github.com/enymuss/FoveatedPyramid.git`
2. Download dataset from http://www-o.ntust.edu.tw/~cweiwang/ISBI2015/challenge1/
3. Make a folder named `images` and unzip the downloaded dataset inside of it. This would result in the following folder structure:

```
- FoveatedPyramid/
  ...
  - images/
    - RawImage/
      - AnnotationsByMD/
      - Test1Data/
      - Test2Data/
      - TrainingData/
  ...
```
4. Run `python doctors.py` to create .npz files of the errors of doctors for all images (\*\_big\_\*), Test1Data only (\*\_lil\_\*) and Test2Data only (\*\_lil2\_\*)
5. (Optional) To make a quick run, to test that everything works, you can run the following:
```
python train.py 3 && python tester.py 2 && python results.py 1
```

  `python train.py 3` Creates a `.pth` model for each landmark trained on the first 3 images and validated on the next 3 images for 40 epochs. (Change the `num_epochs=40` variable in train.py for quicker run.)

  `python tester.py 2` Iterates over all 19 models created by the previous command and runs the saved models on all images of Test1Data (150 images) and saves the errors.

  `python results.py 1` Loads the error file created by the previous command and the corresponding doctor errors for Test1Data and outputs a Latex table comparing them, with the best result in bold.

6. Run
```
python train.py 1 && python results.py 3
```

  `python train.py 1` Splits the 400 images (TrainingData+Test1Data+Test2Data) into 4-folds of 100 images each. For each fold, it trains the model and saves it. This is repeated for all 19 landmarks. This results in 76 (4*19) .pth models and 76 .npz files. The model settings are the same as in the code and published paper.

  `python results.py 3`  Loads the created .npz files and doctor errors and outputs a comparison table in LaTex.

Sample Output:

| Landmark | PEL (mm) | IOV (mm) | SDR 2.0mm | SDR 2.5mm | SDR 3.0mm | SDR 4.0mm
| - | - | - | - | - | - | -
Sella (L1) | 0.60 ± 0.83 | **0.46** ± 0.59 | 98.50 | 99.50 | 99.50 | 99.50|
Nasion (L2) | 0.99 ± 1.13 | **0.76** ± 0.98 | 87.50 | 90.75 | 92.75 | 96.25|
Orbitale (L3) | **1.20** ± 1.14 | 1.54 ± 0.94 | 81.25 | 86.75 | 90.75 | 95.75|
Porion (L4) | **1.56** ± 1.78 | 1.66 ± 1.14 | 79.25 | 86.75 | 90.00 | 92.25|
Subspinale (L5) | 1.48 ± 1.12 | **1.45** ± 1.15 | 77.00 | 84.00 | 89.50 | 95.50|
Supramentale (L6) | **1.07** ± 0.73 | 1.51 ± 0.98 | 88.75 | 95.75 | 97.25 | 99.25|
Pogonion (L7) | 0.98 ± 0.70 | **0.62** ± 0.45 | 89.75 | 95.75 | 98.75 | 100.00|
Menton (L8) | 0.80 ± 0.64 | **0.66** ± 0.48 | 94.75 | 96.50 | 98.50 | 100.00|
Gnathion (L9) | 0.82 ± 0.71 | **0.50** ± 0.36 | 94.50 | 97.75 | 98.75 | 99.25|
Gonion (L10) | 1.47 ± 1.17 | **1.43** ± 1.03 | 76.00 | 85.25 | 91.25 | 96.50|
Incision inferius (L11) | 0.55 ± 0.74 | **0.33** ± 0.36 | 95.50 | 96.75 | 98.00 | 99.25|
Incision superius (L12) | 0.47 ± 0.84 | **0.24** ± 0.34 | 95.00 | 96.50 | 97.25 | 98.75|
Upper lip (L13) | 1.44 ± 0.76 | **1.36** ± 0.74 | 77.75 | 89.25 | 96.50 | 100.00|
Lower lip (L14) | **1.08** ± 0.65 | 1.09 ± 0.65 | 89.25 | 96.25 | 98.50 | 100.00|
Subnasale (L15) | 1.06 ± 0.87 | **0.81** ± 0.56 | 90.75 | 94.50 | 96.50 | 98.25|
Soft tissue pogonion (L16) | **1.25** ± 1.10 | 3.29 ± 1.78 | 83.00 | 89.50 | 92.75 | 97.75|
Posterior nasal spine (L17) | 0.93 ± 0.83 | **0.72** ± 0.59 | 93.50 | 96.50 | 97.25 | 98.50|
Anterior nasal spine (L18) | 1.34 ± 1.18 | **0.91** ± 0.82 | 78.00 | 84.25 | 89.25 | 94.50|
Articulare (L19) | **0.98** ± 1.23 | 1.06 ± 1.25 | 91.00 | 94.75 | 96.50 | 98.00|
Average | **1.06** ± 0.95 | 1.07 ± 0.80 | 87.42 | 92.47 | 95.24 | 97.86|

# How to run models on WUT-ML Cephalo dataset with GPU
## Glossary
ISBI means dataset from ISBI challange: 400 images and 19 landmarks.

Cephalo is short for Cephalomateric and refers to WUT-ML private dataset of 1000+ images, each with 20 landmarks. There is an overlap of 11 landmarks between the two datasets, it is those 11 models we will run.

## Preparing The Data

1. `git clone https://github.com/enymuss/FoveatedPyramid.git`
2. You should have a folder called `Models` with 44 models, created by following "How to reproduce paper on GPU" above.
3. Place Cephalo dataset into folder `data/2304/`. `data/2304/images/` should have 1192 images. Obtaining those images is left as an exercise to the reader.

The resulting file structure is as follows:
```
- FoveatedPyramid/
  ...
  - Models/
    - big_hybrid_{isbi_landmark_number}_{fold_num}.pt
    ...
  - data/
    - 2304/
      - cephalo_landmarks.csv
      - images/
  ...
```
## Data Augmentation
4. Run
```
python src/preprocessing/data_augmentation.py 0
```

  This does two things. Firstly, it takes all images in `data/2304/images`, rescales them to have 512px height with the same aspect ratio, and saves the copy of the image in `data/512/images/`.

  Next, it creates 8 augmented images for each image in `data/512/images/` and saves them to `data/512/augmented_images/`.

Result:
```
- FoveatedPyramid/
...
  - data/
    - 512/
      - augmented_images/
      - cephalo_landmarks.csv
      - images/
    - 2304/
      - cephalo_landmarks.csv
      - images/
...
```

5. Move all images from  `512/augmented_images` to `512/images`
```
find data/512/augmented_images/ -name '*.jpg' -exec mv {} data/512/images/ \;
```

## Training Models

6. (Optional) `python CephaloXrayData.py` should plot one 512px image with the Sella Landmark (Landmark 1 in ISBI dataset).

7. (Optional) To make a quick run, to test that everything works, you can run the following:
```
python train.py 6 && python tester.py 4 && python cephaloResults.py 2
```

8. Run the whole thing.

  a) `python train.py 5 && python cephaloResults.py 3`

  b) `python train.py 7 && python cephaloResults.py 3`

  `python train.py 5` does the following: It loads each model for which the landmark exists in Cephalo dataset. It trains it on the 512px images using 4-fold cross validation.

  `python train.py 7` does the same thing, but creates the model from scratch.

  1 fold has 2682 images. The images are taken in sequence, so there is no overlap between train and validation dataset as 2682 images in sequence mean there are 2682/9=298 original images and it's augmentations.

  The settings are as close as possible to the 4-fold cross validation done above on ISBI dataset.

  `python cephaloResults.py 3` takes the .npz files created by the first command and prints a Latex table.

9. (Optional) Visualise Some Cepahlo Images and Landmarks. Run:
```
python tester.py 3 && python cephaloResults.py 1
```
  This plots 4 images with Sella Landmark, ground truth and predictions.


### Notes
List image sizes with: `identify -format "%i: %wx%h\n" *.jpg`
These images have 2256x2304 instead of 2260x2304 size
```
rm 1234.jpg 1240.jpg 134.jpg 159.jpg 188.jpg 254.jpg 435.jpg 608.jpg 609.jpg 759.jpg 769.jpg 779.jpg 938.jpg 1107.jpg
```
