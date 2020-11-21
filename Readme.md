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
Sella (L1) | 12.77 ± 6.59 | 0.46 ± 0.59 | 0.00 | 0.00 | 0.00 | 25.00|
Nasion (L2) | 18.52 ± 5.41 | 0.76 ± 0.98 | 0.00 | 0.00 | 12.50 | 25.00|
Orbitale (L3) | 28.07 ± 6.96 | 1.54 ± 0.94 | 0.00 | 0.00 | 0.00 | 0.00|
Porion (L4) | 28.61 ± 4.70 | 1.66 ± 1.14 | 0.00 | 0.00 | 0.00 | 0.00|
Subspinale (L5) | 19.05 ± 7.62 | 1.45 ± 1.15 | 0.00 | 0.00 | 0.00 | 0.00|
Supramentale (L6) | 26.32 ± 1.68 | 1.51 ± 0.98 | 0.00 | 0.00 | 0.00 | 0.00|
Pogonion (L7) | 24.37 ± 7.37 | 0.62 ± 0.45 | 0.00 | 0.00 | 0.00 | 0.00|
Menton (L8) | 36.98 ± 2.51 | 0.66 ± 0.48 | 0.00 | 0.00 | 0.00 | 12.50|
Gnathion (L9) | 18.10 ± 9.43 | 0.50 ± 0.36 | 0.00 | 0.00 | 0.00 | 0.00|
Gonion (L10) | 12.78 ± 4.57 | 1.43 ± 1.03 | 0.00 | 0.00 | 12.50 | 12.50|
Incision inferius (L11) | 14.61 ± 2.88 | 0.33 ± 0.36 | 0.00 | 0.00 | 0.00 | 0.00|
Incision superius (L12) | 29.74 ± 3.00 | 0.24 ± 0.34 | 0.00 | 12.50 | 12.50 | 50.00|
Upper lip (L13) | 49.13 ± 9.18 | 1.36 ± 0.74 | 0.00 | 0.00 | 0.00 | 0.00|
Lower lip (L14) | 35.90 ± 5.55 | 1.09 ± 0.65 | 0.00 | 0.00 | 0.00 | 0.00|
Subnasale (L15) | 25.47 ± 9.32 | 0.81 ± 0.56 | 0.00 | 0.00 | 0.00 | 0.00|
Soft tissue pogonion (L16) | 46.10 ± 6.94 | 3.29 ± 1.78 | 0.00 | 0.00 | 0.00 | 0.00|
Posterior nasal spine (L17) | 18.90 ± 2.32 | 0.72 ± 0.59 | 0.00 | 0.00 | 0.00 | 0.00|
Anterior nasal spine (L18) | 16.46 ± 2.59 | 0.91 ± 0.82 | 0.00 | 0.00 | 0.00 | 0.00|
Articulare (L19) | 45.40 ± 27.11 | 1.06 ± 1.25 | 0.00 | 0.00 | 0.00 | 12.50|
Average | 26.70 ± 6.62 | 1.07 ± 0.80 | 0.00 | 0.66 | 1.97 | 7.24|
