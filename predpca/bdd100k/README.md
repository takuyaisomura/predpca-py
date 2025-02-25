# BDD100K
This directory demonstrates the application of PredPCA on the BDD100K naturalistic driving video dataset.
The code here is based on what was used to generate Figure 4 in the paper.

## Dataset download
1. Download the dataset (~**380GB**) from "Video Parts" button in http://bdd-data.berkeley.edu/download.html  
We use the following files:
    - `bdd100k_videos_test_00.zip`
    - `bdd100k_videos_train_00.zip` - `bdd100k_videos_train_19.zip`
2. Extract the downloaded files and place them in `bdd100k/data/` directory
3. Resize and crop the videos using `bdd100k/data/process_video_parts.sh`


## Run
### main_preprocess.py
This script generates compressed input data fed to PredPCA.
```bash
python main_preprocess.py
```

### main_train.py
This script runs PredPCA to learn the optimal weight matrices.
```bash
python main_train.py
```

### main_test.py
This script predicts 0.5 second future of test data.
```bash
python main_test.py
```

### main_analyze_categorical_features.py
This script extracts categorical features from the estimated states.
```bash
python main_analyze_categorical_features.py
```
![predpca_ica_cat100](https://github.com/user-attachments/assets/1e02be01-6a67-4d8a-9dc7-8f7f0b015e7e)  
100 major categorical features ($\overline{\mathbf{x}}_t$) representing different categories of scenes.

![predpca_cat_analysis](https://github.com/user-attachments/assets/2aa80e17-86f1-4ff6-a550-f0ba8ba68223)  
PC1–PC3 of the categorical features representing the brightness and vertical and lateral symmetries of scenes.

### main_plot_dynamical_features.py
This script extracts dynamical features from the estimated states.
```bash
python main_plot_dynamical_features.py
```
![predpca_ica_dyn100](https://github.com/user-attachments/assets/94286930-b4c4-4e8e-ac0d-80f0e4983d36)  
100 major dynamical features ($\Delta \mathbf{x}_{t+3 \mid t}$) responding to motions at different positions of the screen.  
The white areas indicate the receptive field of each encoder.

<img src="https://github.com/user-attachments/assets/15d5010d-8979-443e-b2ae-b8776771a427" height=300 alt="predpca_dyn_analysis"><br>
PC1 of the dynamical features representing the lateral motion.

