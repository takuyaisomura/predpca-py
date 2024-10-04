# Demo: ALOI
This directory demonstrates the application of PredPCA on the ALOI 3D rotating object image dataset.  
It is based on the code that generated Fig.3 in the paper.


## Dataset download
Please download the dataset (~1.8GB) from http://aloi.science.uva.nl  
In the "Download" tab, select:
- Full color (24 bit)
- Quarter resolution (192 x 144)
- Viewing direction

and expand `aloi_red4_view.tar` in `aloi/data/` directory.  

Reference to ALOI dataset  
Geusebroek JM, Burghouts GJ, Smeulders AWM, The Amsterdam library of object images, Int J Comput Vision, 61, 103-112 (2005)


## Run
### main_preprocess.py
First, run the data preprocessing script:
```bash
python -m predpca.aloi.main_preprocess  # if you installed from PyPI
```
```bash
python main_preprocess.py  # if you installed from source
```

### main.py
```bash
python -m predpca.aloi.main  # if you installed from PyPI
```
```bash
python main.py  # if you installed from source
```
![predpca_movie](https://github.com/user-attachments/assets/7ed4992e-6214-47a4-88e4-a92b7c2b71f2)  
PredPCA of 90° rotated images. The right-hand-side images are the predictions of the corresponding ground truth images on the left-hand side. 100 examples with small prediction errors among 200 test image sequences are presented.

### main_plot_prediction_error.py
```bash
python -m predpca.aloi.main_plot_prediction_error  # if you installed from PyPI
```
```bash
python main_plot_prediction_error.py  # if you installed from source
```
<img src="https://github.com/user-attachments/assets/a9cd3f6a-e17a-40d3-b83f-198450a5dd3d" height="700" alt="test_prediction_error"><br>
Comparison of test prediction error. PredPCA (solid lines) show a smaller test prediction error and an earlier error convergence compared with the naive autoregressive model (dashed lines).
