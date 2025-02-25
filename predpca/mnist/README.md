# MNIST
This directory demonstrates the application of PredPCA on the MNIST handwritten digit image sequence.  
The code here is based on what was used to generate Figure 2 in the paper.

## Dataset download
Please download the following files (~11MB) and place them in `mnist/data/` directory:
- https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
- https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
- https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
- https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz


## Run

### main_plot_encodings.py
```bash
python main_plot_encodings.py
```
<img src="https://github.com/user-attachments/assets/0b1f4997-a92c-47f9-92ce-d64f87b6e293" height="700" alt="encodings"><br>
10-dimensional independent encoders (hidden state estimator) $x_{t+1 \mid t}$ obtained using PredPCA and ICA.  
20,000 test samples that are colour-coded by their digit are plotted.

### main_plot_prediction_error.py
```bash
python main_plot_prediction_error.py
```
![prediction_error_seed0](https://github.com/user-attachments/assets/ec237380-630f-42ff-a1b2-c5a18d2d0195)  
(Left) Parameter estimation error measured by the squared Frobenius norm ratio.  
(Right) Test error in predicting the next handwritten digit images in the ascending-order sequence, measured by the normalized mean squared error over test samples.

### main_plot_predicted_images.py
```bash
python main_plot_predicted_images.py
```
![output_ascending_predpca1_0_99991_100000](https://github.com/user-attachments/assets/b71b9e8c-f1b8-4fca-b8e8-feb2fc8e0f47)  
![output_fibonacci_predpca2_0_99991_100000](https://github.com/user-attachments/assets/3b687ee8-dc1f-4a26-87a2-619a4badb543)  
Long-term prediction using PredPCA and ICA for the ascending (top) and Fibonacci (bottom) sequence (at T = 100,050 - 100,060).  
A winner-takes-all operation is applied to make greedy predictions of the digit sequences.

### main_compare_models.py
```bash
python main_compare_models.py
```
Comparison of categorization error rates (%) between different models. The models compared include:
- PredPCA
- AE: Standard Auto-Encoder
- VAE: Variational Auto-Encoder
- TAE: Time-lagged Auto-Encoder
- LTAE: Linear Time-lagged Auto-Encoder
- TICA: Time-lagged Independent Component Analysis

<img src="https://github.com/user-attachments/assets/4c0d98c9-63d6-4e62-a236-7e5c9ee00532" width="500" alt="model comparison"><br>
The results show performance on both ascending and Fibonacci sequences.  
Lower error rates indicate better performance in predicting and categorizing the digit sequences.
