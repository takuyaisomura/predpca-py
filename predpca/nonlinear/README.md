# Demo: Nonlinear time series prediction
This directory demonstrates the application of PredPCA on a nonlinear time series prediction task.  
It is based on the code that generated Supplementary Fig.1 in the paper.


## Run
### main.py
```bash
python -m predpca.nonlinear.main  # if you installed from PyPI
```
```bash
python main.py  # if you installed from source
```
Edit the `sequence_type` argument in the `main()` function to switch the type of sequence:
- 1: Canonical nonlinear system
- 2: Lorenz attractor

<img src="https://github.com/user-attachments/assets/cefa340f-046e-47f7-82ed-09e7c887ae57" width="600" alt="states">
<img src="https://github.com/user-attachments/assets/0c9885f9-d5bb-47be-b616-508ccec59fc2" width="330" alt="states_zoom">

<img src="https://github.com/user-attachments/assets/b2d48bce-b176-4ffd-a863-4053b979f0f6" width="600" alt="states">
<img src="https://github.com/user-attachments/assets/07d3b776-bb0c-448c-94a4-8f075cb0ebc8" width="330" alt="states_zoom">

Reconstruction of nonlinear dynamical systems by PredPCA.

