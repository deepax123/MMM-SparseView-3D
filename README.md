
# MMM-SparseView-3D

This repository contains the official implementation of the paper:

“Multi-scale Generative Modelling for Enhanced Sparse-View 3D Scene Reconstruction”  
submitted to The Visual Computer.

If you use this code, please cite the corresponding manuscript.

DOI: https://doi.org/10.5281/zenodo.18541704

---

## Requirements

Python 3.8  
CUDA 11.7  
GPU: NVIDIA RTX 4090  

Install dependencies:
pip install -r requirements.txt

---

## Dataset

This project uses the DTU Multi-View Stereo dataset.


  ```
  - dtu/
  - scan1 
    - images
      - 00000000.jpg
      - 00000001.jpg
      - ...
    - cams_1
      - 00000000_cam.txt
      - 00000001_cam.txt
      - ...
    - pair.txt
  ```

Official download:https://github.com/JianfeiJ/DI-MVS
we refered to this repository for dataset

---

## Training

python train.py --config configs/dtu_3view.yaml

---

## Evaluation

python eval.py --config configs/dtu_3view.yaml
