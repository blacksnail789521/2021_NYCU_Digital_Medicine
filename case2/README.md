# Digital Medicine Case 2 - COVID-19 Detection (to classify the typical / atypical pneumonia)
## Overview
- Datasets
    - DICOM Image Format
    - Training set
      -  400 Non-Pneumonia
      -  400 Typical Pneumonia
      -  400 Atypical Pneumonia
    -  Validation set
      -  50 Non-Pneumonia
      -  50 Typical Pneumonia
      -  50 Atypical Pneumonia
-  Evaluation matrix
    -  F1-score
-  Method
    -  Preprocessing
      -  CLAHE
      -  Semantic segmentation
      -  Data augmentation
    -  Model
      -  ResNet
      -  EEGNet/DeepConvNet
      -  DenseNet
      -  EfficientNet
    -  Ensemble
      -  Soft voting

## Prepare
### Install Environment
```=bash
conda env create -f environment.yml
```

Or use docker image from nvidia-ngc (tensorflow)
```
cd create_container
docker-compose build
. create_model.sh <container_suffix>
```

## Usage
### Semantic segmentation + EEGNet
```=bash
python semantic_segmentation.py
python EEGNet.py
```

