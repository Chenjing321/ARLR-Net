
# ARLR-Net: Adaptive Reconstruction and Localization Refinement for Closely-Spaced Infrared Small Target Unmixing

## 📘 Introduction
Closely-spaced infrared small targets often merge into a single mixed spot due to overlapping image responses, making individual target separation and accurate localization challenging. Variations in target number, radiometric intensities, and relative positions further complicate target reconstruction, while discrete-grid predictions introduce residual subpixel localization errors.

To address these challenges, we propose ARLR-Net, an interpretable deep unfolding network that integrates adaptive kernel mixture and dual-level information compensation for closely-spaced infrared small target unmixing.

The proposed framework consists of three key components:

- KernelMix: Dynamically combines multiple kernels to construct a sample-adaptive effective kernel, improving reconstruction adaptability to varying target configurations.
- Multi-context Adaptive Thresholding: Integrates channel, local, and long-range contextual information to generate adaptive thresholds, improving the separation of overlapping targets.
- Subpixel Offset: Estimates bounded two-dimensional residual displacements to compensate for discrete-grid quantization errors and refine target locations.

Experimental results on the CSIST-100K benchmark demonstrate the effectiveness of ARLR-Net in target separation and subpixel localization.

## 🗂 CSIST-100K Dataset
ARLR-Net is evaluated on the publicly available CSIST-100K dataset introduced by DISTA-Net.
The dataset contains 100,000 synthetic samples, divided into 80,000 training samples, 10,000 validation samples, and 10,000 test samples. Each sample provides an infrared observation together with ground-truth target coordinates and corresponding radiant intensities.

Download: [Baidu Pan](https://pan.baidu.com/s/1nuedV5Okng8rgFWKy_sMoA?pwd=Grok) / [OneDrive](https://1drv.ms/f/c/698f69b8b2172561/EnQbsEb_rXpJlsNXinWyBbsBkhCsnSPM7UEgtczt7FDjmQ).

### Simulation Parameters

| Parameter           | Value/Range              |
|---------------------|--------------------------|
| Imaging Size        | 11×11 pixels             |
| $σ_{PSF}$           | 0.5 pixel                |
| Targets per Image   | 1–5 (random)             |
| Intensity Range     | 220–250 units (uniform)  |
| Spatial Constraints | Sub-pixel coordinates within a pixel + 0.52 Rayleigh unit separation |

### Subpixel Division Settings

ARLR-Net is evaluated under three subpixel division settings:

| Magnification Factor | Output Size |
|----------------------|-------------|
| c = 3                | 33 × 33     |
| c = 5                | 55 × 55     |
| c = 7                | 77 × 77     |

The c = 3 configuration is adopted as the primary setting for method comparison, ablation studies, and qualitative analysis, while c = 5 and c = 7 are used for cross-scale evaluation.

## The network

Overview of the proposed ARLR-Net. The framework consists of an L-stage deep unfolding network for adaptive target reconstruction and a Subpixel Offset module for localization refinement.

Each unfolding stage integrates KernelMix, Multi-context Adaptive Thresholding, and a reconstruction mapping. KernelMix constructs sample-adaptive convolutional kernels to accommodate varying target configurations, while Multi-context Adaptive Thresholding incorporates channel, local, and long-range contextual information to improve target separation.

After the final unfolding stage, discrete target candidates are extracted from the reconstructed intensity map. Subpixel Offset then predicts bounded two-dimensional residual displacements to refine the candidate coordinates in the continuous domain.

The network is jointly optimized using reconstruction loss, symmetry loss, and offset loss.

## 📊 Comparison with state-of-the-art methods
Quantitative comparison with representative methods on the CSIST-100K dataset under the primary c = 3 setting.
---

| Method          | #P      | FLOPs   | CSO-mAP | AP-05 | AP-10 | AP-15 | AP-20 | AP-25 | PSNR (dB) | SSIM (%) |
| --------------- | ------- | ------- | ------- | ----- | ----- | ----- | ----- | ----- | --------- | -------- |
| ISTA            | -       | -       | 15.43   | 0.06  | 1.30  | 9.09  | 24.99 | 41.72 | -         | -        |
| CTNet           | 0.347M  | 0.0861G | 45.07   | 0.39  | 7.59  | 40.53 | 82.22 | 94.62 | 34.93     | 98.30    |
| SRFBN           | 0.373M  | 0.0520G | 46.63   | 0.43  | 8.20  | 43.12 | 85.30 | 96.10 | 35.90     | 98.47    |
| SAN             | 4.442M  | 1.1565G | 46.07   | 0.42  | 8.02  | 42.31 | 83.79 | 95.82 | 35.65     | 98.17    |
| FeNet           | 0.682M  | 0.1654G | 46.00   | 0.38  | 7.49  | 41.39 | 84.35 | 96.37 | 37.17     | 98.74    |
| SDS-Net         | 2.504M  | 0.8471G | 45.39   | 0.54  | 8.48  | 41.29 | 82.16 | 94.45 | 35.53     | 98.53    |
| GGLNet          | 8.988M  | 1.2130G | 44.59   | 0.37  | 7.08  | 40.00 | 81.75 | 93.75 | 34.94     | 98.47    |
| SD2Net          | 0.281M  | 0.1952G | 44.98   | 0.42  | 7.79  | 40.59 | 81.72 | 94.40 | 34.94     | 98.46    |
| LISTA           | 21.000M | 0.0425G | 27.94   | 0.22  | 3.67  | 19.93 | 47.36 | 68.51 | 29.25     | 94.91    |
| ISTA-Net+       | 0.337M  | 1.1219G | 46.15   | 0.44  | 7.92  | 42.16 | 84.30 | 95.94 | 35.95     | 98.63    |
| FISTA-Net       | 0.075M  | 0.8737G | 44.96   | 0.43  | 7.58  | 40.05 | 82.07 | 94.70 | 37.44     | 98.55    |
| DISTA-Net       | 2.179M  | 1.5881G | 46.61   | 0.35  | 7.19  | 41.85 | 86.30 | 97.35 | 39.47     | 98.97    |
| ARLR-Net (Ours) | 2.294M  | 1.6013G | 61.46   | 6.01  | 37.62 | 74.31 | 92.03 | 97.32 | 38.55     | 98.87    |

ARLR-Net achieves a CSO-mAP of 61.46%, outperforming the strongest competing method by 14.83 percentage points. The performance improvement is particularly pronounced under stringent localization criteria, demonstrating the effectiveness of adaptive target reconstruction and subpixel localization refinement.

## 📘 Cross-Scale Evaluation

Cross-scale evaluation is conducted under c = 5 and c = 7 to assess the effectiveness of ARLR-Net at finer spatial resolutions.

| Scale | Method          | #P     | FLOPs  | CSO-mAP (%) |
| ----- | --------------- | ------ | ------ | ----------- |
| c = 5 | DISTA-Net       | 5.153M | 4.483G | 69.81       |
| c = 5 | ISTA-Net+       | 0.225M | 2.175G | 68.07       |
| c = 5 | SRFBN           | 0.602M | 0.097G | 68.28       |
| c = 5 | ARLR-Net (Ours) | 5.267M | 4.524G | 79.58       |

| Scale | Method          | #P     | FLOPs  | CSO-mAP (%) |
| c = 7 | DISTA-Net       | 6.409M | 6.085G | 73.12       |
| c = 7 | ISTA-Net+       | 0.150M | 3.068G | 70.32       |
| c = 7 | SRFBN           | 0.889M | 0.158G | 72.58       |
| c = 7 | ARLR-Net (Ours) | 6.486M | 6.143G | 79.90       |

ARLR-Net achieves the highest CSO-mAP at both magnification factors, demonstrating consistent performance across different spatial resolutions.

### 🛠️Environment Preparation  
#### Installation
The experiments were implemented using Python 3.9, PyTorch 2.1.2, MMCV 2.1.0, MMDetection 3.3.0, and MMEngine.

#### Step 1: Create Environment

```shell
conda create --name arlr python=3.9 -y
conda activate arlr
```
#### Step 2: Install PyTorch

```shell
# PyTorch with CUDA 12.1
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia  
```
#### Step 3: Install OpenMMLab and Additional Dependencies

```shell
pip install -U openmim
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install mmdet==3.3.0

# Additional dependencies
python -m pip install numpy==1.26.4 opencv-python==4.8.1.78 scikit-image fsspec
```

#### Step 4: Install `ARLR-Net`  

```shell
cd ARLR-Net
python setup.py develop
```

#### Step 5: Verify Installation

```shell
python -m pip check
python -c "import torch, mmcv, mmdet, arlr; print('ARLR-Net installation successful!')"
```

### 📄Data preparation

#### 👀dataset directory format is as follows：

```shell
data/
├── initial_matrix/
│   ├── Q_3.mat                     # 3×3 sub-pixel division
│   ├── Q_5.mat                     # 5×5 sub-pixel division
│   └── Q_7.mat                     # 7×7 sub-pixel division
│
├── sampling_matrix/
│   ├── a_phi_0_3.mat              # 3x sub-pixel division
│   ├── a_phi_5.mat                # 5x sub-pixel division
│   └── a_phi_7.mat                # 7x sub-pixel division
│
└── cso_data/                       # CSO Dateset
    ├── train/                      # (80,000 samples)
    │   ├── Annotations/           
    │   │   ├── CSO_00000.xml      
    │   │   ├── ...
    │   │   └── CSO_79999.xml
    │   └── cso_img/               # Infrared image files
    │       ├── image_00000.png    
    │       ├── ...
    │       └── image_79999.png
    │
    ├── val/                        
    │   ├── Annotations/           # 80000-89999
    │   └── cso_img/
    │
    └── test/                       
        ├── Annotations/           # 90000-99999
        └── cso_img/
```
### 🚀 Training and Testing

Models at different magnification factors are trained independently. The network is optimized using Adam with a learning rate of 1e-4 and a batch size of 64.
The c = 3 model is trained for 100 epochs, while the c = 5 and c = 7 models are trained for 250 epochs.

#### ✨Train：

```
# c = 3  
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/c_3/arlr.py
  
# c = 5  
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/c_5/alrl.py
  
# c = 7  
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/c_7/alrl.py   
```

#### ✨Test：

```
# c = 3  
CUDA_VISIBLE_DEVICES=0 python tools/test.py --config configs/c_3/cso_ablation_07_full.py --checkpoint /path/to/arlr_c3_checkpoint.pth --work-dir work_dirs/arlr_c3_test
  
# c = 5  
CUDA_VISIBLE_DEVICES=0 python tools/test.py --config configs/c_5/alrl.py --checkpoint /path/to/arlr_c5_checkpoint.pth --work-dir work_dirs/arlr_c5_test
  
# c = 7  
CUDA_VISIBLE_DEVICES=0 python tools/test.py --config configs/c_7/alrl.py --checkpoint /path/to/arlr_c7_checkpoint.pth --work-dir work_dirs/arlr_c7_test
```