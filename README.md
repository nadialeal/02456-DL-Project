# 02456-DL-Project
## Project 21: Deep Learning and semi-supervised techniques for segmentation of micro-tomography images

This project investigates the problem of particle segmentation in 2D micro-tomography slices of a material, a task complicated by the limited availability of annotated data and the large amount of unlabeled imagery. To address this challenge, we first implement a supervised method using a 2D U-Net trained solely on the labeled slices. We then extend this with a semi-supervised learning strategy based on the Mean Teacher framework, which incorporates unlabeled data through consistency regularization between a Student and a Teacher model. Experimental results show that the semi-supervised model outperforms the supervised-only case in terms of segmentation quality, achieving higher Dice and IoU scores in both patch-level and full-image evaluations.


### Installation

```bash
# (optional) fast solver
conda install -n base -c conda-forge mamba -y
# Create environment
mamba create -n SegmentationDLProject python=3.11 -y

# Activate environment
mamba activate SegmentationDLProject

# PyTorch + CUDA (adjust versions as needed)
mamba install -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.1 -y

# Install libraries
mamba install -c conda-forge albumentations pillow matplotlib tifffile pandas scikit-image -y
pip install opencv-python-headless
```

### Repository structure

├── main_freezelayers.ipynb     # Main notebook: preprocessing, supervised, SSL, eval, figures
├── model_unet.py               # U-Net architecture (DoubleConv, Down/Up blocks, OutConv, InstanceNorm)
├── datasets.py                 # Labeled/Unlabeled datasets, ReplayCompose weak/strong pipeline
├── losses.py                   # BCEWithLogits + soft Dice combo
├── utils.py                    # Sliding-window inference, stitching, eval helpers
├── utils_viz3.py               # Augmentation & prediction visualization utilities
├── checkpoints/                # Best supervised Dice + final teacher/student
├── data/                       # Labeled and unlabeled images (+ split)
├── runs/                       # Per-experiment folders with metrics & patch tests
└── README.md


The main file is where the pipeline can be seen in action and main results are produced and plotted. 

### Preprocessing
22 grayscale microscopy images (768x768) were provided as .tif files (data/labeled/Original_images) with their corresponding masks. Masks were binarized via Otsu's method (scikit-image), the masks used then (data/labeled/Original_masks) have a foreground = 1. In this case, these images are referred to as the Labeled set, and they are used for supervised training, validation, and testing. The mask for each labeled image is resolved by stem or numeric ID. On the other hand, the unlabeled set, which contains 150 images is used only in the semi-supervised (SSL) consistency loss.

Models are trained/evaluated on 256x256 patches. For validation/test (patch-level), a deterministic center crop is taken; for full-ROI evaluation a sliding-window is used. In the case of labeled sampling, to avoid empty patches, a foreground-aware crop that samples around mask positives (with jitter) is used. If no FG is present in a slice, then the code falls back to a random crop.

Augmentations:
- Shared geometry (same for weak and strong): horizontal flip (p=0.5), vertical flip (p=0.2), small rotation, pad-then-crop to 256x256.
- Intensity: weak = none/minimal; strong = light blur or motion blur (p=0.5).

The geometric transform is recorded once and replayed so that the weak and strong crops are spatially aligned; the mask is transformed with the same geometry. This guarantees label consistency.

Images are scaled to [0,1] via Albumentations Normalize(mean=0, std=1, max_pixel_value=255); masks receive geometry only.

### Supervised Block
Supervised learning is done on 256x256 foreground aware random patches training using the U-Net model [1,2] which is a widely used standard architecture for biomedical and material-science segmentation. Concretely, the channel widths follow the typical U-Net pattern (64→128→256→512→1024 and then symmetrically back to 64). For this block, the strong image augmentation is used (no intensity on masks). This aligns with the supervised student distribution in SSL and improves robustness.

Characteristics:
- It performs well on limited labeled datasets.

- Skip connections preserve fine-grained spatial details.

- The encoder learns context, while the decoder reconstructs precise boundaries.

- It is computationally efficient and works well on 256×256 patches.

- It has been shown to outperform classical thresholding or edge-based techniques in micro-CT settings.


| 💪 Pros | ⚠️ Cons |
|---------|---------|
| High accuracy even on small training sets | Performance drops when contrast is extremely poor |
| Good at segmenting objects with fuzzy boundaries (as in low-contrast CT) | Sensitive to intensity distributions |
| Simple, stable, widely validated |Receptive field depends on depth |


- **Loss and metrics**:
Training is done with a BCE + Dice objective on logits, and report Dice/IoU on the thresholded probabilities. For micro-CT segmentation, masks are often imbalanced (small particles vs large background). Dice loss is excellent for overlap but unstable when predictions are near zero and BCE is stable but insensitive to region-level overlap. Therefore, combining them gives the best of both worlds. 

- **Optimization**:
We train the U-Net with AdamW (initial learning rate 1e-3, weight decay 1e-4). AdamW was chosen because its decoupled weight decay acts as true weight shrinkage (better generalization in small, noisy datasets), while retaining Adam’s fast, adaptive convergence. A ReduceLROnPlateau scheduler (mode=max) monitors validation Dice and halves the learning rate (factor=0.5) after 10 epochs without improvement (patience=10), with a minimum learning rate of 1e-6. The checkpoint with the highest validation Dice is selected and saved to initialize the semi-supervised (Mean Teacher) phase.

- **Normalization**:
In this case, we also utilize InstanceNorm instead of the usual BatchNorm, as Instance Normalization normalizes each image independently, making it robust to illumination differences, contrast variations, and domain shifts.

### Semi-Supervised Block
To leverage the available unlabeled data and improve generalization, this project adopts a Semi-Supervised Learning (SSL) framework based on the Mean Teacher[3] method. 
The framework utilizes two distinct models with identical U-Net architectures: a "Student" model and a "Teacher" model. Both are initialized from the supervised checkpoint (best Dice score after 200 epochs). During the SSL phase, the Teacher receives no gradients and its weights are updated by an Exponential Moving Average (EMA) of the Student’s weights (`alpha=0.99` as suggested on [3]).

1. **Setup**
- **Data**: For this case, we defined a Region of Interest (768x768 size, where the actual part of the material we want to segment is located) and we also use random crop of 256x256 with the weak augmentation for the Teacher and strong augmentation for the Student. The adopt a weak/strong consistency scheme was inspired by UDA (Unsupervised Data Augmentation)[4]; strong, label-preserving augmentations improve consistency training—while following Mean-Teacher for EMA targets. For segmentation, other works such as the one described in [5] also show that strong perturbations are beneficial, which motivates the blur/motion-blur strong view.

- **Initialization**: Student and Teacher load the best-Dice supervised checkpoint.

- **Freezing**: the early encoder blocks of the Student are frozen to stabilize training with few labels; decoder/head remain trainable. The Teacher never receives gradients (EMA only).
  
2. **Loss**

- **Consistency loss** (unlabeled batch): MSE( sigmoid(Student(strong)), sigmoid(Teacher(weak)) ).

3. **Weighting & schedule**

AdamW (lr=1e-3, weight_decay=1e-4), with the same ReduceLROnPlateau schedule on validation Dice. A ramp-up is applied on the consistency weight.

4. **Batch composition (per iteration)**

- Take one labeled batch → compute supervised loss on Student (strong).

- Take one unlabeled batch → build weak (Teacher) and strong (Student) views of the same crop → compute consistency loss.

- Backprop only through the Student; update Teacher by EMA.

### Results (summary)
We convert U-Net probabilities to a mask by thresholding the sigmoid outputs at 0.5. All reported Dice/IoU and visualizations use this fixed threshold. 

**Patch-level (256×256) results** 
- Supervised U-Net: Mean Dice score = 0.7354, Mean IoU = 0.5945 
- **SSL Mean Teacher:** **Mean Dice score = 0.7996, Mean IoU = 0.6769**


### References
[1]: O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention (MICCAI). Springer, 2015, pp. 234–241, arXiv:1505.04597.
[2]: Wang Jiangtao, Nur Intan Raihana Ruhaiyem, and Fu Panpan, “A comprehensive review of u-net and its variants: Advances and applications in medical image segmentation,” IET Image Processing, vol. 19, no. 1, pp.e70019, 2025.
[3]: Antti Tarvainen and Harri Valpola, “Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,” 2018. 
[4]: Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, and Quoc V. Le, “Unsupervised data augmentation for consistency training,” 2020.
[5]: Geoff French, Timo Aila, Samuli Laine, Michal Mackiewicz, and Graham Finlayson, “Semi-supervised semantic segmentation needs strong, high-dimensional perturbations,” 2020.
