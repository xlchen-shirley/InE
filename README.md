# Instance-Aware Encoder for Complementary  Information Maximization in Self-Supervised  Infrared and Visible Image Fusion

-
This is the official PyTorch implementation of the InE 2025 paper.

Data
-
If you want to obtain the training data for PMT or Encoder_train, please run dataprocessing.py in the Encoder_train directory to generate the required training data.

Recommended Environment(win10 4070TiSuper)
-
 - torch 1.10.0 
 - numpy 1.26.0
 - opencv-python 4.10.0.84
 - cupy-cuda
 - kornia 0.2.0

Train
-
1. train PMT
"If you want to train the PMT model, please make sure the dataset is ready and run PMT/train.py"
2.train InE
(1) Prepare the required datasets.
(2) Since two encoders are needed to extract features from infrared and visible images, prepare two datasets and train the encoders separately.
(3) Download the pretrained model `model_final.pth` from [this link](https://www.123865.com/s/QmjfTd-JxWc?pwd=fEm0#) (extraction code: `fEm0`)
 and place it in Encoder_train/model.

Test
-
test PMT
If you want to test PMT or generate pseudo-images, please use PMT/test.py. We also provide a pretrained model. Download the pretrained model VI2IR119.pth from [this link](https://www.123865.com/s/QmjfTd-JxWc?pwd=fEm0#) (extraction code: fEm0)，and place it in the ./models/1 directory.

# Use InE

## ⚡ What is InE?

**InE** is an auxiliary fusion loss designed to improve the performance of image fusion.  
By incorporating InE during training, the quality of the fused images can be significantly enhanced.

---

## 🛠 How to Use InE

### Step 1: Prepare Pretrained Models

To use InE, you need the following pretrained models:

**1. Encoder Pretrained Models**  
- Infrared branch: `encoder_ir.pth`  -InE/InELoss/models/ir
- Visible branch: `encoder_vi.pth`   -InE/InELoss/models/vi

**2. Semantic Segmentation Model**  
- BiSeNet segmentation model: `model_final.pth`  -Encoder_train/model

**3. Modality Transformation Model**  
- VI-to-IR model: `VI2IR119.pth`  -InE/SAM/model

### Step 2: Integrate InE into Your Fusion Algorithm
The InE loss is implemented in:  

> 💡 Quick tip: From the root directory of this repository, navigate to `InE/InELoss/` to find `loss.py`.

**Example:**

Suppose your original fusion loss is:  
```python
L = SSIM(vi, f) + SSIM(ir, f)

To use InE:
1. Generate pseudo-infrared and pseudo-visible images using transnet from PMT_test:
vi2ir, ir2vi = transnet(vi, ir)
2. Add the InE loss:
L = SSIM(vi, f) + SSIM(ir, f) + α * InELoss(vi, f, vi2ir, ir, ir2vi)

Note:Since InELoss is an auxiliary fusion loss, when setting α, ensure that α * InELoss is 1/5 to 1/15 of the original loss
```
📊 Results

We provide models trained with InE for the following fusion methods: CDDFuse, SHIP, and SwinFusion.

Download all models here:  [this link](https://www.123865.com/s/QmjfTd-JxWc?pwd=fEm0#)(extraction code: fEm0)

**Important:**
When testing CDDFuse, remove nn.DataParallel to avoid model mismatch errors.

Citation
-
