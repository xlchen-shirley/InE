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
1. test PMT
If you want to test PMT or generate pseudo-images, please use PMT/test.py. We also provide a pretrained model. Download the pretrained model VI2IR119.pth from [this link](https://www.123865.com/s/QmjfTd-JxWc?pwd=fEm0#) (extraction code: fEm0)，and place it in the ./models/1 directory.
2. test InE

Use InE
-

Citation
-

Contacts
