***
# Classifying the visibility of ID cards in photos
The folder images inside data contains several different types of ID documents 
taken in different conditions and backgrounds. The goal is to use the images 
stored in this folder and to design an algorithm that identifies the visibility 
of the card on the photo (FULL_VISIBILITY, PARTIAL_VISIBILITY, NO_VISIBILITY).

## Data
Inside the data folder you can find the following:

### 1) Folder images
A folder containing the challenge images.

### 2) gicsd_labels.csv
A CSV file mapping each challenge image with its correct label.
 - **IMAGE_FILENAME**: The filename of each image.
 - **LABEL**: The label of each image, which can be one of these values: 
 FULL_VISIBILITY, PARTIAL_VISIBILITY or NO_VISIBILITY. 

## Dependencies
This pipeline was developed on a GPU machine with a Tesla T4 GPU, on 
Ubuntu 20.04, with Nvidia Driver Version 440.64 and CUDA Version: 10.2, using
 Python 3.8.2, and PyTorch 1.5, the exact python requirements are given in 
 the `requirements.txt`. 

## Run Instructions
1. With python3.8 install the required libraries: 
`python3.8 -m pip3 install -r requirements.txt`.
2. Run `python3.8 main.py --train` to generate a model in artifacts
3. Run `python3.8 main.py --predict {path_to_image}` to run prediction against 
an image.

## Approach
This approach uses transfer learning and a ResNet50 model, fine-tuning the 
last layer in the architecture with a new network block with a 3 class,
 output replacing the 1000 linear layers that normally perform ImageNet 
 classification. Training is performed only on the parameters in the new layer, 
 whilst using the activations from our frozen layers. 

## Future Work
- Investigate data augmentation of input images, rotate, crop, scale, gaussian
 noise, etc
- Try alternative pretrained models: AlexNet, Vgg-16, ResNet-18, and Resnet-34
- Investigate keeping BatchNorm layers unfrozen
```
for name, param in transfer_model.named_parameters():
    if("bn" not in name):
        param.requires_grad = False
```
