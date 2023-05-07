# LinkSeg: An improved LinkNet based approach for Retinal Image Segmentation
Existing systems for Retinal Blood Vessel Segmentation such as UNETs, CNNs, and SVMs have a common issue of losing spatial information during feature extraction, which results in reduced accuracy. This loss of information is caused by multiple downsampling, which compresses the data and makes it more difficult to detect subtle differences. To address this problem, our paper proposes an improved LinkNet-based architecture for semantic segmentation of Retinal Blood Vessels. Our network uses skip connections to link each Encoder Block to its corresponding Decoder Block, thus preserving the lost spatial information and concatenating it during Upsampling. Additionally, we use Upsample layers instead of Transpose Convolution to counteract the issue of noisy outputs, which is often observed in networks that use Transpose Convolutions for the Up-sampling task.

![annot](https://user-images.githubusercontent.com/66861243/236678652-807d9e9d-b30d-405f-ab78-47b83e5d0f17.png)

## Abstract
The characteristics of Retinal Blood Vessels helps identify various eye ailments. The proper localization, extraction and segmentation of blood vessels are essential for the treatment of the eye. Manual segmentation of blood vessels may be error-prone and inaccurate, leading to difficulty in further treatment, and causing problems for both operators and ophthalmologists. We present a novel method of semantic segmentation of Retinal Blood Vessels using Linked Networks to account for spatial information that is lost during feature extraction. The implementation of the segmentation technique involves using Residual Networks as a feature extractor and Transpose Convolution and Upsample Blocks for image-to-image translation thereby giving a segmentation mask as an output. The use of Upsample Blocks arises from its ability to give noise-free output while 18 layered Residual Networks using skip connections are used in the feature extraction without the vanishing gradient issues. The main feature of this architecture is the links between the Feature Extractor and the Decoder networks that improve the performance of the network by helping in the recovery of lost spatial information. Training and Validation using the Pytorch framework have been performed on the Digital Retinal Images for Vessel Extraction (DRIVE) Dataset to establish quality results.

## Implementation Details
The complete workflow of the model training has been shown in the diagram below!

![workflow](https://user-images.githubusercontent.com/66861243/236680436-9947a230-2fcc-4212-a5e6-559509cf521d.png)
- The input image of size (512x512) is first passed through augmentations (horizontal flip, vertical flip and rotation).
- The augmented images undergo normalization and standardization so that they are in a Standard Gaussian Distribution (zero mean and unit standard deviation).
- The normalized images are the passed to the network for training that generates the output segmentation mask.

Check out the UML sequence diagram to further see the working!

![sequence](https://user-images.githubusercontent.com/66861243/236680625-018013ae-10d0-457b-99c9-4d6f8849745e.png)

## Installation and Quick Start
This app is deployed on Streamlit. Check out the deployed web application [here](https://srijarkoroy-linkseg-main-qgis5i.streamlit.app/).

### Local Installation
The application may be run locally on any compatible platform. 

- Cloning the Repository: 

        git clone https://github.com/srijarkoroy/LinkSeg.git

- Entering the directory: 

        cd LinkSeg

- Setting up the Python Environment with Dependencies:

        pip install virtualenv
        python -m venv env
        source env/bin/activate
        pip install -r requirements.txt

- Running the Application

        streamlit run main.py
        
## Results
Validation was performed on the DRIVE Dataset and the STARE Dataset using NVIDIA Tesla T4 GPU. The weights were hosted and downloaded via gdown for further tests. The Figure on the next slide shows the test results of the input retinal images from DRIVE Dataset and validation results are mentioned in the Table below.

Metrics | DRIVE | STARE |
:----------: | :-----------: | :-----------: |
Dice Loss | 0.1164 | 0.1277
IoU Loss | 0.2086 | 0.1962

### Segmented Masks
![results](https://user-images.githubusercontent.com/66861243/236681616-db715247-b8c8-450e-a4ef-aa545e60f010.png)

### Local Testing
![local](https://user-images.githubusercontent.com/66861243/236681855-424394c7-0586-4516-aaeb-8c682d7ceb6a.png)

### Comparison with State-of-the-Art
Metrics | DRIVE |
:----------: | :-----------: |
UNETs	| 0.9790 (AUC ROC)
Lattice NN with Dendrite Processing	| 0.81 (F1 Score)
Multi-level CNN with Conditional Random Fields | 0.9523 (Accuracy)
Multi-scale Line Detection | 0.9326 (Accuracy)
CLAHE	| 0.9477 (Accuracy)
Modified SUSAN edge detector | 0.9633 (Accuracy)
**Improved LinkNet** | **0.8836 (Dice Score)**

## Authors
- [Srijarko Roy](https://github.com/srijarkoroy)
- [Ankit Mathur](https://github.com/am9964)
