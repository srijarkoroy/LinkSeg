from inference import LinkNetSeg

import streamlit as st
from PIL import Image
import time
import urllib.request
import cv2
import numpy as np

opt = st.sidebar.selectbox("Main",("Home", "Architecture", "Visualizer"), label_visibility="hidden")

if opt == "Home":

    html_temp = """
        <div>
        <h2></h2>
        <center><h3>LinkSeg</h3></center>
        </div>
        <hr>
        <div style = "text-align:justify">
        The characteristics of Retinal Blood Vessels help diagnose various eye ailments. The proper localization, extraction and segmentation of the blood vessels is essential to the cause of treatment of the eye. Manual segmentation of blood vessels may be error prone and inaccurate leading to difficulty in further treatment. We present a novel approach of semantic segmentation of Retinal Blood Vessels using Linked Networks to account for lost spatial information during feature extraction. The implementation of the segmentation technique involves using Residual Networks as a feature extractor and Transpose Convolution for image to image translation thereby giving a segmentation mask as an output. The main feature of the architecture is the links between the Feature Extractor and the Decoder networks that enhance the performance of the network by helping in the recovery of lost spatial information. Training and Validation using the Pytorch framework has been performed on the Digital Retinal Images for Vessel Extraction (DRIVE) Dataset to establish quality results.
        </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.header("")

    prob = st.checkbox("Problem")

    if prob:
        st.write("Manual segmentation of Retinal Blood Vessels is often error prone and exhausting for skilled ophthalmologists, leading to improper diagnosis of the eye ailments.")

    obj = st.checkbox("Objective")

    if obj:
        st.write("Implementations of a Semantic Segmentation Network to accurately segment the Blood Vessels present in the retina for proper treatment and reduction of operator fatigue.")

    res = st.checkbox("Check out the Results of our Implementation!")

    if res:
        st.image("utils/assets/results.png", caption="Results")

elif opt == "Architecture":

    arch = Image.open("utils/assets/linknet.jpeg")
    st.image(arch, caption="LinkNet")

    encoder = st.checkbox("Encoder")

    if encoder:

        enc = Image.open("utils/assets/encoder.jpeg")
        st.image(enc, caption="Encoder Blocks")
        
        html_temp = """
            <div>
            The encoder network comprises of an initial block and 4 residual blocks  for feature extraction purposes.<hr>

            - Initial Block: The initial block of the encoder performs a convolution operation on the input image, with a (7x7) kernel and a stride of 2. This is followed by Batch Normalization and ReLU activation before a spatial max-pooling operation with a kernel size of (3x3) and a stride of 2.<br>

            - Residual Blocks: Following the initial block of the encoder are the residual blocks, used for feature extraction. Each residual block has a strided convolution operation followed by 3 convolutional operations with a (3x3) kernel accompanied by skip connections.
            <div>
        """

        st.markdown(html_temp, unsafe_allow_html=True)

        disp_code = st.checkbox("Encoder Code")

        if disp_code:
            
            code = """
class Encoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
      
        super(Encoder, self).__init__()

        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)

        return x
            """
            st.code(code, language='python')

    decoder = st.checkbox("Decoder")

    if decoder:

        dec = Image.open("utils/assets/decoder.jpeg")
        st.image(dec, caption="Encoder Blocks")
        
        html_temp = """
            <div>
            The decoder network takes the output from the encoder as its input. It comprises 4 decoder blocks and a final segmentation block to perform the main segmentation task.<hr>

            - Decoder Blocks: Each decoder block has 2 convolutional operations with a (1x1) kernel and an Upsample operation in trilinear mode with scale factor 2, between them. The Upsample operation counters the noisy output produced by transpose convolution. The decoder blocks are followed by the final segmentation block.<br>

            - Final Segmentation Block: The final segmentation block of the decoder performs the main segmentation task on the output received from Decoder Block 1. A convolution operation with kernel size (3x3) is performed on the decoder output, followed by an Upsample operation with scale factor 2 in the trilinear mode for a noise-reduced output. Finally, another convolution operation with kernel size (3x3) is performed before passing it through a softmax layer to get the final segmentation mask with pixel probabilities.
            <div>
        """

        st.markdown(html_temp, unsafe_allow_html=True)

        disp_code = st.checkbox("Decoder Code")

        if disp_code:
            
            code = """
class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        
        super(Decoder, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):

        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x
            """
            st.code(code, language='python')


    link = st.checkbox("Linked Network")

    if link:
        
        html_temp = """
            <div>

            The linking between the Feature Extractor and the Decoder networks enhance the performance of the network by helping in the recovery of lost spatial information.<hr>
            Each Encoder Block in the Encoder Network has been linked to its corresponding Decoder in the Decoder Network. The linking is achieved by using skip connections, concatenating the output from ***Encoder Block (i)*** to the input to ***Decoder Block (i)***. The output from Encoder Block (4) goes is passed directly as the input to Decoder Block (4)  as it has no scope for concatenation.

            **Strided Convolutions have been used in order to enable linking between Encoders and Decoders.**
        """

        st.markdown(html_temp, unsafe_allow_html=True)

        disp_code = st.checkbox("Linked Network (LinkNet) Code")

        if disp_code:
            
            code = """
class LinkNet(nn.Module):

    '''
    Generate Model Architecture
    '''

    def __init__(self, n_classes = 1):

        super(LinkNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.encoder1 = Encoder(64, 64, 3, 1, 1)
        self.encoder2 = Encoder(64, 128, 3, 2, 1)
        self.encoder3 = Encoder(128, 256, 3, 2, 1)
        self.encoder4 = Encoder(256, 512, 3, 2, 1)

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax(dim=1)


    def forward(self, x):

        # Initial block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        y = self.lsm(y)

        return y
            """
            st.code(code, language='python')

elif opt == "Visualizer":

    upload_method = st.selectbox("How do you want to upload the image for segmentation?\n", ('Please Select', 'Upload image via link', 'Upload image from device'))

    if upload_method == 'Upload image from device':

        file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
        st.set_option('deprecation.showfileUploaderEncoding', False)
        if file is not None:
            image = Image.open(file)

    elif upload_method == 'Upload image via link':

        try:
            img = st.text_input('Enter the Image Address')
            image = Image.open(urllib.request.urlopen(img))
            
        except:
            if st.button('Submit'):
                show = st.error("Please Enter a valid Image Address!")
                time.sleep(4)
                show.empty()

    try:
        if image is not None:
            st.image(image, width = 500, caption = 'Uploaded Image')
            image = np.array(image)
            test_image = image[:, :, ::-1].copy()

            test_path = "misc/streamlit_uploads/input.png"
            cv2.imwrite(test_path, test_image)

            # Initializing the LinkSeg Inference
            lns = LinkNetSeg(test_path)

            # Running inference
            lns.inference(set_weight_dir = 'linknet.pth', path = 'misc/streamlit_downloads/output.png', blend_path = 'misc/streamlit_downloads/blend.png')

            if st.checkbox("Mask"):
                mask_image = Image.open("misc/streamlit_downloads/output.png")
                st.image(mask_image, width = 500, caption = 'Mask Image')

            if st.checkbox("Blend"):
                blend_image = Image.open("misc/streamlit_downloads/blend.png")
                st.image(blend_image, width = 500, caption = 'Mask Image')

    except Exception as e:
        pass