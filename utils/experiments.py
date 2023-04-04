import streamlit as st

def experiments():

    html_temp = """
        <div>
        <h2></h2>
        <center><h3>Experiments</h3></center>
        <hr>
        </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    st.write("A. Dataset")
    
    html_temp = """
        <div style = "text-align:justify">
        We have used the Digital Retinal Images for Vessel Extraction (DRIVE) dataset for retinal vessel segmentation. It consists of a total of JPEG 40 color fundus images; including 7 abnormal pathology cases. Each image resolution is 584x565 pixels with eight bits per color channel (3 channels), resized to 512x512 for our model. The data has been augmented using the albumentations module. Data Augmentation has been accomplished by doing HorizontalFlip, VerticalFlip and Rotate, thereby generating 160 images for training.
        </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.write("")

    st.image("docs/experiments/augmentation.png", width=600, caption="Original, Horizontal Flip, Vertical Flip")
    st.header("")

    st.write("B. Metrics")
    st.write("  Dice Loss:")

    html_temp = """
        <div style = "text-align:justify">
        We have adopted Dice Loss as a metric for evaluating the performance of our model. The overlap or resemblance between two sets is calculated using the dice coefficient. This metric is particularly popular with class-imbalance problems. The predicted label and the ground truth can be considered as two sets for the purposes of semantic segmentation. The Dice coefficient ranges in value from 0 to 1, where 0 indicates no overlap and 1 indicates total overlap. The dice coefficient is calculated using the following formula.

        Dice Coefficient = 2 | T ∩ P| / |T| + |P|

        The generalised loss function is calculated from the dice coefficient by subtracting it from 1. This dice loss is minimised which maximises the dice coefficient as higher value implies a better overlap.
        </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    disp_code = st.checkbox("Dice Loss Code")

    if disp_code:

        code = """
class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return dice, 1 - dice
        """

        st.code(code, language='python')

    st.write("  IoU Loss:")

    html_temp = """
        <div style = "text-align:justify">
        We have used Intersection over Union as another metric for evaluating the performance of our model. IoU, as the name suggests, is calculated as a ratio of the overlap of the predicted label with the ground truth to their union, i.e. the total area they cover. Similar to the dice loss, it ranges from 0 to 1 , with 0 indicating no overlap and 1 indicating total overlap. The Intersection over Union value is calculated as:

        Intersection over Union = | T ∩ P| / |T U P|

        The generalised IoU loss function is calculated by subtracting the IoU Score from 1, and similar to the Dice Loss, a lower IoU loss gives a better overlap, hence is minimised.
        </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    disp_code = st.checkbox("IoU Loss Code")

    if disp_code:

        code = """
class IoU(nn.Module):
    
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        iou = (intersection + smooth)/(inputs.sum() + targets.sum() - intersection + smooth)

        return iou, 1 - iou
        """

        st.code(code, language='python')

    st.write("C. Training Details")

    html_temp = """
        <div style = "text-align:justify">
        The PyTorch framework was used to code the architecture, and the NVIDIA Tesla T4 GPU and CUDA integration were used to train it on 80 images with a batch size of 4. Before being loaded into a dataloader object for training, each picture underwent pre-processing, was enhanced, normalised, and standardised. To reduce the dice and IoU loss, the model was trained using the Adam optimizer for 100 iterations at a learning rate 0.0001.
        </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.write("")

    st.image("docs/experiments/gpu.png", caption="NVIDIA Tesla T4 GPU Instance")