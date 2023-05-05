import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html

from inference import LinkNetSeg

from linkseg import LinkNet, DiceLoss, IoU, Train, Evaluate
from utils.home import *
from utils.review import *
from utils.architecture import *
from utils.workflow import *
from utils.experiments import *
from utils.visualizer import *
from utils.trainer import *
from utils.results import *
from utils.documents import *

# opt = st.sidebar.selectbox("Main",("Home", "Literature Review", "Architecture", "Workflow", "Experiments", "Visualizer", "Train", "Results and Analysis", "Documents"), label_visibility="hidden")

with st.sidebar:
    opt = option_menu("LinkSeg", ["Home", "Literature Review", "Architecture", "Workflow", "Experiments", "Visualizer", "Train", "Results and Analysis", "Documents"],
                         icons=['house', 'book', 'diagram-3', 'arrow-bar-right','kanban','camera fill','activity','bar-chart-line', 'archive'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#0e117"},
        "icon": {"color": "#08ECAB", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#262730"},
        "nav-link-selected": {"background-color": "#262730"},
    }
    )

if opt == "Home":

    home()

elif opt == "Literature Review":

    literature_review()

elif opt == "Architecture":

    architecture()

elif opt == "Workflow":

    workflow()

elif opt == 'Experiments':

    experiments()

elif opt == "Visualizer":

    try:

        input_image = image_uploader()

        if input_image is not None:

            center(input_image)

            test_path = writer(input_image)

            # Initializing the LinkSeg Inference
            lns = LinkNetSeg(test_path)

            # Running inference
            lns.inference(set_weight_dir = 'linknet.pth', path = 'misc/streamlit_downloads/output.png', blend_path = 'misc/streamlit_downloads/blend.png')

            if st.button("Segment!"):
                display()

    except Exception as e:
        pass


elif opt == "Train":

    try:

        path = file_upload()

        if path:

            train_loader, val_loader = dataloader()

            # Training and Evaluate object
            train = Train(dice=DiceLoss(), iou=IoU())
            eval = Evaluate(dice=DiceLoss(), iou=IoU())

            # Model Initialization
            model = LinkNet()
            print(model)

        if train_loader and val_loader:
            optimizer, lr, epochs= hyperparameters(model.parameters())
            st.info("Optimizer: {}\n\nLearning Rate: {}\n\nEpochs: {}".format(optimizer, lr, epochs))

            if lr != 0:

                if st.button("Train!"):

                    for epoch in range(epochs):
                        train_dice, train_iou = train.forward(model=model, loader=train_loader, optimizer=optimizer)
                        val_dice, val_iou = eval.forward(model=model, loader=val_loader)

                        st.write("Epoch: {}, Dice Loss: {}, IoU Loss: {}".format(epoch, train_dice, train_iou))
    
    except Exception as e:
        pass

elif opt == "Results and Analysis":

    results()

elif opt == "Documents":

    documents()