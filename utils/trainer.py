import streamlit as st

import torch
from torch.utils.data import DataLoader

from data.augment import *
from data.dataset import DriveDataset

import os
import time
from glob import glob
from zipfile import ZipFile


__PREFIX__ = os.path.dirname(os.path.realpath(__file__))
# print(__PREFIX__)


def unzip(files):

    for file in files:
        with ZipFile(file, "r") as zip:
            zip.extractall("utils/datasets/")

def file_upload():

    try:

        st.info("Training the LinkNet architcture is computationally expensive, we recommend connecting to a GPU!")

        file = st.file_uploader('Select', type = ['zip'])
        st.set_option('deprecation.showfileUploaderEncoding', False)

        if file is not None:

            with ZipFile(file, "r") as zip:
                zip.extractall(".")
            
            data_path = __PREFIX__+"/datasets/"

            if not os.path.exists(data_path):
                os.mkdir(data_path)
            
            # unzip(["training.zip", "test.zip"])
            with ZipFile("training.zip", "r") as zip:
                zip.extractall("utils/datasets/")
            with ZipFile("test.zip", "r") as zip:
                zip.extractall("utils/datasets/")        

            (train_x, train_y), (test_x, test_y) = load_data(data_path)

            show = st.info(f"File Unzipped Succesfully! Please wait while the unzipped images undergo Augmentations!")
            time.sleep(2)
            show.empty()

            create_dir("new_data/train/image")
            create_dir("new_data/train/mask")
            create_dir("new_data/test/image/")
            create_dir("new_data/test/mask/")

            augment_data(train_x, train_y, "new_data/train", augment=True)
            augment_data(test_x, test_y, "new_data/test", augment=False)
        
        return data_path
    
    except Exception as E:
        st.info("File format should be '.zip'!")

def dataloader():

    data_path = "new_data"

    train_x = sorted(glob(os.path.join(data_path, "train", "image", "*.png")))
    train_y = sorted(glob(os.path.join(data_path, "train", "mask", "*.png")))

    valid_x = sorted(glob(os.path.join(data_path, "test", "image", "*.png")))
    valid_y = sorted(glob(os.path.join(data_path, "test", "mask", "*.png")))

    train_dataset = DriveDataset(train_x, train_y)
    
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2
        )

    val_dataset = DriveDataset(valid_x, valid_y)

    val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2
        )
    
    return train_loader, val_loader

def hyperparameters(params):

    optimize = st.selectbox("Optimizers",("Adam (Recommended)", "Adagrad", "Adadelta", "RMSprop", "SGD"))
    lr = st.number_input('Enter Learning Rate', max_value=0.1, value=0.0001, format="%0.6f")
    epochs = st.slider('Enter no. of Iterations', min_value=10, max_value=100, value=50, step=10)

    if optimize == "Adam (Recommended)":
        optimizer = torch.optim.Adam(params, lr=lr)
    
    elif optimize == "Adagrad":
        optimizer = torch.optim.Adagrad(params, lr=lr) 

    elif optimize == "Adadelta":
        optimizer = torch.optim.Adadelta(params, lr=lr)            

    elif optimize == "RMSprop":
        optimizer = torch.optim.RMSprop(params, lr=lr)

    elif optimize == "SGD":
        optimizer = torch.optim.SGD(params, lr=lr) 

    return optimizer, lr, epochs

# def trainer(train_loader, val_loader):

