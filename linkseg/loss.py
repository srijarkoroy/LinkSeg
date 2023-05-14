import torch
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')

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

class PixelAccuracy(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(PixelAccuracy, self).__init__()

    def forward(self, inputs, targets):

        acc = 0

        for i in range (inputs.shape[0]):

            input_val = inputs[i, :, :, :].view(-1).detach().cpu().numpy().argmax(0)
            target_val = targets[i, :, :, :].view(-1).detach().cpu().numpy().argmax(0)

            intersection = (input_val == target_val).sum()

            # dim1, dim2 = input_val.shape
            # acc += intersection/(dim1 * dim2)

            acc += intersection
        
        pixel_accuracy = acc/inputs.shape[0]
        return pixel_accuracy


## Usage ##

# if __name__ == "__main__":
#     loss_dice = DiceLoss()
#     score, loss = loss_dice(output_var, target_var)
#     loss_iou = IoU()
#     score, loss = loss_iou(output_var, target_var)
    # acc_pixel = PixelAccuracy()
    # pixel_accuracy = acc_pixel(output_var, target_var)