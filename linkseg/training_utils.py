import torch
import torch.nn as nn

from .loss import DiceLoss, IoU


class Train(nn.Module):

    def __init__(self, dice, iou):

        super().__init__()
        
        """
        This class is used for Training a LinkNet model.
        
        Parameters:

        - dice: DiceLoss object
        - iou: IoU object
        """

        self.loss_fn1 = dice
        self.loss_fn2 = iou
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, model, loader, optimizer):

        """
        Parameters:
        
        - model: Model Object/LinkNet Object
        - loader: DataLoader Object
        - optimizer: optimizer Object
        """

        model.train()

        if torch.cuda.is_available():
             print("Shifting the model to cuda!")
             model.cuda()

        epoch_loss1 = 0.0
        epoch_loss2 = 0.0

        for x, y in loader:

            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.float32)

            optimizer.zero_grad()
            y_pred = model(x)

            score1,loss1 = self.loss_fn1(y_pred, y)
            score2,loss2 = self.loss_fn2(y_pred, y)

            loss1.backward(retain_graph = True)
            loss2.backward(retain_graph = True)

            optimizer.step()

            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()

        epoch_loss1 = epoch_loss1/len(loader)
        epoch_loss2 = epoch_loss2/len(loader)

        print("Train Dice Loss: {}, ".format(epoch_loss1),"Train IoU Loss: {}, ".format(epoch_loss2))

        return epoch_loss1, epoch_loss2


class Evaluate(nn.Module):

    def __init__(self, dice, iou):

        super().__init__()
        
        """
        This class is used for Evaluating a LinkNet model.
        
        Parameters:

        - dice: DiceLoss object
        - iou: IoU object
        """

        self.loss_fn1 = dice
        self.loss_fn2 = iou
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, model, loader):

        """
        Parameters:

        - model: Model Object/LinkNet Object
        - loader: DataLoader Object
        """

        model.eval()

        if torch.cuda.is_available():
             print("Shifting the model to cuda!")
             model.cuda()


        epoch_loss1 = 0.0
        epoch_loss2 = 0.0
        
        with torch.no_grad():
            for x, y in loader:

                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)

                # optimizer.zero_grad()
                y_pred = model(x)

                score1,loss1 = self.loss_fn1(y_pred, y)
                score2,loss2 = self.loss_fn2(y_pred, y)

                epoch_loss1 += loss1.item()
                epoch_loss2 += loss2.item()

            epoch_loss1 = epoch_loss1/len(loader)
            epoch_loss2 = epoch_loss2/len(loader)

            print("\nValidation Dice Loss: {}, ".format(epoch_loss1),"Validation IoU Loss: {}, ".format(epoch_loss2))

        return epoch_loss1, epoch_loss2


## Usage ##

# if __name__ == "__main__":

#     train = Train(dice = DiceLoss(), iou = IoU())
#     model = LinkNet()
#     lr = 1e-4
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     # Training
#     train.forward((model = model, loader = <dataloader_object>, optimizer = optimizer, epoch = 50)