## Model Architecture
![linknet](https://user-images.githubusercontent.com/66861243/236676524-3227cbbf-e4ec-4e7a-a759-1c949ec273a9.jpeg)

Modules | Block | Actions | Out Channels |
:----------: | :-----------: | :-----------: | :-----------: |
Encoder | Initial Input Block | Convolution operation and max-pool | 64
Encoder | Residual Blocks | Feature Extraction with skip connections | 64, 128, 256, 512
Decoder | Decoder Blocks | Upscale channels concatenated to the required size | 256, 128, 64, 64
Decoder | Final Segmentation Block | Uses softmax to produce final output mask | 1 

## Model Initialization
The LinkNet model for retina blood vessel segmentation may be initialized and the state dict may be viewed by running the following code snippet:

```python
from linknet import LinkNet

net = LinkNet()
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())
```

## Loss Functions

### Dice Loss
The Dice Loss Function may be initialized and called by running the following code snippet:

```python
from loss import DiceLoss

loss_fn = DiceLoss()
score, loss = loss_fn(output_var, target_var)  #output_var is the output mask and target_var is the label
```
### Intersection over Union
The IoU Loss Function may be initialized and called by running the following code snippet:

```python
from loss import IoU

loss_fn = IoU()
score, loss = loss_fn(output_var, target_var)  #output_var is the output mask and target_var is the label
```

## Model Training
We train the LinkNet model with Dice Loss and Intersection over Union as the loss functions and Adam as the optimizer with a learning rate of 1e-4 for 100 epochs. Since the images are large (512x512) we use a batch size of 4. 

The training object may be initialized and the unet model can be properly trained by running the following code snippet:
```python
from linknet import LinkNet
from loss import DiceLoss, IoU
from training_utils import Train

from torch.optim import Adam

# Training and Evaluate object
train = Train(dice=DiceLoss(), iou=IoU())
eval = Evaluate(dice=DiceLoss(), iou=IoU())

# Model Initialization and setting up hyperparameters
model = LinkNet()
print(model)

# Hyperparameters
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 100

dice_loss = []
iou_loss = []

# Training
for epoch in tqdm(range(epochs)):
    print("Epoch: ", epoch)

    train_dice, train_iou = train.forward(model=model, loader=<train_dataloader_object>, optimizer=optimizer)
    val_dice, val_iou = eval.forward(model=model, loader=<val_loader_object>)

    dice_loss.append(train_dice)
    iou_loss.append(train_iou)

# Plotting Training Curves
plotter.plot(dice_loss, iou_loss)

# Saving model weights
torch.save(model.state_dict(), 'linknet.pth')
```
## Training Results
![graph](https://user-images.githubusercontent.com/66861243/236677612-8a9dd772-eb43-425e-a609-1f3057598c22.png)
