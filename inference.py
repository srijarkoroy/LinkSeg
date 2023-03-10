import torch
from PIL import Image
import numpy as np
import os
import json
import gdown
import time
import cv2

from linkseg import LinkNet

__PREFIX__ = os.path.dirname(os.path.realpath(__file__))
#print(os.path.dirname(os.path.realpath(__file__)))

class LinkNetSeg(object):

    def __init__(self, img_path, size = (512, 512)):

        self.img_path = img_path
        self.size = size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def inference(self, set_weight_dir = 'linknet.pth', path = 'output.png', blend = True, blend_path = 'blend.png'):

        set_weight_dir = __PREFIX__ + "/weights/" + set_weight_dir


        ''' saving generated images in a directory '''

        def save_image(path):

            if os.path.exists(path):
                print("Found directory for saving generated images")
                return 1

            else:
                print("Directory for saving images not found, making a directory named 'result_img'")
                os.mkdir(path)
                return 1
        

        ''' dimension expansion and concatenation '''

        def mask_parse(mask):

            mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
            mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
            return mask

        ''' checking if weights are present '''

        def check_weights(set_weight_dir):

            if os.path.exists(set_weight_dir):
                print("Found weights")
                return 1

            else:
                print("Downloading weights")
                download_weights()


        ''' downloading weights if not present '''

        def download_weights():

            with open(__PREFIX__+"/config/weights_download.json") as fp:

                json_file = json.load(fp)

                if not os.path.exists(__PREFIX__+"/weights/"):
                    os.mkdir(__PREFIX__+"/weights/")

                url = 'https://drive.google.com/uc?id={}'.format(json_file['linknet.pth'])
                gdown.download(url, __PREFIX__+"/weights/linknet.pth", quiet=False)
                set_weight_dir = "linknet.pth"

                print("Download finished")

        check_weights(set_weight_dir)

        model = LinkNet()
        model = model.to(self.device)
        
        model.load_state_dict(torch.load(set_weight_dir, map_location=self.device))

        image = cv2.imread(self.img_path, cv2.IMREAD_COLOR)  # (512, 512, 3)
        
        image = cv2.resize(image, self.size)
        x = np.transpose(image, (2, 0, 1))  # (3, 512, 512)
        x = x/255.0
        x = np.expand_dims(x, axis=0)  # (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(self.device)

        time_taken = []

        with torch.no_grad():

            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        #ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((self.size[1], 10, 3)) * 128

        cat_images = np.concatenate(
            [line, pred_y * 255], axis=1
        )

        image = cv2.resize(image, self.size)

        cv2.imwrite(path, cat_images)

        if blend:

            cat_images = cv2.imread(path)
            cat_images = cv2.resize(cat_images, self.size)

            blend = cv2.addWeighted(image, 0.8, cat_images, 0.5, 0)

            cv2.imwrite(blend_path, blend)

        fps = 1/np.mean(time_taken)

        print("Segmented Output Generated! Please check in 'misc/results/'!")
        print("FPS: ", fps)