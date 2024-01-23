import argparse
import os
import sys
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import folder_paths

def load_image(imfile,DEVICE):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)


class RAFTRun:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("OPTICAL_FLOW", )
    FUNCTION = "run"
    CATEGORY = "RAFT"

    def run(self, images):
        DEVICE = 'cuda'
        comfy_path = os.path.dirname(folder_paths.__file__)
        sys.path.append(f'{comfy_path}/custom_nodes/ComfyUI-RAFT/core')

        from .core.raft import RAFT
        from .core.utils import flow_viz
        from .core.utils.utils import InputPadder
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--model')
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args = parser.parse_args(['--model', f'{comfy_path}/custom_nodes/ComfyUI-RAFT/models/raft-things.pth'])
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))

        model = model.module
        model.to(DEVICE)
        model.eval()

        motionbrush = []
        
        with torch.no_grad():
            for image1, image2 in zip(images[:-1], images[1:]):
                image1=image1.permute(2, 0, 1).float()[None].to(DEVICE)
                image2=image2.permute(2, 0, 1).float()[None].to(DEVICE)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                motionbrush.append(flow_up)
        ret=torch.cat(tuple(motionbrush), dim=0).permute(0, 2, 3, 1)
        print(ret.shape)
        return (ret,)
        
NODE_CLASS_MAPPINGS = {
    "RAFT Run":RAFTRun
}