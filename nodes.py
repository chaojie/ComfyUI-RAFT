import argparse
import os
import sys
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import folder_paths
import glob

from .core.raft import RAFT
from .core.utils import flow_viz, frame_utils
from .core.utils.utils import InputPadder, forward_interpolate

def load_image(imfile,DEVICE):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(flo):
    #img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    #img_flo = np.concatenate([img, flo], axis=0)
    img_flo=flo

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    return img_flo[:, :, [2,1,0]]/255.0

class SaveMotionBrush:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_brush": ("MotionBrush",),
                "save_category": ("STRING", {"default": "smoke"}),
                "save_name": ("STRING", {"default": "smoke1"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "RAFT"
    OUTPUT_NODE = True

    def run(self, motion_brush, save_category, save_name):
        directory=os.path.join(os.path.join(os.path.join(folder_paths.output_directory, "motionbrush"),save_category),save_name)

        os.makedirs(directory, exist_ok=True)

        np.set_printoptions(threshold=np.inf)
        i=0
        for optical_flow in motion_brush:
            print(optical_flow.shape)
            padder = InputPadder([1,3,optical_flow.shape[0],optical_flow.shape[1]])
            flow_up=torch.unsqueeze(optical_flow.permute(2, 0, 1),0)
            flow = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()
            frame_utils.writeFlow(os.path.join(directory,f'{i}.flo'), flow)
            viz_out=viz(flow_up.float())
            image = 255.0 * viz_out
            image_pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
            image_pil.save(os.path.join(directory,f'{i}.png'))
            i=i+1

        np.save(os.path.join(directory,f'{save_name}.npy'),motion_brush.float().cpu().numpy())
        return ()


class LoadMotionBrush:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": (glob.glob(f'{os.path.join(folder_paths.output_directory, "motionbrush")}/**/*.npy', recursive=True), {"default": "smoke/smoke1/smoke1.npy"}),
            }
        }
        
    RETURN_TYPES = ("MotionBrush",)
    FUNCTION = "run_inference"
    CATEGORY = "RAFT"
    def run_inference(self, file_path):
        motion_brush=torch.from_numpy(np.load(file_path))
        print(motion_brush.shape)
        return (motion_brush,)

class RAFTRun:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("OPTICAL_FLOW", "IMAGE", )
    FUNCTION = "run"
    CATEGORY = "RAFT"

    def run(self, images):
        #torch.set_printoptions(threshold=np.inf)
        #np.set_printoptions(threshold=np.inf)
        DEVICE = 'cuda'
        comfy_path = os.path.dirname(folder_paths.__file__)
        
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
        vizs = []
        #flow_prev = None
        with torch.no_grad():
            for image1, image2 in zip(images[:-1], images[1:]):
                preimage=image1
                image1=image1.permute(2, 0, 1).float()[None].to(DEVICE)
                image2=image2.permute(2, 0, 1).float()[None].to(DEVICE)
                print(image1.shape)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                print(flow_up.shape)
                #flow_prev = forward_interpolate(flow_up[0])[None].cuda()
                viz_out=viz(flow_up.float())
                viz_tensor_out = torch.tensor(viz_out)  # Convert back to CxHxW
                viz_tensor_out = torch.unsqueeze(viz_tensor_out, 0)

                motionbrush.append(flow_up.float())
                vizs.append(viz_tensor_out)
        ret=torch.cat(tuple(motionbrush), dim=0).permute(0, 2, 3, 1)
        vizs_tensor=torch.cat(tuple(vizs), dim=0)
        
        return (ret, vizs_tensor, )
        
NODE_CLASS_MAPPINGS = {
    "RAFT Run":RAFTRun,
    "Save MotionBrush":SaveMotionBrush,
    "Load MotionBrush":LoadMotionBrush,
}