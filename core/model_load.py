import torch
import AdaFace.net as net
import numpy as np
from config import MODEL_CKPT, DEVICE, MODEL_ARCHITECTURE

def load_pretrained_model(architecture = MODEL_ARCHITECTURE):
    model = net.build_model(architecture)
    statedict = torch.load(MODEL_CKPT, map_location = DEVICE)['state_dict']
    model_statedict = {k[6:]: v for k, v in statedict.items() if k.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    model.to(DEVICE)
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    if np_img.ndim != 3:
        raise ValueError(f"Invalid image shape: {np_img.shape}")
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor.to(DEVICE)


