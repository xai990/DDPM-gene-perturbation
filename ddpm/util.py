import argparse
from .gaussian_diffusion import DenoiseDiffusion
import torch.utils.data
from .nn import MLPModel

def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        diffusion_steps=1000,
        device = None,
        din = 32,
        dout = 2,
        
    )



def create_model_and_diffusion(diffusion_steps, device,din,dout):
    model = create_model(din, dout, diffusion_steps)
    diffusion = create_diffusion(model,diffusion_steps,device)
    
    return diffusion, model 


def create_model(din, dout, num_steps):
    
    return MLPModel(din, dout, num_steps) 


def create_diffusion(model,n_steps,device):

    return DenoiseDiffusion(model,n_steps,device) 


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        #print(k)
        #print(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k:getattr(args,k) for k in keys}


def str2bool(v):
    if isinstance(v,bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
        

