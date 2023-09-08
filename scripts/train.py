import argparse 

import torch 
from torch.utils.data import DataLoader

from ddpm.util import (
    add_dict_to_argparser, 
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    )


def main():
    args = creat_argparser().parse_args()
    #print(args)
    """ creat the test data """
    dataset = torch.rand(200,1,1,20).to(args.device).float()
    t = torch.randint(0, args.diffusion_steps, size = (200//2,)).to(args.device)
    t = torch.cat([t, args.diffusion_steps - 1 - t], dim = 0)
    noise = torch.randn_like(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print("input dim:",dataset.size()[-1])
    """ ------------------ """
    # change to logger later 
    print("creating model and diffusion...")
    args.din = dataset.size()[-1]
    args.dout = dataset.size()[-1]
    #print(args_to_dict(args,model_and_diffusion_defaults().keys()))
    diffusion, model = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    for t in range(args.num_epoch):
        for idx, batch_x in enumerate(dataloader): # size(32,1,1,20)
            print("batch_x size is:", batch_x.size())
            loss = diffusion.loss(batch_x)
            print(loss)



def creat_argparser():
    defaults = dict(
        data_dir = "",
        lr = 1e-4,
        lr_anneal_steps = 0,
        batch_size = 32,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        diffusion_steps = 1000,
        num_epoch = 1, 
        
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()