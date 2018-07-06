import os
import torch
from BEGAN import BEGAN
from DCGAN import DCGAN
from WGAN import WGAN
from config import get_config

def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_index
    torch.manual_seed(config.random_seed)
    if config.use_cuda:
        torch.cuda.manual_seed(config.random_seed)
    if config.model == 'BEGAN':
        model = BEGAN(config)
    elif config.model == 'DCGAN':
        model = DCGAN(config)
    elif config.model == 'WGAN':
        model = WGAN(config)
    
    if config.is_train:
        model.train()
    else:
        model.load_model()
        
if __name__ == "__main__":
    config = get_config()
    main(config)