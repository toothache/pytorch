import torch
import torch.nn as nn
 
from sync_batchnorm import patch_replication_callback, SynchronizedBatchNorm2d
from models.resnet import resnet18
 
def main():
    """Main function"""
    resnet = resnet18(norm_layer=SynchronizedBatchNorm2d) # torchvision.models.resnet18()
    print(resnet)
    resnet.cuda()
    resnet = nn.DataParallel(resnet)
    patch_replication_callback(resnet)
 
    tensor = torch.ones(())
    resnet.train()
    resnet(tensor.new_empty((16, 3, 1024, 320)))
 
if __name__ == '__main__':
    main()
