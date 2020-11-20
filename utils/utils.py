import collections
from torch.optim import SGD
from torch.utils.data import DataLoader

from utils.dataset import TrkDataset
def process_loss(loc_loss,cls_loss):

    outputs=collections.OrderedDict()
    outputs["total_loss"]=loc_loss+cls_loss
    outputs["cls_loss"]=cls_loss
    outputs["loc_loss"]=loc_loss
    return outputs

def build_data_loader(batch,num_workers):
    train_dataset = TrkDataset()
    train_loader = DataLoader(train_dataset,
                              batch_size=batch,
                              num_workers=num_workers,
                              pin_memory=True,)
    return train_loader
def build_optimizer(model,lr_init):
    optimizer = SGD(model.parameters(), lr=lr_init, momentum=0.9, weight_decay=0.0005)
    return optimizer

def adjust_optimizer(model,total_epoch,epoch,lr_init):
    current_lr=lr_init*((total_epoch-epoch)/total_epoch)
    optimizer = SGD(model.parameters(), lr=current_lr, momentum=0.9, weight_decay=0.0005)
    return optimizer