
import argparse
from criterion.criterion import LogSoftMax,SelectCrossEntropyLoss,WeightL1Loss
from network.backbone import buildlModel
from utils.utils import build_data_loader,build_optimizer,process_loss,adjust_optimizer

def train(args):
    total_epoch=args.epoch
    batch=args.batch
    num_workers=args.num_workers
    lr_init = args.lr

    logSoftMax=LogSoftMax()
    selCroEntLoss=SelectCrossEntropyLoss()
    weigthL1Loss=WeightL1Loss()
    model=buildlModel(batch,num_workers)
    trainLoader=build_data_loader()
    data_length=len(trainLoader)
    optimizer=build_optimizer(model,trainLoader,total_epoch,lr_init)

    for epoch,data in enumerate(trainLoader):

        label_cls = data['label_cls']
        label_loc = data['label_loc']
        label_loc_weight = data['label_loc_weight']

        cls,loc=model(data)
        cls=logSoftMax(cls)
        cls_loss=selCroEntLoss(cls,label_cls)
        loc_loss=weigthL1Loss(loc,label_loc,label_loc_weight)
        loss=process_loss(loc_loss,cls_loss)["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer=adjust_optimizer(model,total_epoch,epoch,lr_init)


def main(args):
    if args.mode=="train":
        train(args)
    if args.mode=="eval":
        pass

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mode", type=str, default=train)
    args=parser.parse_args()
    main(args)
