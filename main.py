
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
    model=buildlModel("train")

    trainLoader=build_data_loader(batch,num_workers)
    data_length=len(trainLoader)
    optimizer=build_optimizer(model,lr_init)
    for epoch in range(total_epoch):

        try:
            for step,data in enumerate(trainLoader):

                label_cls = data['label_cls']
                label_loc = data['label_loc']
                label_loc_weight = data['label_loc_weight']

                cls,loc=model(data)

                cls=logSoftMax(cls)
                cls_loss=selCroEntLoss(cls,label_cls)
                loc_loss=weigthL1Loss(loc,label_loc,label_loc_weight)
                loss=process_loss(loc_loss,cls_loss)["total_loss"]
                print("epoch:{epoch} step:{step} loss:{loss}".format(epoch=epoch,step=step,loss=loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer=adjust_optimizer(model,total_epoch,epoch,lr_init)
        except:
            pass

def demo(args):
    pass

def main(args):
    if args.mode=="train":
        train(args)
    if args.mode=="demo":
        demo(args)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mode", type=str, default="train")
    args=parser.parse_args()
    main(args)
