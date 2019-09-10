import os
import argparse
from Dataset import Dataset
from torch.utils.data import DataLoader
from NetA import NetA
import numpy as NP
import torch
import matplotlib.pyplot as pyplot
import torch.nn as NN
import torch.optim as Optim
import cv2
import skimage.measure

#import torchvision.utils as utils
#from torch.autograd import Variable
#from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
#from models import DnCNN
#from dataset import prepare_data, Dataset
#from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="NetA")
parser.add_argument("--batchSize", type=int, default=4, help="Training batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=list, default=[5,10,20,30], help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--weightDecay", type=float, default=1e-8, help="Initial weight decay")
parser.add_argument("--momentum", type=float, default=0.9, help="Initial momentum")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
opt = parser.parse_args()

def calc_accuracy(output,target):
    #output = output[:, :,:,NP.newaxis]
    #target = target[:, :, :, NP.newaxis]
    ssim_i=[]
    psnr_i=[]
    output=NP.minimum(NP.maximum(0,output),1)
    target=NP.minimum(NP.maximum(0,target),1)
    for i in range(0,output.shape[0]):
        ssim_i.append(skimage.measure.compare_ssim(output[i],target[i]))
        psnr_i.append(skimage.measure.compare_psnr(output[i],target[i]))
    ssim=sum(ssim_i)/output.shape[0]
    psnr=sum(psnr_i)/output.shape[0]

    #diff_ssim=abs(output[:,1]-label_train[:,1])/NP.maximum(label_train[:,1],output[:,1])
    #accuracy_ssim=torch.mean((1-diff_ssim))
    #psnr_out=torch.log10(1/output[:,0])*20
    #psnr_label=torch.log10(1/label_train[:,0])*20
    #diff_psnr=abs(psnr_out-psnr_label)/NP.maximum(psnr_label,psnr_out)
    #accuracy_psnr = torch.mean((1 - diff_psnr))
    #diff=output-target
    #iff=1-(NP.abs(diff) / NP.maximum(NP.abs(output),NP.abs(target)))
    #acc=diff.mean()

    return psnr,ssim


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True,path="D:/Data/Denoise/Train/")
    dataset_val = Dataset(train=False,path="D:/Data/Denoise/Valid/")
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    loader_val = DataLoader(dataset=dataset_val, num_workers=4, batch_size=opt.batchSize, shuffle=False)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    netA = NetA()

    # Move to GPU
    device_ids = [0]
    model = NN.DataParallel(netA, device_ids=device_ids).cuda()
    # Optimizer
    #optimizer = Optim.SGD(model.parameters(),lr=opt.lr,momentum=opt.momentum,weight_decay=opt.weightDecay,nesterov=True)
    optimizer=Optim.Adam(model.parameters(), lr=opt.lr,weight_decay=opt.weightDecay)
    # training
    for epoch in range(opt.epochs):
        if not (epoch in opt.milestone):
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        pyplot.figure(1)
        # train

        for i, data in enumerate(loader_train, 0):
            if i==0:
                pass
                # training step
            model.train()
            optimizer.zero_grad()
            model.zero_grad()
            img_train = data[0].cuda()
            target_train=data[1].cuda()
            output=netA(img_train)
            loss = netA.calcLoss(output,target_train)
            loss.backward()
            optimizer.step()
            if i%50==0:
                loss_show=loss.cpu().data
                img=(torch.squeeze(img_train,1).cpu().data-output.cpu().data)
                img_target = (torch.squeeze(img_train, 1).cpu().data - target_train.cpu().data)
                psnr,ssim = calc_accuracy(img.cpu().numpy(), img_target.cpu().numpy())
                img=img[0]
                img=img[:,:,NP.newaxis]*255
                img = NP.round(img)
                img=NP.minimum(NP.maximum(0,img),255)
                #img_real=(torch.squeeze(img_train[0],1).cpu().data-target_train[0].cpu().data)
                #img_real=img_real[:,:,NP.newaxis]*255
                #img_real=NP.minimum(NP.maximum(0, img_real), 255)
                print("EPOCH:" + str(epoch) + " " + "ITER:" + str(i), end=" ")
                print('learning rate %f' % current_lr)
                print("Loss:" + str(loss_show), end=" ")
                print("PSNR:" + str(psnr)+" SSIM:" + str(ssim))
                #cv2.imshow("Real", img_real.numpy().astype(NP.uint8))
                #cv2.waitKey(1)
                cv2.imshow("Train",img.numpy().astype(NP.uint8))
                cv2.waitKey(1)

        torch.save(model.state_dict(), os.path.join(opt.outf, 'net_epoch_'+str(epoch)+'.pth'))
        if epoch%2==0:
            total_num_of_val=len(loader_val)
            loss_val=0
            psnr_val=0
            ssim_val=0
            with torch.no_grad():
                for i, data in enumerate(loader_val, 0):
                    if i == 0:
                        pass
                    model.eval()
                    model.zero_grad()
                    img_train = data[0].cuda()
                    target_train = data[1].cuda()
                    output = netA(img_train)
                    loss_val += netA.calcLoss(output, target_train)
                    loss_show = (loss_val/i ).cpu().data
                    img = (torch.squeeze(img_train, 1).cpu().data - output.cpu().data)
                    img_target = (torch.squeeze(img_train, 1).cpu().data - target_train.cpu().data)

                    currentPSNR,currentSSIM= calc_accuracy(img.cpu().numpy(), img_target.cpu().numpy())
                    psnr_val+=currentPSNR
                    ssim_val+=currentSSIM
                    psnr = psnr_val /i
                    ssim = ssim_val /i
                    print("(" + str(i) + "/" + str(total_num_of_val) + ")", end=" ")
                    print("Loss:" + str(loss_show), end=" ")
                    print("PSNR:" + str(psnr) + " SSIM:" + str(ssim))
                    if i % 10 == 0:
                        img = img[0]
                        img = img[:, :, NP.newaxis] * 255
                        img = NP.minimum(NP.maximum(0, img), 255)
                        cv2.imshow("Val", img.numpy().astype(NP.uint8))
                        cv2.waitKey(1)



if __name__ == "__main__":
    main()


