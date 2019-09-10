import os
import random
import torch.utils.data as UtilsData
import skimage
import skimage.util as SkUtil
import skimage
import numpy as NP
import torch

class Dataset(UtilsData.Dataset):
    def __init__(self, train=True,path=''):
        super(Dataset,self).__init__()
        self.train = train
        assert os.path.exists(path)
        if path[-1]!='/':
            path+='/'
        self.path=path
        self.dataList= os.listdir(path)
        self.mean=0.5
        random.shuffle(self.dataList)

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, index):
        if self.train:
            image=skimage.data.imread(self.path+self.dataList[index],True)
            targetPsnr=random.randint(10,40)
            initNoise=SkUtil.random_noise(image,'gaussian')-image
            imageAddNoise=image+initNoise
            imageAddNoise = NP.minimum(NP.maximum(imageAddNoise, 0), 1)
            initPsnr = skimage.measure.compare_psnr(image, imageAddNoise)
            noiseTarget=initNoise*NP.power(10,(initPsnr-targetPsnr)/20)
            imageAddNoise=image+noiseTarget
            imageAddNoise=imageAddNoise[NP.newaxis,:]
            return torch.Tensor(imageAddNoise),torch.Tensor(noiseTarget)
        else:
            image = skimage.data.imread(self.path + self.dataList[index], True)
            targetPsnr = random.randint(10, 40)
            initNoise = SkUtil.random_noise(image, 'gaussian') - image
            imageAddNoise = image + initNoise
            imageAddNoise = NP.minimum(NP.maximum(imageAddNoise, 0), 1)
            initPsnr = skimage.measure.compare_psnr(image, imageAddNoise)
            noiseTarget = initNoise * NP.power(10, (initPsnr - targetPsnr) / 20)
            imageAddNoise = image + noiseTarget
            imageAddNoise = imageAddNoise[NP.newaxis, :]
            return torch.Tensor(imageAddNoise), torch.Tensor(noiseTarget)