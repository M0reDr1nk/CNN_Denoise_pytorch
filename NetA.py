import torch
import torch.nn as NN
import torch.functional


class NetA(NN.Module):
    def __init__(self):
        super(NetA, self).__init__()
        self.in_padding=NN.Sequential(
            NN.ReplicationPad2d((8,8,8,8))
        )
        self.conv_in = NN.Sequential(
            NN.Conv2d(1, 32, 7, 1, 3),
            NN.GroupNorm(4,32),
            NN.Conv2d(32, 32, 7, 1, 3),
            NN.GroupNorm(4, 32),
            NN.ELU(inplace=True)
        )
        self.conv0 = NN.Sequential(
            NN.Conv2d(32, 64, 3, 2, 1),
            NN.GroupNorm(8,64),
            NN.ELU(inplace=True),
        )
        self.conv1 = NN.Sequential(
            NN.Conv2d(64, 64, 3, 1, 1),
            NN.GroupNorm(8,64),
            NN.ELU(inplace=True),
            NN.Conv2d(64, 128, 3, 2, 1),
            #NN.GroupNorm(8,128),
            NN.SELU(inplace=True),
        )

        self.conv2 = NN.Sequential(
            NN.Conv2d(128, 128, 3, 1, 1),
            NN.GroupNorm(8,128),
            NN.ELU(inplace=True),
            NN.Conv2d(128, 128, 3, 1, 1),
            #NN.GroupNorm(8,128),
            NN.SELU(inplace=True),
        )

        self.conv3 = NN.Sequential(
            NN.Conv2d(128, 128, 3, 1, 1),
            NN.GroupNorm(8,128),
            NN.ELU(inplace=True),
            NN.Conv2d(128, 256, 3, 2, 1),
            #NN.GroupNorm(16,256),
            NN.SELU(inplace=True),
        )
        self.conv4 = NN.Sequential(
            NN.Conv2d(256, 256, 3, 1, 1),
            NN.GroupNorm(16,256),
            NN.ELU(inplace=True)
        )
        self.conv_in_up=NN.Sequential(
            NN.Conv2d(32, 256, 1, 1, 0),
            #NN.GroupNorm(16, 256),
            NN.SELU(inplace=True)
        )
        self.conv0_up = NN.Sequential(
            NN.Conv2d(64, 256, 1, 1, 0),
            #NN.GroupNorm(16, 256),
            NN.SELU(inplace=True)
        )

        self.conv1_up = NN.Sequential(

            NN.Conv2d(128, 128, 1, 1, 0),
            #NN.GroupNorm(8, 128),
            NN.SELU(inplace=True)
        )
        self.conv2_up = NN.Sequential(

            NN.Conv2d(128, 128, 1, 1, 0),
            #NN.GroupNorm(8, 128),
            NN.SELU(inplace=True)
        )

        self.conv3_up = NN.Sequential(
            NN.Conv2d(256, 256, 1, 1, 0),
            #NN.GroupNorm(8, 64),
            NN.SELU(inplace=True)
        )
        self.conv4_up = NN.Sequential(
            NN.Conv2d(256, 256, 1, 1,0),
            NN.SELU(inplace=True)
            #NN.GroupNorm(16, 256),
            #NN.ELU(inplace=True)
        )

        self.convUp_0=NN.Sequential(
            NN.Conv2d(256, 256, 1, 1, 0),
            NN.SELU(inplace=True),
            NN.PixelShuffle(2),
            NN.Conv2d(64, 128, 3, 1, 1),
            #NN.ReplicationPad2d((0, 1, 0, 1)),
            NN.SELU(inplace=True)
        )
        self.convUp_1 = NN.Sequential(
            NN.Conv2d(128, 256, 1, 1, 0),
            NN.SELU(inplace=True),
            NN.PixelShuffle(2),
            NN.Conv2d(64, 256, 3, 1, 1),
            #NN.ReplicationPad2d((1, 0, 1, 0)),
            NN.SELU(inplace=True)
        )
        self.convUp_2 = NN.Sequential(
            NN.Conv2d(256, 512, 1, 1, 0),
            NN.SELU(inplace=True),
            NN.PixelShuffle(2),
            NN.Conv2d(128, 256, 3, 1, 1),
            # NN.ReplicationPad2d((1, 0, 1, 0)),
            NN.SELU(inplace=True)
        )


        self.res1 = NN.Sequential(
            NN.AvgPool2d(3, 2, 1),
            NN.Conv2d(64, 128, 1, 1, 0),
            NN.SELU(inplace=True)
        )
        self.res2 = NN.Sequential(
            # Nothing is Everything
        )
        self.res3 = NN.Sequential(
            NN.AvgPool2d(3, 2, 1),
            NN.Conv2d(128, 256, 1, 1, 0),
            NN.SELU(inplace=True)
        )
        self.res4 = NN.Sequential(
            # Nothing is Everything
        )

        self.end = NN.Sequential(
            NN.Conv2d(256,64,3,1,1),
            NN.GroupNorm(8,64),
            NN.ELU(inplace=True),
            NN.Conv2d(64,1,3,1,1),
            NN.SELU(inplace=True)
        )
        #self.end_ssim = NN.Sequential(
        #
        #)
        # self.output=NN.Sigmoid()
        self.criterion=NN.L1Loss()
        #self.criterion_psnr = NN.SmoothL1Loss()
        #self.criterion_ssim = NN.SmoothL1Loss()
        self.weightInit(self.conv_in)
        self.weightInit(self.conv0)
        self.weightInit(self.conv1)
        self.weightInit(self.conv2)
        self.weightInit(self.conv3)
        self.weightInit(self.conv4)
        self.weightInit(self.res1)
        self.weightInit(self.res2)
        self.weightInit(self.res3)
        self.weightInit(self.res4)
        self.weightInit(self.conv_in_up)
        self.weightInit(self.conv0_up)
        self.weightInit(self.conv1_up)
        self.weightInit(self.conv2_up)
        self.weightInit(self.conv3_up)
        self.weightInit(self.conv4_up)
        self.weightInit(self.convUp_0)
        self.weightInit(self.convUp_1)
        self.weightInit(self.convUp_2)
        self.weightInit(self.end)
        # self.weightInit(self.end)

    def calcLoss(self, output, target):
        #alpha = 0.8
        ## print("TEST")
        ## print(label[0].data.cpu())
        #label[:, 0] = 1 / pow(10, (label[:, 0] / 20))
        ## print(label[0].data.cpu())
        #loss_psnr = self.criterion_psnr(output_psnr, label[:, 0])
        #loss_ssim = self.criterion_ssim(output_ssim, label[:, 1])
        #loss = alpha * loss_psnr + loss_ssim

        loss=self.criterion(output,target)
        return loss

    def weightInit(self, nnSequential: NN.Sequential):
        for module in nnSequential.modules():
            if isinstance(module, NN.Conv2d):
                NN.init.xavier_normal_(module.weight)
                NN.init.normal_(module.bias, 0, 0.01)
            elif isinstance(module, NN.ConvTranspose2d):
                NN.init.xavier_normal_(module.weight)
                NN.init.normal_(module.bias, 0, 0.01)
                #NN.init.kaiming_normal(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(module, NN.BatchNorm2d):
                NN.init.constant_(module.weight, 1)
                NN.init.constant_(module.bias, 0)
            elif isinstance(module, NN.GroupNorm):
                NN.init.constant_(module.weight, 1)
                NN.init.constant_(module.bias, 0)
            elif isinstance(module, NN.Linear):
                NN.init.xavier_normal_(module.weight)
                #NN.init.normal(module.weight, 0, 0.01)
            else:
                pass

    def forward(self, input):
        input=self.in_padding(input)
        result_conv0_in = self.conv_in(input)
        result_conv_in_up=self.conv_in_up(result_conv0_in)
        mainSteream = self.conv0(result_conv0_in)
        result_conv0_up = self.conv0_up(mainSteream)
        result_conv1_main=self.conv1(mainSteream)
        result_conv1_up=self.conv1_up(result_conv1_main)
        mainSteream = result_conv1_main + self.res1(mainSteream)
        result_conv2_main = self.conv2(mainSteream)
        result_conv2_up = self.conv1_up(result_conv2_main)
        mainSteream = result_conv2_main + self.res2(mainSteream)
        result_conv3_main = self.conv3(mainSteream)
        result_conv3_up = self.conv3_up(result_conv3_main)
        mainSteream = result_conv3_main + self.res3(mainSteream)
        result_conv4_main = self.conv4(mainSteream)
        result_conv4_up = self.conv4_up(result_conv4_main)

        resultUp_0=self.convUp_0(result_conv4_up+result_conv3_up)
        resultUp_1 = self.convUp_1(resultUp_0 + result_conv2_up+result_conv1_up)
        resultUp_2 = self.convUp_2(resultUp_1 + result_conv0_up)
        result_up=self.end(resultUp_2+result_conv_in_up)
        result=input-result_up
        result=torch.squeeze(result,1)
        #mainSteream = result_conv4_main + self.res4(mainSteream)

        #conv1Reslut = conv0Reslut + (-self.conv1(conv0Reslut))
        #conv2Reslut = conv1Reslut + (-self.conv2(conv1Reslut))
        #conv3Reslut = self.conv3(conv2Reslut) + self.res3(conv2Reslut)
        #conv4Reslut = self.conv4(conv3Reslut) + self.res4(conv3Reslut)
        #convEndResult = self.convEnd(conv4Reslut)
        #convEndResult = torch.squeeze(convEndResult, 2)
        #convEndResult = torch.squeeze(convEndResult, 2)
        #
        #result_psnr = self.end_psnr(convEndResult)
        #result_ssim = self.end_ssim(convEndResult)
        result=result[:,8:-8,8:-8]
        return result#result_psnr, result_ssim
