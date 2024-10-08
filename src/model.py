import torch
import torch.nn as nn
from torchsummary import summary

class block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(block,self).__init__()
        self.elu = nn.LeakyReLU()
        self.cv_branch_1_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=(1,1),
                                     stride=1,
                                     padding=0)
        self.cv_branch_2_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=(1,1),
                                     stride=1,
                                     padding=0)
        self.cv_branch_2_2 = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(1,3),
                                       stride=1,
                                       padding=1)
        self.cv_branch_2_3 = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(3,1),
                                       stride=1,
                                       padding=0)
        self.cv_branch_3_1 = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(1,1),
                                       stride=1,
                                       padding=1)
        self.cv_branch_3_2 = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(1,3),
                                       stride=1,
                                       padding=1)
        self.cv_branch_3_3 = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(3,1),
                                       stride=1,
                                       padding=0)
        self.cv_branch_3_4 = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(1, 3),
                                       stride=1,
                                       padding=0)
        self.cv_branch_3_5 = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(3,1),
                                       stride=1,
                                       padding=0)
        self.mp_branch_4_1 = nn.MaxPool2d(kernel_size=(3,3),
                                          stride=1,
                                          padding=1)
        self.cv_branch_4_2 = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(1,1),
                                       stride=1,
                                       padding=0)

    def forward(self,x):
        y_branch_1 = self.cv_branch_1_1(x)
        #
        y_branch_2 = self.cv_branch_2_1(x)
        y_branch_2 = self.cv_branch_2_2(y_branch_2)
        y_branch_2 = self.cv_branch_2_3(y_branch_2)
        #
        y_branch_3 = self.cv_branch_3_1(x)
        y_branch_3 = self.cv_branch_3_2(y_branch_3)
        y_branch_3 = self.cv_branch_3_3(y_branch_3)
        y_branch_3 = self.cv_branch_3_4(y_branch_3)
        y_branch_3 = self.cv_branch_3_5(y_branch_3)
        #
        y_branch_4 = self.mp_branch_4_1(x)
        y_branch_4 = self.cv_branch_4_2(y_branch_4)
        #
        y_branchs = [y_branch_1,y_branch_2,y_branch_3,y_branch_4]
        for i in range(len(y_branchs)):
            y_branchs[i] += x
        # for b in y_branchs:
        #     print(b.shape)
        y = torch.cat(y_branchs,dim=1)
        y = self.elu(y)
        #print(y.shape,x.shape)
        return y

class net(nn.Module):
    def __init__(self,in_channels,out_channels,num_classes):
        super(net,self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cv_init = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(4,3),
                                 stride=1,
                                 padding=0,
                                 bias=False)
        self.bn_init = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.elu = nn.LeakyReLU()
        self.block_1 = block(in_channels=out_channels,out_channels=out_channels)
        #
        self.cv_out = nn.Conv2d(in_channels=out_channels * 4,out_channels=out_channels * 4,
                                kernel_size=(3,3),
                                stride=2,
                                padding=0)
        self.bn_out = nn.BatchNorm2d(num_features=out_channels * 4)
        #
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
        self.fc_1 = nn.Linear(in_features=out_channels * 4,out_features=num_classes)
    def forward(self,input):
        mask = None
        if isinstance(input,list):
            [x,mask] = input
        else:
            x = input
        y = self.cv_init(x)
        #print(y.shape)
        y = self.bn_init(y)
        y = self.elu(y)
        y = self.block_1(y)
        #print(y.shape)
        y = self.cv_out(y)
        y = self.bn_out(y)
        y = self.elu(y)
        #print(y.shape)
        y = self.max_pool(y)
        y = torch.flatten(y,1)
        y = self.fc_1(y)
        if mask is not None:
            y = torch.relu(y)
            y = torch.mul(y,mask)
        return y

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = net(in_channels=33,out_channels=64,num_classes=2086).to(device)
    summary(model,input_size=(33,10,9))