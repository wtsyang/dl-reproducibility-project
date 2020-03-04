import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, inputSize=32, n_classes=10, baseModel=[True,False,False], modifiedModel=[True, False, False, False],dropOut=True, BN=False, **kwargs):
        super(Model, self).__init__()
        self.inputSize=inputSize
        self.n_classes=n_classes
        self.modifiedModel=modifiedModel
        self.baseModel=baseModel
        self.dropOut=dropOut
        self.BN=BN

        # Debug
        if np.sum(self.baseModel)>=2:
            print('Error in baseModel. Choose model A')
            self.baseModel[0]=True
            self.baseModel[1]=False
            self.baseModel[2]=False

        elif np.sum(self.baseModel)==0:
            print('Error in baseModel. Choose model A')
            self.baseModel[0] = True

        if np.sum(self.modifiedModel) >= 2:
            print('Error in modifiedModel. Choose model without modification')
            self.modifiedModel[0] = True
            self.modifiedModel[1] = False
            self.modifiedModel[2] = False
            self.modifiedModel[3] = False

        elif np.sum(self.modifiedModel) == 0:
            print('Error in modifiedModel. Choose model without modification')
            self.modifiedModel[0] = True



    def forward(self,x):
        if self.baseModel[0]:
            x=self.__buildModelA(x)
        elif self.baseModel[1]:
            x=self.__buildModelB(x)
        elif self.baseModel[2]:
            x=self.__buildModelC(x)
        return x

    def __buildModelC(self,x):

        convInput=nn.Conv2d(input_size, 96, 3, padding=1)
        conv_3_96_96_1=nn.Conv2d(96, 96, 3, padding=1)
        conv_3_96_96_2=nn.Conv2d(96, 96, 3, padding=1, stride=2)

        conv_3_96_192_1=nn.Conv2d(96, 192, 3, padding=1)
        conv_3_192_192_1=nn.Conv2d(192, 192, 3, padding=1)
        conv_3_192_192_2=nn.Conv2d(192, 192, 3, padding=1, stride=2)

        conv_1_192_192 = nn.Conv2d(192, 192, 1)
        conv_1_192_class = nn.Conv2d(192, self.n_classes, 1)
        BN_96=nn.BatchNorm2d(96)
        BN_192=nn.BatchNorm2d(192)
        #BN_class=nn.BatchNorm2d(self.n_classes)

        if seld.dropOut:
            x = F.dropout(x, .2)
        x=convInput(x)
        if self.BN:
            x=BN_96(x)
        x=F.relu(x)
        








