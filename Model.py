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

        # Model A and B
        self.conv_5_Input_96_1= nn.Conv2d(self.inputSize, 96, 5, padding=1)
        self.conv_5_96_192_1= nn.Conv2d(96, 192, 5, padding=1)
        # Strided A
        self.conv_5_Input_96_2=nn.Conv2d(self.inputSize, 96, 5, padding=1,stride=2)
        self.conv_5_96_192_2=nn.Conv2d(96, 192, 5, padding=1,stride=2)
        # ConvPool
        self.conv_5_96_96_1=nn.Conv2d(96, 96, 5, padding=1)
        self.conv_5_192_192_1=nn.Conv2d(192, 192, 5, padding=1)
        # All A
        self.conv_5_96_96_2 = nn.Conv2d(96, 96, 5, padding=1, stride=2)
        self.conv_5_192_192_2 = nn.Conv2d(192, 192, 5, padding=1, stride=2)


        # Model B
        self.conv_1_96_96_1 = nn.Conv2d(96, 96, 1, padding=1)
        self.conv_1_96_192_1= nn.Conv2d(96, 192, 1, padding=1)

        #  Model C
        self.conv_3_Input_96_1 = nn.Conv2d(self.inputSize, 96, 3, padding=1)
        self.conv_3_96_96_1 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv_3_96_96_2 = nn.Conv2d(96, 96, 3, padding=1, stride=2)

        self.conv_3_96_192_1 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv_3_192_192_1 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv_3_192_192_2 = nn.Conv2d(192, 192, 3, padding=1, stride=2)

        self.conv_1_192_192 = nn.Conv2d(192, 192, 1)
        self.conv_1_192_class = nn.Conv2d(192, self.n_classes, 1)

        # Batch Normalzation
        self.BN_96 = nn.BatchNorm2d(96)
        self.BN_192 = nn.BatchNorm2d(192)

        # Max Pooling
        self.maxP = nn.MaxPool2d(3, stride=2)

        # Other
        self.softMax=nn.Softmax()
        self.flatten=nn.Flatten()
        self.dropOut_2=nn.Dropout(p=0.2)
        self.dropOut_5=nn.Dropout(p=0.5)
        self.relu=nn.ReLU()


        # Debug of baseModel
        if np.sum(self.baseModel)>=2 or np.sum(self.baseModel)==0:
            print('Error in baseModel. Choose model A')
            self.baseModel[0]=True
            self.baseModel[1]=False
            self.baseModel[2]=False

        # Debug of modifiedModel
        if np.sum(self.modifiedModel) >= 2 or np.sum(self.modifiedModel) == 0:
            print('Error in modifiedModel. Choose model without modification')
            self.modifiedModel[0] = True
            self.modifiedModel[1] = False
            self.modifiedModel[2] = False
            self.modifiedModel[3] = False



    def forward(self,x):
        if self.baseModel[0]:
            x=self.__buildModelA(x)
        elif self.baseModel[1]:
            x=self.__buildModelB(x)
        elif self.baseModel[2]:
            x=self.__buildModelC(x)

        # The top layers are identical in each model
        # Layer 6
        x = self.conv_3_192_192_1(x)
        if self.BN:
            x = self.BN_192(x)
        x = self.relu(x)

        # Layer 7
        x = self.conv_1_192_192_1(x)
        if self.BN:
            x = self.BN_192(x)
        x = self.relu(x)

        # Layer 8
        x = self.conv_1_192_class(x)
        if self.BN:
            x = self.BN_192(x)
        x = self.relu(x)

        # The Last Layers
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.flatten(x)
        x = self.softMax(x)

        return x

    def __buildModelA(self, x):

        if self.dropOut:
            x = self.dropOut_2(x)

        # Layer 1
        x = self.conv_5_Input_96_1(x)
        if self.BN:
            x = self.BN_96(x)
        x = self.relu(x)

        # Max Pooling
        x = self.maxP(x)
        if self.dropOut:
            x = self.dropOut_5(x)

        # Layer 2
        x=self.conv_5_96_192_1(x)
        if self.BN:
            x = self.BN_192(x)
        x = self.relu(x)

        # Max Pooling
        x = self.maxP(x)
        if self.dropOut:
            x = self.dropOut_5(x)

        return x

    def __buildModel_stridedA(self, x):

        if self.dropOut:
            x = self.dropOut_2(x)

        # Layer 1
        x = self.conv_5_Input_96_2(x)
        if self.BN:
            x = self.BN_96(x)
        x = self.relu(x)
        if self.dropOut:
            x = self.dropOut_5(x)

        # Layer 2
        x = self.conv_5_96_192_2(x)
        if self.BN:
            x = self.BN_192(x)
        x = self.relu(x)
        if self.dropOut:
            x = self.dropOut_5(x)

        return x

    def __buildModel_ConvPoolA(self, x):

        if self.dropOut:
            x = self.dropOut_2(x)

        # Layer 1
        x = self.conv_5_Input_96_1(x)
        if self.BN:
            x = self.BN_96(x)
        x = self.relu(x)

        # Layer 2
        x = self.conv_5_96_96_1(x)
        if self.BN:
            x = self.BN_96(x)
        x = self.relu(x)

        # Max Pooling
        x = self.maxP(x)
        if self.dropOut:
            x = self.dropOut_5(x)

        # Layer 3
        x = self.conv_5_96_192_1(x)
        if self.BN:
            x = self.BN_192(x)
        x = self.relu(x)

        # Layer 4
        x = self.conv_5_192_192_1(x)
        if self.BN:
            x = self.BN_192(x)
        x = self.relu(x)

        # Max Pooling
        x = self.maxP(x)
        if self.dropOut:
            x = self.dropOut_5(x)

        return x

    def __buildModelB(self, x):

        if self.dropOut:
            x = self.dropOut_2(x)

        # Layer 1
        x = self.conv_5_Input_96_1(x)
        if self.BN:
            x = self.BN_96(x)
        x = self.relu(x)

        # Layer 2
        x = self.conv_1_96_96_1(x)
        if self.BN:
            x = self.BN_96(x)
        x = self.relu(x)

        # Max Pooling
        x = self.maxP(x)
        if self.dropOut:
            x = self.dropOut_5(x)

        # Layer 3
        x = self.conv_5_96_192(x)
        if self.BN:
            x = self.BN_192(x)
        x = self.relu(x)

        # Layer 4
        x = self.conv_1_96_192_1(x)
        if self.BN:
            x = self.BN_192(x)
        x = self.relu(x)

        # Max Pooling
        x = self.maxP(x)
        if self.dropOut:
            x = self.dropOut_5(x)

        return x

    def __buildModelC(self,x):

        if self.dropOut:
            x = self.dropOut_2(x)

        # Layer 1
        x=self.conv_3_Input_96_1(x)
        if self.BN:
            x=self.BN_96(x)
        x=self.relu(x)

        # Layer 2
        x = self.conv_3_96_96_1(x)
        if self.BN:
            x = self.BN_96(x)
        x = self.relu(x)

        # Layer 3
        x = self.conv_3_96_96_1(x)
        if self.BN:
            x = self.BN_96(x)
        x = self.relu(x)

        # Max Pooling
        x =self.maxP(x)
        if self.dropOut:
            x = self.dropOut_5(x)

        # Layer 4
        x=self.conv_3_96_192_1(x)
        if self.BN:
            x = self.BN_192(x)
        x = self.relu(x)

        # Layer 5
        x = self.conv_3_192_192_1(x)
        if self.BN:
            x = self.BN_192(x)
        x = self.relu(x)

        # Max Pooling
        x = self.maxP(x)
        if self.dropOut:
            x = self.dropout(x, .5)

        return x






