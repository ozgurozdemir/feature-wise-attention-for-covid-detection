import os 
import numpy as np
from torch import nn
import torchvision


class FeatureAttentionNetwork(nn.Module):
    """
        Feature-wise Attentive Network
        
        Args:
            num_classes  : int, number of the classes
            feature_dim  : int, dimension of the features
            attention_dim: int, dimension of the attentive features
            backbone     : str, backbone architecture to extract features 
            pretrain_path: str, location of the pretrained. None if the weights will be initialized randomly
    """

    def __init__(self, args):
        super(FeatureAttentionNetwork, self).__init__()
        self.num_classes   = args["num_classes"]
        self.feature_dim   = args["feature_dim"]
        self.attention_dim = args["attention_dim"]
        self.pretrained    = args["pretrain_path"] != None 
        backbone           = args["backbone"] 
        
        if args["pretrain_path"] != None:
            os.environ['TORCH_HOME'] = args["pretrain_path"]
        
        # ResNet
        if 'resnet' in backbone or 'resnext' in backbone:
            if backbone == 'resnet18':
                self.backbone = torchvision.models.resnet18(pretrained=self.pretrained)
            elif backbone == 'resnet34':
                self.backbone = torchvision.models.resnet34(pretrained=self.pretrained)
            elif backbone == 'resnet50':
                self.backbone = torchvision.models.resnet50(pretrained=self.pretrained)
            elif backbone == 'resnet101':
                self.backbone = torchvision.models.resnet101(pretrained=self.pretrained)
            elif backbone == 'resnext50_32x4d':
                self.backbone = torchvision.models.resnext50_32x4d(pretrained=self.pretrained)
            elif backbone == 'resnext101_32x8d':
                self.backbone = torchvision.models.resnext101_32x8d(pretrained=self.pretrained)
            elif backbone == 'wide_resnet50':
                self.backbone = torchvision.models.wide_resnet50_2(pretrained=self.pretrained)
            elif backbone == 'wide_resnet101':
                self.backbone = torchvision.models.wide_resnet101_2(pretrained=self.pretrained)

            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, self.feature_dim)

        # DenseNet
        elif 'dense' in backbone:
            if backbone == 'densenet121':
                self.backbone = torchvision.models.densenet121(pretrained=self.pretrained)
            elif backbone == 'densenet161':
                self.backbone = torchvision.models.densenet161(pretrained=self.pretrained)
            elif backbone == 'densenet169':
                self.backbone = torchvision.models.densenet169(pretrained=self.pretrained)
            elif backbone == 'densenet201':
                self.backbone = torchvision.models.densenet201(pretrained=self.pretrained)

            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(num_features, self.feature_dim)

        # VGG
        elif 'vgg' in backbone:
            if backbone == 'vgg11':
                self.backbone = torchvision.models.vgg11_bn(pretrained=self.pretrained)
            elif backbone == 'vgg13':
                self.backbone = torchvision.models.vgg13_bn(pretrained=self.pretrained)
            elif backbone == 'vgg16':
                self.backbone = torchvision.models.vgg16_bn(pretrained=self.pretrained)
            elif backbone == 'vgg19':
                self.backbone = torchvision.models.vgg19_bn(pretrained=self.pretrained)

            num_features = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Linear(num_features, self.feature_dim)

        # Inception
        elif backbone == 'inception':
            self.backbone = torchvision.models.inception_v3(pretrained=self.pretrained)
            num_features = self.backbone.AuxLogits.fc.in_features
            self.backbone.AuxLogits.fc = nn.Linear(num_features, self.feature_dim)
    
        # SqueezeNet
        elif backbone == 'squeezenet':
            self.backbone = torchvision.models.squeezenet1_1(pretrained=self.pretrained)
            self.backbone.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))

        # AlexNet
        elif backbone == 'alexnet':
            self.backbone = torchvision.models.alexnet(pretrained=self.pretrained)
            num_features = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Linear(num_features, self.feature_dim)

        self.attention = AttentionLayer(self.feature_dim, self.feature_dim//2, self.attention_dim)
        self.hidden    = nn.Linear(self.feature_dim, self.feature_dim//2)
        
        self.fcn     = nn.Linear(self.feature_dim, self.num_classes)
        self.softmax = nn.Softmax(dim=1) 

        
    def forward(self, inp):
        feat   = self.backbone(inp)
        hidden = self.hidden(feat)
        
        attention, alphas = self.attention(feat, hidden)
        
        out = self.fcn(attention)
        return self.softmax(out)



class AttentionLayer(nn.Module):
    """
        Feature-wise Attention layer
        
        Args:
            feature_dim  : int, dimension of the features
            hidden_dim   : int, dimension of the auxilary hidden layer  
            attention_dim: int, dimension of the attentive-features
        
    """
    def __init__(self, feature_dim, hidden_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.feature_att = nn.Linear(feature_dim, attention_dim)  
        self.hidden_att  = nn.Linear(hidden_dim, attention_dim)  
        self.full_att    = nn.Linear(attention_dim, 1)  
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, features, hidden):
        att1 = self.feature_att(features) 
        att2 = self.hidden_att(hidden)  
        
        atten   = self.full_att(self.sigmoid(att1 + att2.unsqueeze(1))).squeeze(2)  
        alpha   = self.softmax(atten)  
        context = (features * alpha.unsqueeze(2)).sum(dim=1)  

        return context, alpha