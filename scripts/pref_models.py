import torch
import torch.nn as nn
from torchvision import models


PREF_MODELS = ['resnet', 'dinov2', 'hiera']

class ResNetPreferenceModel(nn.Module):
    def __init__(self, pretrained_model_name="resnet50"):
        super(ResNetPreferenceModel, self).__init__()
        
        # Load pre-trained ResNet for two independent feature heads
        self.resnet1 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.resnet2 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Modify input layers of both ResNets to accept single 3-channel image
        self.resnet1.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet2.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Extract number of features from ResNet output
        num_features = self.resnet1.fc.in_features
        
        # Remove the classification heads; we only need features
        self.resnet1.fc = nn.Identity()
        self.resnet2.fc = nn.Identity()
        
        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(num_features * 2, 1),  # Binary classification
            nn.Sigmoid()  # Output probability
        )
        
        self.linear = nn.Linear(num_features, 1)

    def forward(self, img1, img2):
        # Extract features from both images
        features1 = self.resnet1(img1)
        features2 = self.resnet2(img2)
        
        x1 = self.linear(features1)
        x2 = self.linear(features2)
        
        output = torch.exp(x1) / (torch.exp(x1) + torch.exp(x2))
        
        # Element-wise max of the two feature vectors
        # combined_features = torch.cat((features1, features2), dim=1)
        
        # Classification based on the selected features
        # output = self.fc(combined_features)
        return output

class Dinov2PreferenceModel(nn.Module):
    def __init__(self):
        super(Dinov2PreferenceModel, self).__init__()
        
        # Load pre-trained DINOv2 models; assumes using torch.hub
        self.dino1 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        self.dino2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        
        # Remove classification heads if present
        if hasattr(self.dino1, 'head'):
            self.dino1.head = nn.Identity()
        if hasattr(self.dino2, 'head'):
            self.dino2.head = nn.Identity()
        
        # Assume the feature dimension is accessible as embed_dim; adjust if needed
        num_features = self.dino1.embed_dim if hasattr(self.dino1, 'embed_dim') else 1024
        
        # Define a linear layer to map features to a single scalar
        self.linear = nn.Linear(num_features, 1)
        
    def forward(self, img1, img2):
        # Extract features from both images using the DINOv2 models
        features1 = self.dino1(img1)
        features2 = self.dino2(img2)
        
        # Map features to scalar outputs
        x1 = self.linear(features1)
        x2 = self.linear(features2)
        
        # Compute probability as a softmax-like ratio between the two scores
        output = torch.exp(x1) / (torch.exp(x1) + torch.exp(x2))
        return output    

class HieraPreferenceModel(nn.Module):
    def __init__(self):
        super(HieraPreferenceModel, self).__init__()
        
        # Load pre-trained Hiera model; assumes using torch.hub
        self.hiera1 = torch.hub.load("facebookresearch/hiera", model="hiera_base_224", pretrained=True, checkpoint="mae_in1k_ft_in1k")
        self.hiera2 = torch.hub.load("facebookresearch/hiera", model="hiera_base_224", pretrained=True, checkpoint="mae_in1k_ft_in1k")
        
        # Remove classification heads if present
        if hasattr(self.hiera1, 'head'):
            self.hiera1.head = nn.Identity()
        if hasattr(self.hiera2, 'head'):
            self.hiera2.head = nn.Identity()
        
        # Assume the feature dimension is accessible as embed_dim; adjust if needed
        num_features = self.hiera1.embed_dim if hasattr(self.hiera1, 'embed_dim') else 768 # Output dimension of Hiera base 
        
        # Define a linear layer to map features to a single scalar
        self.linear = nn.Linear(num_features, 1)
        
    def forward(self, img1, img2):
        # Extract features from both images using the Hiera model
        features1 = self.hiera1(img1)
        features2 = self.hiera2(img2)
        
        # Map features to scalar outputs
        x1 = self.linear(features1)
        x2 = self.linear(features2)
        
        # Compute probability as a softmax-like ratio between the two scores
        output = torch.exp(x1) / (torch.exp(x1) + torch.exp(x2))
        return output


if __name__== "__main__":

    # Test
    model = HieraPreferenceModel()
    print(model)

    # Print model attributes like dimensions
    # print("Model Name:", model.__class__.__name__)

    # print("Model Parameters:")
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.shape}")

    ## Test with random images
    img1 = torch.randn(1, 3, 224, 224)  # Example input image tensor
    img2 = torch.randn(1, 3, 224, 224)  # Example input image tensor

    output = model(img1, img2)
    print(output.shape)  # Should be [1, 1]