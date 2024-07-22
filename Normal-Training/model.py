import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm  # Make sure to install timm using pip install timm


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding="same")
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding="same")
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        orig_x = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + orig_x
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale1 = nn.Parameter(torch.randn(6, 6))
        self.bias1 = nn.Parameter(torch.randn(4, 6, 6))
        self.conv1 = nn.Conv2d(4, 64, 3, padding="same")
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 5, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(64, 3, 9, padding="same")

    def forward(self, x):
        x = x.type(torch.float32)
        x = (x - 127.5) / 127.5
        x = x.reshape(-1, 6, 6, 4)
        x = x.permute(0, 3, 1, 2)
        x = torch.mul(x, self.scale1) + self.bias1
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = x * 150
        x = x + 127.5
        return x

def calculate_conv_params(input_size, desired_output_size, kernel_size=7, stride=2, padding=3):
    # Calculate the output size after the convolution
    output_size = (input_size - kernel_size + 2 * padding) // stride + 1
    # If the output size is different from the desired size, adjust the parameters
    if output_size != desired_output_size:
        # Adjust the kernel size, stride, or padding based on your specific needs
        # For simplicity, let's just adjust the padding in this example
        padding = (stride * (desired_output_size - 1) + kernel_size - input_size) // 2
    return kernel_size, stride, padding

class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=mid_channels)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = self.relu(out)

        return out

class ImageHashingModel(nn.Module):
    def __init__(self, input_size=64,hash_size=144):
        super(ImageHashingModel, self).__init__()

        kernel_size, stride, padding = calculate_conv_params(input_size, desired_output_size=50)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.layer1 = Bottleneck(64, 64, 256, stride=1)
        self.layer2 = Bottleneck(256, 128, 512, stride=2)
        self.layer3 = Bottleneck(512, 256, 1024, stride=2)
        self.layer4 = Bottleneck(1024, 512, 2048, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, hash_size)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class resnet18(torch.nn.Module):
    """Constructs a ResNet-18 model.
    """
    def __init__(self, hash_size=144):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        n_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(n_ftrs, hash_size)
    def forward(self, x):
        logits = self.backbone(x)
        # print(output)
        return logits


class ViTForHashing(nn.Module):
    """Constructs a Vision Transformer model for generating hash vectors."""

    def __init__(self, hash_size=144):
        super(ViTForHashing, self).__init__()
        # Load a pre-trained Vision Transformer model
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
        # Get the number of input features to the classifier head
        n_ftrs = self.backbone.head.in_features
        # Replace the classifier head with a new one that outputs 'hash_size' features
        self.backbone.head = nn.Linear(n_ftrs, hash_size)

    def forward(self, x):
        logits = self.backbone(x)
        return logits

