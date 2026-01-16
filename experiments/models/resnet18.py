import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dtype=None, device=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                  stride=stride, padding=1, bias=False,
                                  dtype=dtype, device=device)
        self.bn1 = nn.BatchNorm2d(out_channels, dtype=dtype, device=device)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                  stride=1, padding=1, bias=False,
                                  dtype=dtype, device=device)
        self.bn2 = nn.BatchNorm2d(out_channels, dtype=dtype, device=device)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False,
                             dtype=dtype, device=device),
                nn.BatchNorm2d(out_channels, dtype=dtype, device=device)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, madam=False, dtype=None, device=None):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                  padding=1, bias=False,
                                  dtype=dtype, device=device)
        self.bn1 = nn.BatchNorm2d(64, dtype=dtype, device=device)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1, dtype=dtype, device=device)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, dtype=dtype, device=device)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, dtype=dtype, device=device)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, dtype=dtype, device=device)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes, dtype=dtype, device=device)

        if madam:
            self._initialize_weights_madam()
        else:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride, dtype=None, device=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, dtype=dtype, device=device))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)

        out = torch.flatten(out, 1)
        out = self.fc(out)

        return nn.functional.log_softmax(out, dim=1)

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # zero-initialize the last BN in each residual block's residual branch
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    def _initialize_weights_madam(self, eps=1e-1):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -eps, eps)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -eps, eps)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.uniform_(m.bias, -eps, eps)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.uniform_(m.bn2.weight, -eps, eps)