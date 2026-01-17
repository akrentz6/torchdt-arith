import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride, dtype=None, device=None):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                         padding=1, groups=in_channels, bias=False,
                         dtype=dtype, device=device),
            nn.BatchNorm2d(in_channels, dtype=dtype, device=device),
            torch.nn.ReLU(inplace=False)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=1, padding=0, bias=False,
                         dtype=dtype, device=device),
            nn.BatchNorm2d(out_channels, dtype=dtype, device=device),
            torch.nn.ReLU(inplace=False)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNetV1(nn.Module):

    # Depthwise separable conv blocks
    # Each tuple is (out_channels, stride) for the depthwise step
    cfg = [
        (64, 1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (1024, 2),
        (1024, 1),
    ]

    def __init__(self, num_classes=10, in_channels=3, dropout=0.0, madam=False, dtype=None, device=None):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1,
                         padding=1, bias=False,
                         dtype=dtype, device=device),
            nn.BatchNorm2d(32, dtype=dtype, device=device),
            torch.nn.ReLU(inplace=False)
        )

        layers = []
        in_c = 32
        for out_c, stride in self.cfg:
            layers.append(DepthwiseSeparableConv(in_c, out_c, stride, dtype=dtype, device=device))
            in_c = out_c

        self.features = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_c, num_classes, dtype=dtype, device=device)

        if madam:
            self._initialize_weights_madam()
        else:
            self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.reshape(x, (-1, 1024))
        x = self.dropout(x)
        x = self.classifier(x)
        return torch.nn.functional.log_softmax(x, dim=1)

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _initialize_weights_madam(self, eps=1e-1):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -eps, eps)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.uniform_(m.bias, -eps, eps)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.uniform_(m.bias, -eps, eps)