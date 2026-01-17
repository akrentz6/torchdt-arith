import torch
import torch.nn as nn

def channel_shuffle(x, groups=2):
    b, c, h, w = x.size()
    assert c % groups == 0, "Channels must be divisible by groups."
    x = torch.reshape(x, (b, groups, c // groups, h, w))
    x = torch.transpose(x, 1, 2)
    x = torch.reshape(x, (b, c, h, w))
    return x

class ShuffleV2Block(nn.Module):

    def __init__(self, in_c, out_c, stride, dtype=None, device=None):
        super().__init__()
        self.stride = stride
        branch_out = out_c // 2

        if stride == 1:
            self.branch2 = nn.Sequential(
                # 1 × 1 pw-conv
                nn.Conv2d(in_c // 2, branch_out, 1, 1, 0,
                          bias=False, dtype=dtype, device=device),
                nn.BatchNorm2d(branch_out, dtype=dtype, device=device),
                torch.nn.ReLU(),

                # 3 × 3 dw-conv
                nn.Conv2d(branch_out, branch_out, 3, 1, 1,
                          groups=branch_out, bias=False,
                          dtype=dtype, device=device),
                nn.BatchNorm2d(branch_out, dtype=dtype, device=device),

                # 1 × 1 pw-conv
                nn.Conv2d(branch_out, branch_out, 1, 1, 0,
                          bias=False, dtype=dtype, device=device),
                nn.BatchNorm2d(branch_out, dtype=dtype, device=device),
                torch.nn.ReLU(),
            )

        else:
            self.branch1 = nn.Sequential(
                # 3 × 3 dw-conv
                nn.Conv2d(in_c, in_c, 3, stride, 1,
                          groups=in_c, bias=False,
                          dtype=dtype, device=device),
                nn.BatchNorm2d(in_c, dtype=dtype, device=device),

                # 1 × 1 pw-conv
                nn.Conv2d(in_c, branch_out, 1, 1, 0,
                          bias=False, dtype=dtype, device=device),
                nn.BatchNorm2d(branch_out, dtype=dtype, device=device),
                torch.nn.ReLU(),
            )

            self.branch2 = nn.Sequential(
                # 1 × 1 pw-conv
                nn.Conv2d(in_c, branch_out, 1, 1, 0,
                          bias=False, dtype=dtype, device=device),
                nn.BatchNorm2d(branch_out, dtype=dtype, device=device),
                torch.nn.ReLU(),

                # 3 × 3 dw-conv
                nn.Conv2d(branch_out, branch_out, 3, stride, 1,
                          groups=branch_out, bias=False,
                          dtype=dtype, device=device),
                nn.BatchNorm2d(branch_out, dtype=dtype, device=device),

                # 1 × 1 pw-conv
                nn.Conv2d(branch_out, branch_out, 1, 1, 0,
                          bias=False, dtype=dtype, device=device),
                nn.BatchNorm2d(branch_out, dtype=dtype, device=device),
                torch.nn.ReLU(),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), 1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)

        out = channel_shuffle(out, 2)
        return out

class ShuffleNetV2(nn.Module):

    stage_repeats = [4, 8, 4]

    def __init__(self, num_classes = 10, in_channels = 3, dropout = 0.0, madam=False, dtype=None, device=None):
        super().__init__()

        out_channels = [24, 116, 232, 464, 1024]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], 3, 2, 1,
                      bias=False, dtype=dtype, device=device),
            nn.BatchNorm2d(out_channels[0], dtype=dtype, device=device),
            torch.nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        input_c = out_channels[0]
        stage_idx = 0
        blocks = []
        for repeats, output_c in zip(self.stage_repeats, out_channels[1:-1]):
            blocks.append(ShuffleV2Block(input_c, output_c, stride=2, dtype=dtype, device=device))
            input_c = output_c
            for _ in range(repeats - 1):
                blocks.append(ShuffleV2Block(input_c, output_c, stride=1, dtype=dtype, device=device))
            stage_idx += 1
        self.stages = nn.Sequential(*blocks)

        self.conv5 = nn.Sequential(
            nn.Conv2d(input_c, out_channels[-1], 1, 1, 0,
                      bias=False, dtype=dtype, device=device),
            nn.BatchNorm2d(out_channels[-1], dtype=dtype, device=device),
            torch.nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_channels[-1], num_classes, dtype=dtype, device=device)

        if madam:
            self._initialize_weights_madam()
        else:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
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
