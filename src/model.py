import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class Phase0(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)

        resnet.conv1.stride = (3, 3)
        resnet.maxpool.stride = 3

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )

        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=128, kernel_size=5, stride=3, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_attention = nn.Linear(2048, 2048)

        self.lstm = nn.LSTM(input_size=49664, hidden_size=768, num_layers=1, batch_first=True)

        self.view_attention = nn.Linear(768, 16)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 17)


    def encode(self, image):
        features = self.backbone(image)

        # Channel Attention
        channel_weights = self.global_pool(features)
        channel_weights = channel_weights.flatten(start_dim=1)
        channel_weights = self.channel_attention(channel_weights)
        channel_weights = channel_weights.unsqueeze(-1).unsqueeze(-1)

        features = features * channel_weights

        # Multi-scale CNN
        global_features = self.global_pool(features)
        features_1x1 = self.branch_1x1(features)
        features_3x3 = self.branch_3x3(features)
        features_5x5 = self.branch_5x5(features)

        global_features = global_features.flatten(start_dim=1)
        features_1x1 = features_1x1.flatten(start_dim=1)
        features_3x3 = features_3x3.flatten(start_dim=1)
        features_5x5 = features_5x5.flatten(start_dim=1)

        view_features = torch.cat([global_features, features_1x1, features_3x3, features_5x5], dim=1)

        return view_features



    def forward(self, scan):
        outputs = []

        for index in range(scan.shape[1]):
            image = scan[:, index]
            features = self.encode(image)
            outputs.append(features)

        outputs = torch.stack(outputs, dim=1)
        outputs, (hidden_state, cell_state) = self.lstm(outputs)

        final_output = outputs[:, -1, :]

        attention_score = self.view_attention(final_output)
        attention_weights = torch.softmax(attention_score, dim=1)

        scan_features = torch.bmm(attention_weights.unsqueeze(1), outputs)
        scan_features = scan_features.squeeze(1)

        scan_features = self.dropout(scan_features)

        result = self.classifier(scan_features)

        return result


if __name__ == "__main__":
    model = Phase0(pretrained=False)
    model.eval()

    scan = torch.randn(1, 16, 3, 661, 512)

    with torch.no_grad():
        result = model(scan)

    print(result.shape)
