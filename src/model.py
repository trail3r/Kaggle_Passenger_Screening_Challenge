import torch
from torch import nn

from torchvision.models import ResNet50_Weights
from torchvision.models import resnet50

from torchvision.models import ConvNeXt_Tiny_Weights
from torchvision.models import convnext_tiny


# Notes: zero_grad -> forward -> loss -> backward -> step

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


class Phase1(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)

        self.backbone = convnext.features
        self.backbone_norm = convnext.classifier[0]

        self.feature_pool = nn.AdaptiveAvgPool2d((10, 8))
        self.feature_adapter = nn.Conv2d(in_channels=768, out_channels=2048, kernel_size=1)
        # ConvNeXt의 출력은 768채널이기 때문에 1*1 Convolution 연산을 수행해 채널 수를 2048채널로 증가시킵니다.
        # 이는 Phase0 모델과의 Backbone 교체 효과만을 비교하기 위한 조치입니다.

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
        features = self.backbone_norm(features)
        features = self.feature_pool(features)
        features = self.feature_adapter(features)

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
            image = scan[:, index].contiguous()
            features = self.encode(image)
            outputs.append(features)

        outputs = torch.stack(outputs, dim=1)
        outputs, _ = self.lstm(outputs)

        final_output = outputs[:, -1, :]

        attention_score = self.view_attention(final_output)
        attention_weights = torch.softmax(attention_score, dim=1)

        scan_features = torch.bmm(
            attention_weights.unsqueeze(1),
            outputs
        )
        scan_features = scan_features.squeeze(1)

        scan_features = self.dropout(scan_features)

        result = self.classifier(scan_features)

        return result


if __name__ == "__main__":
    torch.manual_seed(42)

    """Phase0 Test Codes: ResNet-50 Pretrianed Model
    phase0 = Phase0(pretrained=False)
    phase0.eval()

    scan = torch.randn(1, 16, 3, 661, 512)

    with torch.no_grad():
        result = phase0(scan)

    print(result.shape) """

    phase1 = Phase1(pretrained=False)
    phase1.train()

    # 역전파 테스트에서는 메모리와 시간을 줄이기 위해 작은 이미지를 사용합니다.
    scan = torch.randn(1, 16, 3, 128, 96)
    labels = torch.randint(low=0, high=2, size=(1, 17)).float()

    criterion = nn.BCEWithLogitsLoss()

    result = phase1(scan)
    loss = criterion(result, labels)

    loss.backward()

    print(f"Result shape: {result.shape}")
    print(f"Loss: {loss.item()}")

    print("Backbone gradient:", phase1.backbone[0][0].weight.grad.norm().item())
    print("Adapter gradient:", phase1.feature_adapter.weight.grad.norm().item())
    print("LSTM gradient:", phase1.lstm.weight_ih_l0.grad.norm().item())
    print("Classifier gradient:", phase1.classifier.weight.grad.norm().item())
