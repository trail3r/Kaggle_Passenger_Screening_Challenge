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
        features = features.contiguous()  # Ensure MPS-compatible memory layout.

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


class Phase3(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)

        self.backbone = convnext.features
        self.backbone_norm = convnext.classifier[0]

        self.feature_pool = nn.AdaptiveAvgPool2d((10, 8))
        self.feature_adapter = nn.Conv2d(in_channels=768, out_channels=2048, kernel_size=1)
        # ConvNeXt의 출력은 768채널이기 때문에 1*1 Convolution 연산을 수행해 채널 수를 2048채널로 증가시킵니다.

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

        # Change LSTM to Transformer
        self.projection = nn.Sequential(nn.Linear(49664, 768), nn.LayerNorm(768))
        tx_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=tx_encoder_layer, num_layers=1, norm=nn.LayerNorm(768), enable_nested_tensor=False)

        self.view_position = nn.Parameter(torch.zeros(1, 16, 768))
        nn.init.trunc_normal_(self.view_position, std=0.02)

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
        features = features.contiguous()  # Ensure MPS-compatible memory layout.

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
        outputs = self.projection(outputs)
        outputs = outputs + self.view_position
        outputs = self.transformer(outputs)

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


class Phase4(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)

        self.backbone = convnext.features
        self.backbone_norm = convnext.classifier[0]

        self.feature_pool = nn.AdaptiveAvgPool2d((10, 8))
        self.feature_adapter = nn.Conv2d(in_channels=768, out_channels=2048, kernel_size=1)
        # ConvNeXt의 출력은 768채널이기 때문에 1*1 Convolution 연산을 수행해 채널 수를 2048채널로 증가시킵니다.

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

        # Change LSTM to Transformer
        self.projection = nn.Sequential(nn.Linear(49664, 768), nn.LayerNorm(768))
        tx_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=tx_encoder_layer, num_layers=1, norm=nn.LayerNorm(768), enable_nested_tensor=False)

        self.view_attention = nn.Linear(768, 1)  # 모든 뷰에 동일한 함수를 적용해 각 뷰의 중요도를 계산합니다.
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
        features = features.contiguous()  # Ensure MPS-compatible memory layout.

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
        outputs = self.projection(outputs)

        outputs = self.transformer(outputs)

        attention_score = self.view_attention(outputs)
        attention_score = attention_score.squeeze(-1)
        attention_weights = torch.softmax(attention_score, dim=1)

        scan_features = torch.bmm(attention_weights.unsqueeze(1), outputs)
        scan_features = scan_features.squeeze(1)

        scan_features = self.dropout(scan_features)

        result = self.classifier(scan_features)

        return result


def test(phase):
    torch.manual_seed(42)

    if phase == 0:
        model = Phase0(pretrained=False)

        # Phase0는 Adaptive Pooling을 사용하지 않아 원본 크기를 전달합니다.
        height = 661
        width = 512
    elif phase == 1:
        model = Phase1(pretrained=False)  # Optimizer로 SGD를 사용합니다.

        height = 128
        width = 96
    elif phase == 2:
        # Phase0에서 학습 조건을 모두 유지한 채 Backbone을 ConvNeXt-Tiny로 교체하였습니다.
        # 실험 결과, Backbone 교체만으로는 유의미한 성능 상승을 관찰하지 못 하였습니다.
        # Phase1은 Phase0의 학습 조건을 유지하고 Backbone만 ConvNeXt-Tiny로 교체하였습니다.
        # Phase2는 Phase1의 모델 구조를 유지하고 ConvNeXt에 맞게 학습 조건을 변경합니다.
        model = Phase1(pretrained=False)

        height = 128
        width = 96
    elif phase == 3:
        model = Phase3(pretrained=False)

        height = 128
        width = 96
    elif phase == 4:
        model = Phase4(pretrained=False)

        height = 128
        width = 96
    else:
        raise ValueError(f"Unsupported Phase: We don't have Phase{phase}, please check the valid phase.")

    model.train()

    scan = torch.randn(1, 16, 3, height, width)
    labels = torch.randint(low=0, high=2, size=(1, 17)).float()

    criterion = nn.BCEWithLogitsLoss()

    result = model(scan)
    loss = criterion(result, labels)

    loss.backward()

    print(f"Phase{phase} Test")
    print(f"Result Shape: {result.shape}")
    print(f"Loss: {loss}")

    backbone_parameters = next(model.backbone.parameters())
    print(f"Backbone Gradient: {backbone_parameters.grad.norm().item()}")

    if hasattr(model, "lstm"):  # Phase0, Phase1 and Phase2
        print(f"LSTM Gradient: {model.lstm.weight_ih_l0.grad.norm().item()}")

    if hasattr(model, "feature_adapter"):  # Phase1, Phase 2, Phase3 and Phase4
        print(f"Adapter Gradient: {model.feature_adapter.weight.grad.norm().item()}")

    if hasattr(model, "projection"):  # Phase3 and Phase4
        print(f"Projection Gradient: {model.projection[0].weight.grad.norm().item()}")

    if hasattr(model, "transformer"):  # Phase3 and Phase4
        print(f"Transformer Gradient: {model.transformer.layers[0].self_attn.in_proj_weight.grad.norm().item()}")

    if hasattr(model, "view_position"):  # Phase3
        print(f"View Position Gradient: {model.view_position.grad.norm().item()}")

    if hasattr(model, "view_attention"):
        print(f"View Attention Gradient: {model.view_attention.weight.grad.norm().item()}")

    print(f"Classifier Gradient: {model.classifier.weight.grad.norm().item()}")

    model.eval()

    with torch.no_grad():
        original_result = model(scan)

        rolled_scan = torch.roll(scan, shifts=5, dims=1)
        rolled_result = model(rolled_scan)

    difference = torch.max(torch.abs(original_result - rolled_result))

    print(f"Rotation Difference: {difference.item()}")
    print(f"Rotation Invariant: {torch.allclose(original_result, rolled_result, atol=1e-5, rtol=1e-5)}")


if __name__ == "__main__":
    test(phase=2)
