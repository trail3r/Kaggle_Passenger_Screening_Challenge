import torch
from torch import nn

from torchvision.models import ResNet50_Weights
from torchvision.models import resnet50

from torchvision.models import ConvNeXt_Tiny_Weights
from torchvision.models import convnext_tiny

# Notes: zero_grad -> forward -> loss -> backward -> step


# Phase0
# Kaggle Competition: Passenger Screening Algorithm Challenge에서 10위 공개 솔루션을 참고하여 구현하였습니다.
# ResNet-50(Pretrained), Channel Attention, Multi-scale CNN, LSTM, View Attention 구조를 PyTorch로
# 재구성한 Baseline 모델입니다.
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
            nn.BatchNorm2d(512),
        )

        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=128, kernel_size=5, stride=3, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
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


# Phase1
# Phase0의 구조와 학습 조건을 유지한 채 Backbone만 ConvNeXt-Tiny로 교체하였습니다.
# ConvNeXt 출력을 기존 구조에 맞춰 연결하기 위해 768채널을 2048채널로 변환하였습니다.
# 학습 실패: ConvNeXt의 표현이 학습 중 약해지고 예측이 데이터의 양성 비율 수준으로 수렴하였습니다.
# 추정 원인: ResNet-50에 맞춰진 SGD 학습 조건과 Head 연결 방식이 ConvNeXt와 적합하지 않았을 수 있습니다.
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
            nn.BatchNorm2d(512),
        )

        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=128, kernel_size=5, stride=3, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
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

        scan_features = torch.bmm(attention_weights.unsqueeze(1), outputs)
        scan_features = scan_features.squeeze(1)

        scan_features = self.dropout(scan_features)

        result = self.classifier(scan_features)

        return result


# Phase2
# Phase1과 동일한 모델을 사용하고 Optimizer와 Learning Rate만 변경하였습니다.
# SGD 대신 AdamW를 사용하고 Backbone과 새로 추가한 레이어의 Learning Rate를 분리하였습니다.
# 학습 실패: 학습 조건만 변경해서는 Phase1의 데이터의 양성 비율 수준으로 수렴하는 문제를 해결하지 못했습니다.


# Phase3
# ConvNeXt의 768차원 출력을 직접 사용하도록 모델 구조를 재설계하였습니다.
# 각 View의 특징을 LSTM으로 순차 처리하고 하나의 Attention 평가 함수를 모든 View에 공유합니다.
# Attention 평가 함수를 하나로 변경한 것은 추후 입력 View 개수를 줄여보는 상황을 고려하였습니다.
class Phase3(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)

        self.backbone = convnext.features
        self.global_pooling = convnext.avgpool
        self.backbone_normalization = convnext.classifier[0]

        self.lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=1, batch_first=True)

        self.view_attention = nn.Linear(768, 1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 17)

        nn.init.zeros_(self.view_attention.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, images):
        features = self.backbone(images)
        features = self.global_pooling(features)
        features = self.backbone_normalization(features)
        features = features.flatten(start_dim=1)

        return features

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        outputs, _ = self.lstm(view_features)

        attention_score = self.view_attention(outputs)
        attention_score = attention_score.squeeze(-1)
        attention_weights = torch.softmax(attention_score, dim=1)

        scan_features = outputs * attention_weights.unsqueeze(-1)
        scan_features = scan_features.sum(dim=1)

        scan_features = self.dropout(scan_features)
        result = self.classifier(scan_features)

        return result


# Phase4
# Phase3의 LSTM을 1-layer Transformer로 교체하였습니다.
# LSTM과 파라미터 수를 비슷하게 맞추기 위해 FFN을 1536차원으로 설정하였습니다.
# Transformer가 입력된 View의 순서를 구분할 수 있도록 Learned View Position을 추가하였습니다.
class Phase4(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)

        self.backbone = convnext.features
        self.global_pooling = convnext.avgpool
        self.backbone_normalization = convnext.classifier[0]

        self.view_position = nn.Parameter(torch.zeros(1, 16, 768))
        nn.init.trunc_normal_(self.view_position, std=0.02)

        tx_encoder = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=1536,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-6,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=tx_encoder,
            num_layers=1,
            norm=nn.LayerNorm(768, eps=1e-6),
            enable_nested_tensor=False,
        )

        self.view_attention = nn.Linear(768, 1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 17)

        nn.init.zeros_(self.view_attention.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, images):
        features = self.backbone(images)
        features = self.global_pooling(features)
        features = self.backbone_normalization(features)
        features = features.flatten(start_dim=1)

        return features

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        view_position = self.view_position[:, :view_count]

        outputs = view_features + view_position
        outputs = self.transformer(outputs)

        attention_score = self.view_attention(outputs)
        attention_score = attention_score.squeeze(-1)
        attention_weights = torch.softmax(attention_score, dim=1)

        scan_features = outputs * attention_weights.unsqueeze(-1)
        scan_features = scan_features.sum(dim=1)

        scan_features = self.dropout(scan_features)
        result = self.classifier(scan_features)

        return result


# Phase5
# Transformer의 표현 용량 증가가 성능에 미치는 영향을 확인하기 위해 Phase4의 Transformer FFN 크기를 1536차원에서 3072차원으로 확장하였습니다.
# 학습 실패: 학습용 및 검증용 데이터셋에 대한 손실이 데이터의 양성 비율 수준에 정체되고 모든 구역을 음성으로 예측하였습니다.
# FFN 크기와 Learning Rate의 영향을 구분하기 위해 Phase7에서 더 작은 Learning Rate로 재실험을 수행하였습니다.
class Phase5(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)

        self.backbone = convnext.features
        self.global_pooling = convnext.avgpool
        self.backbone_normalization = convnext.classifier[0]

        self.view_position = nn.Parameter(torch.zeros(1, 16, 768))
        nn.init.trunc_normal_(self.view_position, std=0.02)

        tx_encoder = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=3072,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-6,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=tx_encoder,
            num_layers=1,
            norm=nn.LayerNorm(768, eps=1e-6),
            enable_nested_tensor=False,
        )

        self.view_attention = nn.Linear(768, 1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 17)

        nn.init.zeros_(self.view_attention.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, images):
        features = self.backbone(images)
        features = self.global_pooling(features)
        features = self.backbone_normalization(features)
        features = features.flatten(start_dim=1)

        return features

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        view_position = self.view_position[:, :view_count]

        outputs = view_features + view_position
        outputs = self.transformer(outputs)

        attention_score = self.view_attention(outputs)
        attention_score = attention_score.squeeze(-1)
        attention_weights = torch.softmax(attention_score, dim=1)

        scan_features = outputs * attention_weights.unsqueeze(-1)
        scan_features = scan_features.sum(dim=1)

        scan_features = self.dropout(scan_features)
        result = self.classifier(scan_features)

        return result


# Phase6
# FFN 크기가 1536차원인 동일한 Transformer layer를 같은 가중치로 2번 반복하여 사용하였습니다.
# 독립적인 Transformer를 2-layer로 통과하는 것이 아닌 같은 가중치를 가진 하나의 Transformer를 2번 반복 통과하는 구조입니다.
# 학습 실패: 손실이 개선되는 양상을 보였지만 데이터의 양성 비율 수준에서 벗어나지 못해 Epoch 12에서 학습을 중단하였습니다.
class Phase6(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)

        self.backbone = convnext.features
        self.global_pooling = convnext.avgpool
        self.backbone_normalization = convnext.classifier[0]

        self.view_position = nn.Parameter(torch.zeros(1, 16, 768))
        nn.init.trunc_normal_(self.view_position, std=0.02)

        tx_encoder = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=1536,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-6,
        )

        self.transformer = tx_encoder
        self.transformer_normalization = nn.LayerNorm(768, eps=1e-6)
        self.transformer_depth = 2

        self.view_attention = nn.Linear(768, 1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 17)

        nn.init.zeros_(self.view_attention.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, images):
        features = self.backbone(images)
        features = self.global_pooling(features)
        features = self.backbone_normalization(features)
        features = features.flatten(start_dim=1)

        return features

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        view_position = self.view_position[:, :view_count]

        outputs = view_features + view_position

        for _ in range(self.transformer_depth):
            outputs = self.transformer(outputs)

        outputs = self.transformer_normalization(outputs)

        attention_score = self.view_attention(outputs)
        attention_score = attention_score.squeeze(-1)
        attention_weights = torch.softmax(attention_score, dim=1)

        scan_features = outputs * attention_weights.unsqueeze(-1)
        scan_features = scan_features.sum(dim=1)

        scan_features = self.dropout(scan_features)
        result = self.classifier(scan_features)

        return result


# Phase7
# Phase5와 동일한 FFN 3072차원 모델에서 Transformer의 Learning Rate만 2e-4로 낮추었습니다.
# Phase5의 실패에 Transformer의 Learning Rate가 영향을 주었는지 확인합니다.
class Phase7(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)

        self.backbone = convnext.features
        self.global_pooling = convnext.avgpool
        self.backbone_normalization = convnext.classifier[0]

        self.view_position = nn.Parameter(torch.zeros(1, 16, 768))
        nn.init.trunc_normal_(self.view_position, std=0.02)

        tx_encoder = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=3072,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-6,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=tx_encoder,
            num_layers=1,
            norm=nn.LayerNorm(768, eps=1e-6),
            enable_nested_tensor=False,
        )

        self.view_attention = nn.Linear(768, 1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 17)

        nn.init.zeros_(self.view_attention.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, images):
        features = self.backbone(images)
        features = self.global_pooling(features)
        features = self.backbone_normalization(features)
        features = features.flatten(start_dim=1)

        return features

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        view_position = self.view_position[:, :view_count]

        outputs = view_features + view_position
        outputs = self.transformer(outputs)

        attention_score = self.view_attention(outputs)
        attention_score = attention_score.squeeze(-1)
        attention_weights = torch.softmax(attention_score, dim=1)

        scan_features = outputs * attention_weights.unsqueeze(-1)
        scan_features = scan_features.sum(dim=1)

        scan_features = self.dropout(scan_features)
        result = self.classifier(scan_features)

        return result


# Phase8
# Phase6와 동일하게 동일한 가중치를 가진 Transformer를 2번 통과하는 모델을 사용합니다.
# Transformer의 Learning Rate를 2.5e-4로 줄여 반복 사용되는 파라미터의 학습이 과도했는지 확인합니다.
class Phase8(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)

        self.backbone = convnext.features
        self.global_pooling = convnext.avgpool
        self.backbone_normalization = convnext.classifier[0]

        self.view_position = nn.Parameter(torch.zeros(1, 16, 768))
        nn.init.trunc_normal_(self.view_position, std=0.02)

        tx_encoder = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=1536,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-6,
        )

        self.transformer = tx_encoder
        self.transformer_normalization = nn.LayerNorm(768, eps=1e-6)
        self.transformer_depth = 2

        self.view_attention = nn.Linear(768, 1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 17)

        nn.init.zeros_(self.view_attention.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, images):
        features = self.backbone(images)
        features = self.global_pooling(features)
        features = self.backbone_normalization(features)
        features = features.flatten(start_dim=1)

        return features

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        view_position = self.view_position[:, :view_count]

        outputs = view_features + view_position

        for _ in range(self.transformer_depth):
            outputs = self.transformer(outputs)

        outputs = self.transformer_normalization(outputs)

        attention_score = self.view_attention(outputs)
        attention_score = attention_score.squeeze(-1)
        attention_weights = torch.softmax(attention_score, dim=1)

        scan_features = outputs * attention_weights.unsqueeze(-1)
        scan_features = scan_features.sum(dim=1)

        scan_features = self.dropout(scan_features)
        result = self.classifier(scan_features)

        return result


# Phase9
# Phase8의 첫 번째와 두 번째 Transformer 출력을 학습 가능한 View별 Gate로 결합합니다.
# Gate는 sigmoid(-2)로 초기화하여 첫 번째 Transformer의 출력을 약 88%, 두 번째 Transformer의 출력을 약 12% 사용하며 학습을 시작합니다.
# 가설 기각: Gate의 Gradient를 확인해보니 두 번째 Gate가 사실상 닫혀있어 첫 번째 Gate의 출력만을 사용하고 있었습니다.
#          Recall 역시 Phase8에 비하여 약 4.3%p 하락한 수치로 Phase8보다 빨리 수렴한 것은 맞지만 비효율적이라 판단하여 기각하였습니다.
class Phase9(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)

        self.backbone = convnext.features
        self.global_pooling = convnext.avgpool
        self.backbone_normalization = convnext.classifier[0]

        self.view_position = nn.Parameter(torch.zeros(1, 16, 768))
        nn.init.trunc_normal_(self.view_position, std=0.02)

        tx_encoder = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=1536,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-6,
        )

        self.transformer = tx_encoder
        self.transformer_normalization = nn.LayerNorm(768, eps=1e-6)

        self.view_attention = nn.Linear(768, 1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 17)

        self.gate = nn.Linear(768 * 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

        nn.init.zeros_(self.view_attention.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, images):
        features = self.backbone(images)
        features = self.global_pooling(features)
        features = self.backbone_normalization(features)
        features = features.flatten(start_dim=1)

        return features

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        view_position = self.view_position[:, :view_count]

        outputs = view_features + view_position

        outputs_1 = self.transformer(outputs)
        outputs_2 = self.transformer(outputs_1)

        gate_input = torch.cat([outputs_1, outputs_2], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))

        outputs = (1 - gate) * outputs_1 + gate * outputs_2
        outputs = self.transformer_normalization(outputs)

        attention_score = self.view_attention(outputs)
        attention_score = attention_score.squeeze(-1)
        attention_weights = torch.softmax(attention_score, dim=1)

        scan_features = outputs * attention_weights.unsqueeze(-1)
        scan_features = scan_features.sum(dim=1)

        scan_features = self.dropout(scan_features)
        result = self.classifier(scan_features)

        return result


# Phase10
# Phase7과 동일한 1-layer Transformer에서 FFN 크기만 3072차원에서 1536차원으로 줄였습니다.
# 동일한 Learning Rate 조건에서 Transformer FFN 크기가 성능에 미치는 영향을 비교합니다.
class Phase10(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)

        self.backbone = convnext.features
        self.global_pooling = convnext.avgpool
        self.backbone_normalization = convnext.classifier[0]

        self.view_position = nn.Parameter(torch.zeros(1, 16, 768))
        nn.init.trunc_normal_(self.view_position, std=0.02)

        tx_encoder = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=1536,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-6,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=tx_encoder,
            num_layers=1,
            norm=nn.LayerNorm(768, eps=1e-6),
            enable_nested_tensor=False,
        )

        self.view_attention = nn.Linear(768, 1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 17)

        nn.init.zeros_(self.view_attention.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, images):
        features = self.backbone(images)
        features = self.global_pooling(features)
        features = self.backbone_normalization(features)
        features = features.flatten(start_dim=1)

        return features

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        view_position = self.view_position[:, :view_count]

        outputs = view_features + view_position
        outputs = self.transformer(outputs)

        attention_score = self.view_attention(outputs)
        attention_score = attention_score.squeeze(-1)
        attention_weights = torch.softmax(attention_score, dim=1)

        scan_features = outputs * attention_weights.unsqueeze(-1)
        scan_features = scan_features.sum(dim=1)

        scan_features = self.dropout(scan_features)
        result = self.classifier(scan_features)

        return result


# Phase11
# Phase8의 동일한 가중치를 가진 Transformer를 2번 반복 통과하는 구조를 유지하고 Transformer의 Learning Rate를 2e-4로 낮춥니다.
# Phase10과 비교하여 동일한 FFN 크기와 Learning Rate에서 공유 가중치 Transformer 2-layer 통과 효과를 확인하기 위한 실험입니다.
class Phase11(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)

        self.backbone = convnext.features
        self.global_pooling = convnext.avgpool
        self.backbone_normalization = convnext.classifier[0]

        self.view_position = nn.Parameter(torch.zeros(1, 16, 768))
        nn.init.trunc_normal_(self.view_position, std=0.02)

        tx_encoder = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=1536,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-6,
        )

        self.transformer = tx_encoder
        self.transformer_normalization = nn.LayerNorm(768, eps=1e-6)
        self.transformer_depth = 2

        self.view_attention = nn.Linear(768, 1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 17)

        nn.init.zeros_(self.view_attention.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, images):
        features = self.backbone(images)
        features = self.global_pooling(features)
        features = self.backbone_normalization(features)
        features = features.flatten(start_dim=1)

        return features

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        view_position = self.view_position[:, :view_count]

        outputs = view_features + view_position

        for _ in range(self.transformer_depth):
            outputs = self.transformer(outputs)

        outputs = self.transformer_normalization(outputs)

        attention_score = self.view_attention(outputs)
        attention_score = attention_score.squeeze(-1)
        attention_weights = torch.softmax(attention_score, dim=1)

        scan_features = outputs * attention_weights.unsqueeze(-1)
        scan_features = scan_features.sum(dim=1)

        scan_features = self.dropout(scan_features)
        result = self.classifier(scan_features)

        return result


# Phase12
# Phase11이 동일한 가중치를 가진 Transformer를 2번 반복 통과하는 것을 비교하였다면, Phase12는 서로 다른 Transformer를 2-layer로 적층합니다.
class Phase12(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)

        self.backbone = convnext.features
        self.global_pooling = convnext.avgpool
        self.backbone_normalization = convnext.classifier[0]

        self.view_position = nn.Parameter(torch.zeros(1, 16, 768))
        nn.init.trunc_normal_(self.view_position, std=0.02)

        tx_encoder = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=1536,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-6,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=tx_encoder,
            num_layers=2,
            norm=nn.LayerNorm(768, eps=1e-6),
            enable_nested_tensor=False,
        )

        self.view_attention = nn.Linear(768, 1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 17)

        nn.init.zeros_(self.view_attention.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, images):
        features = self.backbone(images)
        features = self.global_pooling(features)
        features = self.backbone_normalization(features)
        features = features.flatten(start_dim=1)

        return features

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        view_position = self.view_position[:, :view_count]

        outputs = view_features + view_position
        outputs = self.transformer(outputs)

        attention_score = self.view_attention(outputs)
        attention_score = attention_score.squeeze(-1)
        attention_weights = torch.softmax(attention_score, dim=1)

        scan_features = outputs * attention_weights.unsqueeze(-1)
        scan_features = scan_features.sum(dim=1)

        scan_features = self.dropout(scan_features)
        result = self.classifier(scan_features)

        return result


# Phase13
# 독립적인 Transformer를 4-layer로 적층합니다. 표현 용량이 더 증가하면 카메라 시퀀스를 더 잘 이해할 수 있는지 확인하기 위한 실험입니다.
class Phase13(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)

        self.backbone = convnext.features
        self.global_pooling = convnext.avgpool
        self.backbone_normalization = convnext.classifier[0]

        self.view_position = nn.Parameter(torch.zeros(1, 16, 768))
        nn.init.trunc_normal_(self.view_position, std=0.02)

        tx_encoder = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=1536,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-6,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=tx_encoder,
            num_layers=4,
            norm=nn.LayerNorm(768, eps=1e-6),
            enable_nested_tensor=False,
        )

        self.view_attention = nn.Linear(768, 1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 17)

        nn.init.zeros_(self.view_attention.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, images):
        features = self.backbone(images)
        features = self.global_pooling(features)
        features = self.backbone_normalization(features)
        features = features.flatten(start_dim=1)

        return features

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        view_position = self.view_position[:, :view_count]

        outputs = view_features + view_position
        outputs = self.transformer(outputs)

        attention_score = self.view_attention(outputs)
        attention_score = attention_score.squeeze(-1)
        attention_weights = torch.softmax(attention_score, dim=1)

        scan_features = outputs * attention_weights.unsqueeze(-1)
        scan_features = scan_features.sum(dim=1)

        scan_features = self.dropout(scan_features)
        result = self.classifier(scan_features)

        return result


# Phase14
# Phase13과 동일한 독립적인 Transformer 4-layer로 적층한 구조를 유지하고 Transformer의 Learning Rate만 1e-4로 낮춥니다.
# Transformer를 4층으로 적층하며 높은 학습률이 최적화의 실패 원인인지 확인하기 위한 실험입니다.
class Phase14(Phase13):
    pass


# Phase15
# Phase12의 독립적인 2-layer Transformer 구조를 유지하고, 학습률을 1e-4로 낮춥니다.
class Phase15(Phase12):
    pass


# Phase16
# Phase10의 1-layer Transformer 구조를 유지하고 학습률을 1e-4로 낮춥니다.
class Phase16(Phase10):
    pass


# Phase17
# Phase16의 1-layer Transformer 구조와 학습 조건을 유지하고, 학습 데이터셋의 무작위 Circular Roll만 제거합니다.
class Phase17(Phase16):
    pass


# Phase18
# Phase16의 1-layer Transformer 구조와 학습 조건을 유지하고, view attention 슬롯 번호 정보를 제거합니다.
class Phase18(Phase16):
    def __init__(self, pretrained=True):
        super().__init__(pretrained=pretrained)

        del self.view_position

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        outputs = self.transformer(view_features)

        attention_score = self.view_attention(outputs)
        attention_score = attention_score.squeeze(-1)
        attention_weights = torch.softmax(attention_score, dim=1)

        scan_features = outputs * attention_weights.unsqueeze(-1)
        scan_features = scan_features.sum(dim=1)

        scan_features = self.dropout(scan_features)
        result = self.classifier(scan_features)

        return result


# Phase19
# Phase17의 구조와 학습 조건을 유지하고, View Attention을 동일한 가중치로 평균내 결합합니다.
class Phase19(Phase17):
    def __init__(self, pretrained=True):
        super().__init__(pretrained=pretrained)

        del self.view_attention

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        view_position = self.view_position[:, :view_count]

        outputs = view_features + view_position
        outputs = self.transformer(outputs)

        scan_features = outputs.mean(dim=1)

        scan_features = self.dropout(scan_features)
        result = self.classifier(scan_features)

        return result


# Phase20
# Phase17은 밀리미터파 스캔 이미지 하나당 하나의 View Attention을 17개 신체 구역이 공유하는 구조입니다.
# Phase19는 밀리미터파 스캔 이미지 하나의 모든 View가 가진 View Attention을 1/16로 평균내어 균등하게 결합합니다.
# Phase20은 17개의 신체 구역이 각자 별도의 View Attention을 가집니다.
class Phase20(Phase17):
    def __init__(self, pretrained=True):
        super().__init__(pretrained=pretrained)

        self.zone_view_attention = nn.Parameter(self.view_attention.weight.detach().repeat(17, 1))
        nn.init.zeros_(self.zone_view_attention)

        del self.view_attention

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        view_position = self.view_position[:, :view_count]

        outputs = view_features + view_position
        outputs = self.transformer(outputs)  # [B, V, 768] => [2, 16, 768]

        attention_score = torch.matmul(outputs, self.zone_view_attention.t())  # [B, 16, 17]
        attention_weights = torch.softmax(attention_score, dim=1)  # [B, 16, 17]

        zone_features = torch.bmm(attention_weights.transpose(1, 2), outputs)
        zone_features = self.dropout(zone_features)  # [B, 17, 768]

        # 신체 구역별 Zone Feature가 필요하기 때문에 Classifier의 행별로 곱셈 수행
        result = (zone_features * self.classifier.weight.unsqueeze(0)).sum(dim=-1)
        result = result + self.classifier.bias

        return result  # [B, 17]


# 마침내! 드디어 우리의 "진짜" 연구를 시작할 수 있습니다!
# 제1 연구 질문: 신체 구역 별로 학습 가능한 Query가 있다면 해당 구역의 위험물을 보다 잘 탐지할 수 있을 것인가?
# 제1 연구 질문 가설: 신체 구역 별로 학습 가능한 Query가 있다면 특정 신체 구역의 위험물을 놓치는 빈도가 줄어들 것이다.


# Phase21: Research Question 1
# Phase20의 Zone별 View Attention을 Multi Head Cross Attention으로 확장합니다.
# Zone Query가 16개 View에서 특정 신체 구역에 필요한 증거를 독립적으로 학습합니다.
class Phase21(Phase17):
    def __init__(self, pretrained=True):
        super().__init__(pretrained=pretrained)

        del self.view_attention

        # Zone Query!
        self.zone_query = nn.Parameter(torch.zeros(1, 17, 768))
        with torch.random.fork_rng(devices=[]):
            self.zone_cross_attention = nn.MultiheadAttention(
                embed_dim=768, num_heads=8, dropout=0, bias=False, batch_first=True
            )

        # 초기에는 모든 View의 평균으로 초기화합니다.
        with torch.no_grad():
            query, key, value = self.zone_cross_attention.in_proj_weight.chunk(3, dim=0)

            nn.init.eye_(query)
            nn.init.eye_(key)
            nn.init.eye_(value)

        nn.init.eye_(self.zone_cross_attention.out_proj.weight)

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        view_position = self.view_position[:, :view_count]

        outputs = view_features + view_position
        outputs = self.transformer(outputs)

        zone_query = self.zone_query.expand(batch_size, -1, -1)

        zone_features, _ = self.zone_cross_attention(query=zone_query, key=outputs, value=outputs, need_weights=False)
        zone_features = self.dropout(zone_features)

        # 각 신체 구역의 Zone Feature에 대응하는 Classifier 행을 적용합니다.
        classifier_weight = self.classifier.weight.unsqueeze(0)
        result = (zone_features * classifier_weight).sum(dim=-1)
        result = result + self.classifier.bias

        return result


# Phase22
# Transformer와 LSTM의 성능 비교를 위한 실험입니다. Phase3의 View Attention을 평균내 결합합니다.
class Phase22(Phase3):
    def __init__(self, pretrained=True):
        super().__init__(pretrained=pretrained)

        del self.view_attention

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        outputs, _ = self.lstm(view_features)

        scan_features = outputs.mean(dim=1)

        scan_features = self.dropout(scan_features)
        result = self.classifier(scan_features)

        return result


# Phase23
# Phase20의 비교 대조군입니다. LSTM에 신체 구역별 View Attention을 가지는 모델을 실험합니다.
class Phase23(Phase3):
    def __init__(self, pretrained=True):
        super().__init__(pretrained=pretrained)

        self.zone_view_attention = nn.Parameter(self.view_attention.weight.detach().repeat(17, 1))
        nn.init.zeros_(self.zone_view_attention)

        del self.view_attention

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        outputs, _ = self.lstm(view_features)

        attention_score = torch.matmul(
            outputs,
            self.zone_view_attention.t(),
        )
        attention_weights = torch.softmax(attention_score, dim=1)

        zone_features = torch.bmm(
            attention_weights.transpose(1, 2),
            outputs,
        )

        zone_features = self.dropout(zone_features)

        classifier_weight = self.classifier.weight.unsqueeze(0)
        result = (zone_features * classifier_weight).sum(dim=-1)
        result = result + self.classifier.bias

        return result


# Phase24
# Phase21의 비교 대조군입니다. LSTM 출력에 Multi-head Zone Query를 적용합니다.
class Phase24(Phase3):
    def __init__(self, pretrained=True):
        super().__init__(pretrained=pretrained)

        del self.view_attention

        self.zone_query = nn.Parameter(torch.zeros(1, 17, 768))

        with torch.random.fork_rng(devices=[]):
            self.zone_cross_attention = nn.MultiheadAttention(
                embed_dim=768,
                num_heads=8,
                dropout=0.0,
                bias=False,
                batch_first=True,
            )

        with torch.no_grad():
            query, key, value = self.zone_cross_attention.in_proj_weight.chunk(3, dim=0)

            nn.init.eye_(query)
            nn.init.eye_(key)
            nn.init.eye_(value)
            nn.init.eye_(self.zone_cross_attention.out_proj.weight)

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        outputs, _ = self.lstm(view_features)

        zone_query = self.zone_query.expand(batch_size, -1, -1)

        zone_features, _ = self.zone_cross_attention(
            query=zone_query,
            key=outputs,
            value=outputs,
            need_weights=False,
        )
        zone_features = self.dropout(zone_features)

        classifier_weight = self.classifier.weight.unsqueeze(0)
        result = (zone_features * classifier_weight).sum(dim=-1)
        result = result + self.classifier.bias

        return result


# Phase25: Research Question 2
# 모델에게 명시적인 각도 정보를 제공하였을 때, 각 뷰와 각도 정보를 결합하여 더 강한 추론을 할 수 있는지 확인합니다.
class Phase25(Phase21):
    def __init__(self, pretrained=True):
        super().__init__(pretrained=pretrained)

        angle_point = torch.arange(16, dtype=torch.float32)
        angle = 2 * torch.pi * angle_point / 16  # 2πθ/16
        angle_features = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        angle_features = angle_features.unsqueeze(0)

        # Checkpoint에 저장
        self.register_buffer("angle_features", angle_features)

        with torch.random.fork_rng(devices=[]):
            self.angle_projection = nn.Linear(in_features=2, out_features=768, bias=False)

        nn.init.zeros_(self.angle_projection.weight)

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        view_position = self.view_position[:, :view_count]
        angle_features = self.angle_features[:, :view_count]
        angle_features = angle_features.to(dtype=view_features.dtype)
        angle_embedding = self.angle_projection(angle_features)

        outputs = view_features + view_position + angle_embedding
        outputs = self.transformer(outputs)

        zone_query = self.zone_query.expand(batch_size, -1, -1)

        zone_features, _ = self.zone_cross_attention(query=zone_query, key=outputs, value=outputs, need_weights=False)
        zone_features = self.dropout(zone_features)

        classifier_weight = self.classifier.weight.unsqueeze(0)

        result = (zone_features * classifier_weight).sum(dim=-1)
        result = result + self.classifier.bias

        return result


# Naming History
# 이전 Phase4a는 현재 Phase4, 이전 Phase4b는 현재 Phase5로 이름을 변경하였습니다.
# 이후 연구 과정에서 이루어지는 실험 조건마다 Phase가 하나씩 증가합니다.
# Unknown 클래스는 초기 Gate 값이 0.5였던 폐기된 실험용 Prototype이었으며,
# Gate의 학습 가능성을 확인하기 위한 모델로 정식 Phase에 포함시키지 않았습니다.
class Unknown(nn.Module):
    """! Not In Use !
    동일한 Transformer layer를 2번 반복하고 게이트 연산을 적용한 사전 실험입니다.
    학습 가능성을 확인했지만, 통제된 조건에서 이루어진 비교가 아니기 때문에서 실험 결과에서 배제하였습니다.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)

        self.backbone = convnext.features
        self.global_pooling = convnext.avgpool
        self.backbone_normalization = convnext.classifier[0]

        self.view_position = nn.Parameter(torch.zeros(1, 16, 768))
        nn.init.trunc_normal_(self.view_position, std=0.02)

        tx_encoder = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=1536,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-6,
        )

        self.transformer = tx_encoder
        self.transformer_normalization = nn.LayerNorm(768, eps=1e-6)

        self.view_attention = nn.Linear(768, 1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 17)

        self.gate = nn.Linear(768 * 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        nn.init.zeros_(self.view_attention.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, images):
        features = self.backbone(images)
        features = self.global_pooling(features)
        features = self.backbone_normalization(features)
        features = features.flatten(start_dim=1)

        return features

    def forward(self, scan):
        batch_size, view_count, channels, height, width = scan.shape

        images = scan.reshape(batch_size * view_count, channels, height, width)

        view_features = self.encode(images)
        view_features = view_features.reshape(batch_size, view_count, 768)

        view_position = self.view_position[:, :view_count]

        outputs = view_features + view_position

        outputs_1 = self.transformer(outputs)
        outputs_2 = self.transformer(outputs_1)

        gate_input = torch.cat([outputs_1, outputs_2], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))

        outputs = (1 - gate) * outputs_1 + gate * outputs_2
        outputs = self.transformer_normalization(outputs)

        attention_score = self.view_attention(outputs)
        attention_score = attention_score.squeeze(-1)
        attention_weights = torch.softmax(attention_score, dim=1)

        scan_features = outputs * attention_weights.unsqueeze(-1)
        scan_features = scan_features.sum(dim=1)

        scan_features = self.dropout(scan_features)
        result = self.classifier(scan_features)

        return result


def test(phase):
    torch.manual_seed(42)

    if phase == 0:
        model = Phase0(pretrained=False)
    elif phase == 1:
        model = Phase1(pretrained=False)  # Optimizer로 SGD를 사용합니다.
    elif phase == 2:
        model = Phase1(pretrained=False)
    elif phase == 3:
        model = Phase3(pretrained=False)
    elif phase == 4:
        model = Phase4(pretrained=False)
    elif phase == 5:
        model = Phase5(pretrained=False)
    elif phase == 6:
        model = Phase6(pretrained=False)
    elif phase == 7:
        model = Phase7(pretrained=False)
    elif phase == 8:
        model = Phase8(pretrained=False)
    elif phase == 9:
        model = Phase9(pretrained=False)
    elif phase == 10:
        model = Phase10(pretrained=False)
    elif phase == 11:
        model = Phase11(pretrained=False)
    elif phase == 12:
        model = Phase12(pretrained=False)
    elif phase == 13:
        model = Phase13(pretrained=False)
    elif phase == 14:
        model = Phase14(pretrained=False)
    elif phase == 15:
        model = Phase15(pretrained=False)
    elif phase == 16:
        model = Phase16(pretrained=False)
    elif phase == 17:
        model = Phase17(pretrained=False)
    elif phase == 18:
        model = Phase18(pretrained=False)
    elif phase == 19:
        model = Phase19(pretrained=False)
    elif phase == 20:
        model = Phase20(pretrained=False)
    elif phase == 21:
        model = Phase21(pretrained=False)
    elif phase == 22:
        model = Phase22(pretrained=False)
    elif phase == 23:
        model = Phase23(pretrained=False)
    elif phase == 24:
        model = Phase24(pretrained=False)
    elif phase == 25:
        model = Phase25(pretrained=False)
    else:
        raise ValueError(f"Unsupported Phase: We don't have Phase{phase}, please check the valid phase.")

    image_size = (661, 512) if phase == 0 else (128, 96)
    # Phase0는 Multi-scale feature map을 그대로 펼치므로 원본 입력 크기를 사용해야 합니다.

    model.train()

    scan = torch.randn(1, 16, 3, *image_size)
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

    if hasattr(model, "lstm"):  # Phase0, Phase1, Phase2 and Phase3
        print(f"LSTM Gradient: {model.lstm.weight_ih_l0.grad.norm().item()}")

    if hasattr(model, "feature_adapter"):  # Phase1 and Phase2
        print(f"Adapter Gradient: {model.feature_adapter.weight.grad.norm().item()}")

    if hasattr(model, "channel_attention"):
        print(f"Channel Attention Gradient: {model.channel_attention.weight.grad.norm().item()}")

    if hasattr(model, "view_attention"):
        print(f"View Attention Gradient: {model.view_attention.weight.grad.norm().item()}")

    if phase in (4, 5, 7, 10, 16, 17, 18, 19, 20, 21, 25):
        gradient = model.transformer.layers[0].self_attn.in_proj_weight.grad
        print(f"Transformer Gradient: {gradient.norm().item()}")
    elif phase in (6, 8, 9, 11):
        gradient = model.transformer.self_attn.in_proj_weight.grad
        print(f"Transformer Gradient: {gradient.norm().item()}")
    elif phase in (12, 13, 14, 15):
        for index, layer in enumerate(model.transformer.layers):
            gradient = layer.self_attn.in_proj_weight.grad
            print(f"Transformer Layer {index + 1} Gradient: {gradient.norm().item()}")

    if hasattr(model, "view_position"):
        print(f"View Position Gradient: {model.view_position.grad.norm().item()}")

    if hasattr(model, "gate"):
        print(f"Gate Gradient: {model.gate.weight.grad.norm().item()}")

    if hasattr(model, "zone_view_attention"):
        print(f"Zone View Attention Gradient: {model.zone_view_attention.grad.norm().item()}")

    if hasattr(model, "zone_cross_attention"):
        gradient = model.zone_cross_attention.in_proj_weight.grad
        print(f"Zone Cross Attention Gradient: {gradient.norm().item()}")

    if hasattr(model, "zone_query"):
        print(f"Zone Query Gradient: {model.zone_query.grad.norm().item()}")

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
    test(phase=25)
