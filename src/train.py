from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import APSDataset
from src.model import Phase0

# Notes: zero_grad() -> forward -> loss -> backward -> step

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    sample_count = 0

    for batch_index, (scan, labels, _) in enumerate(data_loader):
        scan = scan.repeat(1, 1, 3, 1, 1)  # GrayScale 입력을 3번 반복해 3채널 입력으로 생성합니다.

        scan = scan.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        result = model(scan)
        loss = criterion(result, labels)

        loss.backward()
        optimizer.step()

        if (batch_index + 1) % 10 == 0:
            print(f"Batch {batch_index + 1}/{len(data_loader)} | ", end="")
            print(f"Loss: {loss:.5f}")

        batch_size = scan.shape[0]

        total_loss += loss.item() * batch_size
        sample_count += batch_size

    average_loss = total_loss / sample_count

    return average_loss


def validate_one_epoch(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for scan, labels, _ in data_loader:
            scan = scan.to(device)
            labels = labels.to(device)

            scan = scan.repeat(1, 1, 3, 1, 1)

            result = model(scan)
            loss = criterion(result, labels)

            batch_size = scan.shape[0]

            total_loss += loss.item() * batch_size
            sample_count += batch_size

    average_loss = total_loss / sample_count

    return average_loss


# Which device are you using?
if torch.cuda.is_available():

    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    # I hate cpu! It's too slow to train the dataset!

print(f"You are training by using {device}!")

dataset = Path("data/splits/dataset.csv")
data_directory = Path("data")

train_dataset = APSDataset(dataset=dataset, data_directory=data_directory, type="train", augment=True)
validation_dataset = APSDataset(dataset=dataset, data_directory=data_directory, type="validation", augment=False)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0)
# Validation은 결과 재현을 위해 shuffle을 False로 설정하였습니다.

smoke_dataset = Subset(train_dataset, range(10))
validation_smoke_dataset = Subset(validation_dataset, range(10))

smoke_loader = DataLoader(smoke_dataset, batch_size=1, shuffle=False, num_workers=0)
validation_smoke_loader = DataLoader(validation_smoke_dataset, batch_size=1, shuffle=False, num_workers=0)

torch.manual_seed(42)  # 재현성을 높이기 위해 seed를 42로 고정합니다.

model = Phase0(pretrained=True).to(device)

# Hyperparameters.
epochs = 30
patience = 5
early_stop_point = 0

load_saved_model = False
start_epoch = 0

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

best_validation_loss = float("inf")
checkpoint = Path(f"models/phase0_{device.type}_best.pt")
checkpoint.parent.mkdir(parents=True, exist_ok=True)

if load_saved_model:
    saved_checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)

    model.load_state_dict(saved_checkpoint["model_state_dict"])
    optimizer.load_state_dict(saved_checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(saved_checkpoint["scheduler_state_dict"])
    start_epoch = saved_checkpoint["epoch"]
    best_validation_loss = saved_checkpoint["best_validation_loss"]
    early_stop_point = saved_checkpoint["early_stop_point"]

    print(f"모델 학습이 {start_epoch}epoch부터 다시 시작되었습니다.")

for epoch in range(start_epoch, epochs):
    train_loss = train_one_epoch(model=model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device)

    validation_loss = validate_one_epoch(model=model, data_loader=validation_loader, criterion=criterion, device=device)
    learning_rate = optimizer.param_groups[0]["lr"]

    scheduler.step()

    print(f"Epoch {epoch + 1}/{epochs} | ", end="")
    print(f"Train Loss: {train_loss:.5f} | ", end="")
    print(f"Validation Loss: {validation_loss:.5f} | ", end="")
    print(f"Learning Rate: {learning_rate:.5f}")

    # Checkpoint
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        early_stop_point = 0

        checkpoint_state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_loss": best_validation_loss,
            "early_stop_point": early_stop_point,
            "train_loss": train_loss,
            "validation_loss": validation_loss
        }

        torch.save(checkpoint_state, checkpoint)

        print(f"Best Model Saved: {best_validation_loss:.5f}")
    else:
        early_stop_point += 1

        print(f"No Improvement: ", end="")
        print(f"{early_stop_point}/{patience}")

        if early_stop_point >= patience:
            print("Early Stopping!")
            break







# train_loss = train_one_epoch(model=model, data_loader=smoke_loader, criterion=criterion, optimizer=optimizer, device=device)

# print(f"Smoke train loss: {train_loss}")

# validation_loss = validate_one_epoch(model=model, data_loader=validation_smoke_loader, criterion=criterion, device=device)

# print(f"Validation Smoke train loss: {validation_loss}")



# scan, labels, scan_ids = next(iter(train_loader))

# print(f"Scan IDs: {scan_ids[0]}")
# print(f"Scan shape: {scan.shape}")
# print(f"Label shape: {labels.shape}")
# print()

# scan = scan.repeat(1, 1, 3, 1, 1)
# print(f"RGB scaled scan shape: {scan.shape}")
# print()

# torch.manual_seed(42)  # 재현 가능성을 우선으로 판단하였기 떄문에 Baseline 모델의 seed를 42로 고정합니다.

# model = Phase0(pretrained=False)
# model.train()

# criterion = torch.nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# """모델이 정상적으로 빌드되었는지 확인하기 위해 1번의 스텝만 학습을 실험적으로 실행합니다.
# weight_before = model.classifier.weight.detach().clone()

# optimizer.zero_grad()

# result = model(scan)
# loss = criterion(result, labels)

# loss.backward()
# optimizer.step()

# backbone_gradient = model.backbone[0].weight.grad.norm().item()
# lstm_gradient = model.lstm.weight_ih_l0.grad.norm().item()

# gradient_norm = model.classifier.weight.grad.norm().item()

# weight_change = (weight_before - model.classifier.weight.detach()).abs().sum().item()

# print(f"Loss: {loss}")
# print(f"Classifier gradient norm: {gradient_norm}")
# print(F"Classifier weight change: {weight_change}")
# print()
# print(f"Backbone gradient norm: {backbone_gradient}")
# print(f"LSTM gradient norm: {lstm_gradient}") """

# for step in range(20):
#     optimizer.zero_grad()

#     result = model(scan)
#     loss = criterion(result, labels)

#     loss.backward()
#     optimizer.step()

#     print(f"Step {step + 1}: {loss:.6f}")
