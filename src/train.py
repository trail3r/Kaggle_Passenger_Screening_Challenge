from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

from src.dataset import APSDataset
from src.dataset import prepare_scan_data

from src.model import Phase0
from src.model import Phase1
from src.model import Phase3
from src.model import Phase4
from src.model import Phase5
from src.model import Phase6
from src.model import Phase7
from src.model import Phase8
from src.model import Phase9
from src.model import Phase10


# Notes: zero_grad() -> forward -> loss -> backward -> step

KST = ZoneInfo("Asia/Seoul")


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    sample_count = 0

    for batch_index, (scan, labels, _) in enumerate(data_loader):
        scan = prepare_scan_data(scan, device)
        labels = labels.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)  # OMG! I need more VRAM!

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
            scan = prepare_scan_data(scan, device)
            labels = labels.to(device, non_blocking=device.type == "cuda")

            result = model(scan)
            loss = criterion(result, labels)

            batch_size = scan.shape[0]

            total_loss += loss.item() * batch_size
            sample_count += batch_size

    average_loss = total_loss / sample_count

    return average_loss


def adaptive_optimizer(model, phase):
    """Phase별로 적절한 optimizer를 선택합니다."""
    if phase in (0, 1):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)

        return optimizer

    # ConvNeXt는 AdamW를 Optimizer로 사용하였으며, ImageNet fine-tuning에서 매우 작은 learning rate를 사용했습니다.
    # 이 아이디어를 가져와 사전학습 Backbone ConvNeXt-Tiny 학습에는 5e-5, 그 뒤의 새로운 레이어에는 이보다 약간 큰 5e-4로 설정하였습니다.
    backbone_parameters = []
    task_parameters = []

    for name, parameter in model.named_parameters():
        if name.startswith("backbone"):
            backbone_parameters.append(parameter)
        else:
            task_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": 5e-5
            },
            {
                "params": task_parameters,
                "lr": 5e-4
            }
        ],
        betas=(0.9, 0.999),
        weight_decay=0.05
    )

    # !Temp! Test Code!
    if phase in (7, 8, 9, 10):
        transformer_lr = 2e-4 if phase in (7, 10) else 2.5e-4

        backbone_parameters = []
        transformer_parameters = []
        task_parameters = []

        for name, parameter in model.named_parameters():
            if name.startswith("backbone"):
                backbone_parameters.append(parameter)
            elif name.startswith("transformer"):
                transformer_parameters.append(parameter)
            else:
                task_parameters.append(parameter)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": 5e-5
                },
                {
                    "params": transformer_parameters,
                    "lr": transformer_lr
                },
                {
                    "params": task_parameters,
                    "lr": 5e-4
                }
            ],
            betas=(0.9, 0.999),
            weight_decay=0.05
        )



    print(f"Optimizer: {optimizer.__class__.__name__}")

    for index, parameter_group in enumerate(optimizer.param_groups):
        print(f"Parameter Group {index + 1}")
        print(f"Learning Rate: {parameter_group['lr']}")

    return optimizer


def adaptive_scheduler(optimizer, phase, epochs):
    """Phase별로 적절한 learning rate scheduler를 선택합니다."""
    if phase in (0, 1):
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

        return scheduler

    warmup_epochs = 5

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=0.0)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    return scheduler


def main(phase, mode="train"):
    torch.manual_seed(42)  # 재현성을 높이기 위해 seed를 42로 고정합니다.

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        # I hate cpu! It's too slow to train the dataset!

    pin_memory = device.type == "cuda"

    print(f"You are training by using {device}!")

    dataset = Path("data/splits/dataset.csv")
    data_directory = Path("data")

    if phase == 0:
        model = Phase0(pretrained=True).to(device)
    elif phase == 1:
        model = Phase1(pretrained=True).to(device)
    elif phase == 2:
        model = Phase1(pretrained=True).to(device)
    elif phase == 3:
        model = Phase3(pretrained=True).to(device)  # ConvNeXt 특징을 LSTM으로 순차 처리하고, 하나의 평가 함수를 모든 View에 공유합니다.
    elif phase == 4:
        model = Phase4(pretrained=True).to(device)
    elif phase == 5:
        model = Phase5(pretrained=True).to(device)
    elif phase == 6:
        model = Phase6(pretrained=True).to(device)
    elif phase == 7:
        model = Phase7(pretrained=True).to(device)
    elif phase == 8:
        model = Phase8(pretrained=True).to(device)
    elif phase == 9:
        model = Phase9(pretrained=True).to(device)
    elif phase == 10:
        model = Phase10(pretrained=True).to(device)
    else:
        raise ValueError(f"Unsupported Phase: We don't have Phase{phase}, please check the valid phase.")

    print(f"Phase{phase}: {model.__class__.__name__}")

    if mode == "train":
        train_dataset = APSDataset(dataset=dataset, data_directory=data_directory, type="train", augment=True)
        validation_dataset = APSDataset(dataset=dataset, data_directory=data_directory, type="validation", augment=False)

        # Match the 10th-place solution's loader configuration. persistent_workers
        # avoids recreating eight Windows worker processes after every epoch.
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=pin_memory, persistent_workers=True, drop_last=True)
        validation_loader = DataLoader(validation_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=pin_memory)
        # Validation은 결과 재현을 위해 shuffle을 False로 설정하였습니다.

        # Hyperparameters.
        epochs = 401
        patience = 10
        early_stop_point = 0
        start_epoch = 0
        best_epoch = 0

        load_saved_model = False

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = adaptive_optimizer(model, phase)
        scheduler = adaptive_scheduler(optimizer=optimizer, phase=phase, epochs=epochs)

        best_validation_loss = float("inf")
        checkpoint = Path(f"models/phase{phase}.pt")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

        if load_saved_model:
            saved_checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)

            model.load_state_dict(saved_checkpoint["model_state_dict"])
            optimizer.load_state_dict(saved_checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(saved_checkpoint["scheduler_state_dict"])
            start_epoch = saved_checkpoint["epoch"]
            best_epoch = saved_checkpoint["epoch"]
            best_validation_loss = saved_checkpoint["best_validation_loss"]
            early_stop_point = saved_checkpoint["early_stop_point"]

            print(f"모델 학습이 {start_epoch}epoch부터 다시 시작되었습니다.")

        train_start_time_kst = datetime.now(KST)
        train_start_time = perf_counter()

        print()
        print(f"Train started at {train_start_time_kst:%Y-%m-%d %H:%M:%S %Z}")

        for epoch in range(start_epoch, epochs):
            epoch_start_time = perf_counter()

            train_loss = train_one_epoch(model=model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device)
            validation_loss = validate_one_epoch(model=model, data_loader=validation_loader, criterion=criterion, device=device)
            learning_rates = [parameter_group["lr"] for parameter_group in optimizer.param_groups]

            epoch_end_time = perf_counter()
            epoch_elapsed_time = epoch_end_time - epoch_start_time

            epoch_elapsed_time_minutes = int(epoch_elapsed_time // 60)
            epoch_elapsed_time_seconds = epoch_elapsed_time % 60

            scheduler.step()

            print(f"Epoch {epoch + 1}/{epochs} | ", end="")
            print(f"Train Loss: {train_loss:.5f} | ", end="")
            print(f"Validation Loss: {validation_loss:.5f} | ", end="")

            if phase in (0, 1):
                print(f"Learning Rate: {learning_rates[0]:.5f}")
            elif phase in (7, 8, 9, 10):
                print(f"Backbone Learning Rate: {learning_rates[0]:.2e} | Transformer Learning Rate: {learning_rates[1]:.2e} | Task Learning Rate: {learning_rates[2]:.2e}")
            else:
                print(f"Backbone Learning Rate: {learning_rates[0]:.2e} | Task Learning Rate: {learning_rates[1]:.2e}")

            print(f"Epoch Elapsed Time: {epoch_elapsed_time_minutes}min {epoch_elapsed_time_seconds:.2f}sec")

            # Checkpoint
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch + 1
                early_stop_point = 0

                checkpoint_state = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_validation_loss": best_validation_loss,
                    "early_stop_point": early_stop_point,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }

                torch.save(checkpoint_state, checkpoint)

                print(f"Best Model Saved: {best_validation_loss:.5f}")
            else:
                early_stop_point += 1

                print("No Improvement: ", end="")
                print(f"{early_stop_point}/{patience}")

                if early_stop_point >= patience:
                    print("Early Stopping!")
                    break

        train_end_time_kst = datetime.now(KST)
        train_end_time = perf_counter()
        train_elapsed_time = train_end_time - train_start_time

        print()
        print(f"Train finished at {train_end_time_kst:%Y-%m-%d %H:%M:%S %Z}")
        print(f"Total Training Time: {train_elapsed_time / 3600:.2f} hours")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Validation Loss: {best_validation_loss}")
    elif mode == "smoke_test":
        sample = APSDataset(dataset=dataset, data_directory=data_directory, type="train", augment=False, sample_count=5)

        sample_loader = DataLoader(sample, batch_size=1, shuffle=False, num_workers=0)

        sample_epochs = 3
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = adaptive_optimizer(model, phase)
        scheduler = adaptive_scheduler(optimizer=optimizer, phase=phase, epochs=100)

        for epoch in range(sample_epochs):
            learning_rates = [parameter_group["lr"] for parameter_group in optimizer.param_groups]

            if phase in (0, 1):
                print(f"Learning Rate: {learning_rates[0]:.5f}")
            elif phase in (7, 8, 9, 10):
                print(f"Backbone Learning Rate: {learning_rates[0]:.2e} | Transformer Learning Rate: {learning_rates[1]:.2e} | Task Learning Rate: {learning_rates[2]:.2e}")
            else:
                print(f"Backbone Learning Rate: {learning_rates[0]:.2e}")
                print(f"Task Learning Rate: {learning_rates[1]:.2e}")

            sample_loss = train_one_epoch(model=model, data_loader=sample_loader, criterion=criterion, optimizer=optimizer, device=device)

            scheduler.step()

            print(f"Sample Epoch {epoch + 1}/{sample_epochs} | Loss: {sample_loss}")

        model.eval()

        sample_ids = []
        sample_probabilities = []

        with torch.inference_mode():
            for scan, labels, scan_id in sample_loader:
                scan = prepare_scan_data(scan, device)

                result = model(scan)
                probability = torch.sigmoid(result)

                sample_ids.extend(scan_id)
                sample_probabilities.append(probability.cpu())

        sample_probabilities = torch.cat(sample_probabilities, dim=0)

        prediction_difference = sample_probabilities.std(dim=0).mean().item()

        print()
        print("Smoke Test Predictions")

        for scan_id, probability in zip(sample_ids, sample_probabilities):
            print(f"{scan_id} | Mean Probability: {probability.mean().item():.5f}")

        print(f"Mean Prediction Difference: {prediction_difference:.5f}")

    return 0


if __name__ == "__main__":
    main(phase=10)
