from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import APSDataset
from torch.utils.data import Subset

from src.model import Phase0
from src.model import Phase1
from src.model import Phase2
from src.model import Phase3


# Notes: zero_grad() -> forward -> loss -> backward -> step

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    sample_count = 0

    for batch_index, (scan, labels, _) in enumerate(data_loader):
        scan = scan.to(device, non_blocking=device.type == "cuda")
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
            scan = scan.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")

            result = model(scan)
            loss = criterion(result, labels)

            batch_size = scan.shape[0]

            total_loss += loss.item() * batch_size
            sample_count += batch_size

    average_loss = total_loss / sample_count

    return average_loss


def main(mode="train", phase=0):
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
        model = Phase2(pretrained=True).to(device)
    elif phase == 3:
        model = Phase3(pretrained=True).to(device)
    else:
        raise ValueError(f"Unsupported Phase: We don't have Phase{phase}, please check the valid phase.")

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

        load_saved_model = False

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

        best_validation_loss = float("inf")
        checkpoint = Path(f"models/phase{phase}.pt")
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
            epoch_start_time = perf_counter()

            train_loss = train_one_epoch(model=model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device)
            validation_loss = validate_one_epoch(model=model, data_loader=validation_loader, criterion=criterion, device=device)
            learning_rate = optimizer.param_groups[0]["lr"]

            epoch_end_time = perf_counter()
            epoch_elapsed_time = epoch_end_time - epoch_start_time

            scheduler.step()

            print(f"Epoch {epoch + 1}/{epochs} | ", end="")
            print(f"Train Loss: {train_loss:.5f} | ", end="")
            print(f"Validation Loss: {validation_loss:.5f} | ", end="")
            print(f"Learning Rate: {learning_rate:.5f}")
            print(f"Epoch Elapsed Time: {epoch_elapsed_time / 60:.2f} min")

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
    elif mode == "smoke_test":
        sample = APSDataset(dataset=dataset, data_directory=data_directory, type="train", augment=False)
        sample = Subset(sample, range(5))

        sample_loader = DataLoader(sample, batch_size=1, shuffle=False, num_workers=0)

        sample_epochs = 3
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for epoch in range(sample_epochs):
            sample_loss = train_one_epoch(model=model, data_loader=sample_loader, criterion=criterion, optimizer=optimizer, device=device)

            print(f"Sample Epoch {epoch + 1}/{sample_epochs} | ", end="")
            print(f"Loss: {sample_loss}")

        model.eval()

        sample_ids = []
        sample_probabilities = []

        with torch.inference_mode():
            for scan, labels, scan_id in sample_loader:
                scan = scan.to(device, non_blocking=device.type == "cuda")

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
    main(mode="train", phase=0)
