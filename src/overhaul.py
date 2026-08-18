from pathlib import Path

import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import DataLoader

from src.dataset import APSDataset
from src.dataset import prepare_scan_data

from src.model import Phase3
from src.model import Phase4


DATA = Path("data")
DATASET = Path("data/splits/dataset.csv")
RESULT = Path("results")


# Load the trained checkpoint for the specific phase.
def load_model(phase, device):
    if phase == 3:
        model = Phase3(pretrained=False)
    elif phase == "4a":
        model = Phase4(pretrained=False, dim_feedforward=1536)
    else:
        raise ValueError(f"Unsupported Phase: We don't have Phase{phase}, please check the valid phase.")

    checkpoint = torch.load(f"models/phase{phase}.pt", map_location="cpu", weights_only=True, mmap=True)

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    print(f"Phase{phase} Checkpoint Epoch: {checkpoint['epoch']}")
    print(f"Phase{phase} Validation Loss: {checkpoint['best_validation_loss']:.5f}")

    return model


# Combine 16 view features using each phase's sequence module.
def combine_views(model, phase, view_features):
    if phase == 3:
        outputs, _ = model.lstm(view_features)
    elif phase == "4a":
        view_count = view_features.shape[1]
        view_position = model.view_position[:, :view_count]

        outputs = view_features + view_position
        outputs = model.transformer(outputs)
    else:
        raise ValueError(f"Unsupported Phase: {phase}")

    # Convert shared view scores into normalized attention weights.
    attention_score = model.view_attention(outputs)
    attention_score = attention_score.squeeze(-1)
    attention_weights = torch.softmax(attention_score, dim=1)

    scan_features = outputs * attention_weights.unsqueeze(-1)
    scan_features = scan_features.sum(dim=1)

    scan_features = model.dropout(scan_features)
    result = model.classifier(scan_features)

    return result


# Test every circular starting view.
def shift_prediction(model, phase, device, validation_loader):
    scan_ids = []
    labels = []
    shift_logits = []

    with torch.inference_mode():
        for index, (scan, label, scan_id) in enumerate(validation_loader):
            scan = prepare_scan_data(scan, device)

            batch_size, view_count, channels, height, width = scan.shape

            images = scan.reshape(batch_size * view_count, channels, height, width)

            # Extract features.
            view_features = model.encode(images)
            view_features = view_features.reshape(batch_size, view_count, 768)

            rolled_features = []

            # Shift the starting point.
            for shift in range(view_count):
                rolled_view = torch.roll(view_features, shifts=shift, dims=1)

                rolled_features.append(rolled_view)

            rolled_features = torch.stack(rolled_features, dim=1)

            rolled_features = rolled_features.reshape(batch_size * view_count, view_count, 768)

            predict = combine_views(model, phase, rolled_features)

            predict = predict.reshape(batch_size, view_count, 17)

            scan_ids.extend(scan_id)
            labels.append(label.cpu())
            shift_logits.append(predict.cpu())

            if (index + 1) % 10 == 0:
                print(f"Phase{phase} Shift Test: {index + 1}/{len(validation_loader)}")

    labels = torch.cat(labels).numpy().astype(np.int64)
    shift_logits = torch.cat(shift_logits).numpy()

    return scan_ids, labels, shift_logits


def evaluate_predictions(name, labels, probabilities):
    threshold = 0.5
    predictions = (probabilities >= threshold).astype(np.int64)

    flat_labels = labels.ravel()
    flat_probabilities = probabilities.ravel()
    flat_predictions = predictions.ravel()

    tn, fp, fn, tp = confusion_matrix(flat_labels, flat_predictions, labels=[0, 1]).ravel()
    bce_loss = log_loss(flat_labels, flat_probabilities, labels=[0, 1])
    roc_auc = roc_auc_score(labels, probabilities, average="macro")
    pr_auc = average_precision_score(labels, probabilities, average="macro")
    precision = precision_score(flat_labels, flat_predictions, zero_division=0)
    recall = recall_score( flat_labels, flat_predictions, zero_division=0)
    ferrari = f1_score(flat_labels, flat_predictions, zero_division=0)

    print()
    print(name)
    print(f"BCE Log Loss  : {bce_loss:.5f}")
    print(f"Macro ROC-AUC : {roc_auc:.5f}")
    print(f"Macro PR-AUC  : {pr_auc:.5f}")
    print(f"Precision     : {precision:.5f}")
    print(f"Recall        : {recall:.5f}")
    print(f"F1-Score      : {ferrari:.5f}")
    print(f"TP/FP/FN/TN   : {tp}/{fp}/{fn}/{tn}")

    result = {
        "bce_loss": bce_loss,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": ferrari
    }

    return result


# Compare the original view order with shifted view order.
def analyze_shift_predictions(phase, labels, shift_logits):
    # Calculate the probability from raw logit value.
    shift_probabilities = torch.sigmoid(torch.from_numpy(shift_logits)).numpy()

    baseline_probabilities = shift_probabilities[:, 0, :]

    mean_logits = shift_logits.mean(axis=1)
    logit_tta_probabilities = torch.sigmoid(torch.from_numpy(mean_logits)).numpy()
    probability_tta = shift_probabilities.mean(axis=1)  # For reference.

    baseline_result = evaluate_predictions(f"Phase{phase} Original Start View", labels, baseline_probabilities)
    logit_tta_result = evaluate_predictions(f"Phase{phase} Mean Logit TTA", labels, logit_tta_probabilities)
    probability_tta_result = evaluate_predictions(f"Phase{phase} Mean Probability TTA", labels, probability_tta)

    shift_losses = []

    for shift in range(shift_logits.shape[1]):
        probabilities = shift_probabilities[:, shift, :]

        loss = log_loss(labels.ravel(), probabilities.ravel(), labels=[0, 1])

        shift_losses.append(loss)

    shift_losses = np.array(shift_losses)

    probability_std = shift_probabilities.std(axis=1)
    probability_range = np.ptp(shift_probabilities, axis=1)

    positive_mask = labels == 1
    negative_mask = labels == 0

    minimum_probability = shift_probabilities.min(axis=1)
    maximum_probability = shift_probabilities.max(axis=1)

    # Find data that cross the threshold across shifts! Gotcha!
    unstable_predictions = ((minimum_probability < 0.5) & (maximum_probability >= 0.5))

    baseline_predictions = baseline_probabilities >= 0.5
    tta_predictions = logit_tta_probabilities >= 0.5

    # Recommended by Codex: Compare original view order predict result with shifted view order predict.
    fn_to_tp = (positive_mask & ~baseline_predictions & tta_predictions).sum()
    tp_to_fn = (positive_mask & baseline_predictions & ~tta_predictions).sum()
    fp_to_tn = (negative_mask & baseline_predictions & ~tta_predictions).sum()
    tn_to_fp = (negative_mask & ~baseline_predictions & tta_predictions).sum()

    print()
    print(f"Phase{phase} Shift Sensitivity")
    print(f"Shift BCE Min/Mean/Max: {shift_losses.min():.5f} / {shift_losses.mean():.5f} / {shift_losses.max():.5f}")
    print(f"Probability STD All/Positive/Negative: {probability_std.mean():.5f} / {probability_std[positive_mask].mean():.5f} / {probability_std[negative_mask].mean():.5f}")
    print(f"Probability Range All/Positive/Negative: {probability_range.mean():.5f} / {probability_range[positive_mask].mean():.5f} / {probability_range[negative_mask].mean():.5f}")
    print(f"Unstable Predictions: {unstable_predictions.sum()}/{unstable_predictions.size}")
    print(f"FN -> TP: {fn_to_tp}")
    print(f"TP -> FN: {tp_to_fn}")
    print(f"FP -> TN: {fp_to_tn}")
    print(f"TN -> FP: {tn_to_fp}")

    result = {
        "baseline": baseline_result,
        "logit_tta": logit_tta_result,
        "probability_tta": probability_tta_result,
        "shift_losses": shift_losses,
        "probability_std": probability_std,
        "probability_range": probability_range
    }

    return result


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"You are testing by using {device}!")

    validation_dataset = APSDataset(dataset=DATASET, data_directory=DATA, type="validation", augment=False)

    validation_loader = DataLoader(validation_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    phases = [3, "4a"]
    for phase in phases:
        print()
        print(f"Phase{phase} Shift Test")

        model = load_model(phase, device)

        scan_ids, labels, shift_logits = shift_prediction(model, phase, device, validation_loader)

        RESULT.mkdir(parents=True, exist_ok=True)

        output = RESULT / f"phase{phase}_shift_predictions.npz"

        # Save raw shift logits value.
        np.savez_compressed(output, scan_ids=np.array(scan_ids), labels=labels, shift_logits=shift_logits)
        analyze_shift_predictions(phase, labels, shift_logits)

        print()
        print(f"Scan IDs Shape: {np.array(scan_ids).shape}")
        print(f"Labels Shape: {labels.shape}")
        print(f"Shift Logits Shape: {shift_logits.shape}")
        print(f"Shift predictions are saved in {output}!")

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
