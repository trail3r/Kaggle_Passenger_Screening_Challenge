from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

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
from src.model import Phase11
from src.model import Phase12
from src.model import Phase13

DATA_DIRECTORY = Path("data")
DATASET = Path("data/splits/dataset.csv")
RESULT = Path("results")


def synchronize(device):
    """시간을 측정하기 위해 대기 중인 GPU 연산이 끝날 때까지 기다립니다."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main(phase):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        # I hate cpu! It's too slow to train the dataset!

    pin_memory = device.type == "cuda"

    print(f"You are evaluating the model by using {device}!")

    if phase == 0:
        model = Phase0(pretrained=False)
    elif phase == 1:
        model = Phase1(pretrained=False)
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
    else:
        raise ValueError(f"Unsupported Phase: We don't have Phase{phase}, please check the valid phase.")

    print(f"Phase{phase}: {model.__class__.__name__}")

    checkpoint_path = Path(f"models/phase{phase}.pt")

    validation_dataset = APSDataset(dataset=DATASET, data_directory=DATA_DIRECTORY, type="validation", augment=False)
    validation_loader = DataLoader(
        validation_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=pin_memory
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)

    print(f"Checkpoint Epoch: {checkpoint["epoch"]}")
    print(f"Saved Validation Loss: {checkpoint["best_validation_loss"]:.5f}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    del checkpoint  # 메모리 절약을 위해 추론에 불필요한 Checkpoint 객체를 메모리에서 삭제합니다.

    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")

    total_loss = 0.0
    total_label = 0

    scan_ids = []
    labels = []
    probabilities = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    synchronize(device)
    start_time = perf_counter()

    with torch.inference_mode():
        for index, (scan, label, scan_id) in enumerate(validation_loader):
            scan = prepare_scan_data(scan, device)
            label = label.to(device, non_blocking=pin_memory)

            predict = model(scan)
            probability = torch.sigmoid(predict)

            total_loss += criterion(predict, label)
            total_label += label.numel()

            scan_ids.extend(scan_id)
            labels.append(label.cpu())  # Save the result on CPU.
            probabilities.append(probability.cpu())  # Save the result on CPU.

            if (index + 1) % 10 == 0:
                print(f"Evaluation: {index + 1}/{len(validation_loader)}")

    synchronize(device)
    end_time = perf_counter()
    elapsed_time = end_time - start_time

    labels = torch.cat(labels).numpy().astype(np.int64)
    probabilities = torch.cat(probabilities).numpy()

    validation_loss = total_loss.item() / total_label

    threshold = 0.5
    predictions = (probabilities >= threshold).astype(np.int64)

    # Score 계산을 위해 사람*구역 레이블을 1차원으로 펼칩니다.
    flat_labels = labels.ravel()
    flat_predictions = predictions.ravel()

    tn, fp, fn, tp = confusion_matrix(flat_labels, flat_predictions, labels=[0, 1]).ravel()

    false_positive_rate = fp / (fp + tn)  # FPR = 오탐량 / 실제 위험 구역 음성량 (오탐률)
    false_negative_rate = fn / (fn + tp)  # FNR = 미탐량 / 실제 위험 구역 양성량 (미탐률)

    roc_auc = roc_auc_score(
        labels, probabilities, average="macro"
    )  # 모델이 양성과 음성을 얼마나 잘 구분해 순위를 매기는지 측정합니다.
    pr_auc = average_precision_score(labels, probabilities, average="macro")
    # 양성 예측 정확성과 실제 양성 탐지 능력을 평가합니다.
    # 전체 데이터 비율 중 양성 데이터 비율이 10%로 불균형을 가진 데이터셋이기 떄문에 PR-AUC 점수가 보다 현실적인 성능 지표가 될 수 있다 생각합니다.

    precision = precision_score(flat_labels, flat_predictions, zero_division=0)
    recall = recall_score(flat_labels, flat_predictions, zero_division=0)  # recall = 1 - FNR

    f1 = f1_score(flat_labels, flat_predictions, zero_division=0)  # I am a fan of Ferrari F1, Charles Leclerc!

    print()
    print(f"Phase{phase} Validation Result")
    print(f"BCE Log Loss  : {validation_loss:.5f}")
    print(f"Macro ROC-AUC : {roc_auc:.5f}")
    print(f"Macro PR-AUC  : {pr_auc:.5f}")
    print(f"Precision     : {precision:.5f}")
    print(f"Recall        : {recall:.5f}")
    print(f"F1-Score      : {f1:.5f}")
    # F1 Score? Does it mean Formula 1?
    # To Do: Prove that F1 Score is actually a Formula 1 metric.
    # R01 |   Albert Park Grand Prix CIrcuit, Australia  | P1 George Russel (Mercedes)  | P2 Kimi Antonelli (Mercedes)        | P3 Charles Leclerc (Ferrari)
    # R02 |     Shanghai International Circuit, China    | P1 Kimi Antonelli (Mercedes) | P2 George Russell (Mercedes)        | P3 Lewis Hamilton (Ferrari)
    # R03 |             Suzuka Circuit, Japan            | P1 Kimi Antonelli (Mercedes) | P2 Oscar Piastri (McLaren)          | P3 Charles Leclerc (Ferrari)
    # R04 | Miami International Autodrome, United States | P1 Kimi Antonelli (Mercedes) | P2 Lando Norris (McLaren)           | P3 Oscar Piastri (McLaren)
    # R05 |       Circuit Gilles-Villeneuve, Canada      | P1 Kimi Antonelli (Mercedes) | P2 Lewis Hamilton (Ferrari)         | P3 Max Verstappen (Red Bull Racing)
    # R06 |           Circuit de Monaco, Monaco          | P1 Kimi Antonelli (Mercedes) | P2 Lewis Hamilton (Ferrari)         | P3 Pierre Gasly (Alpine)
    # R07 |     Circuit de Barcelona-Catalunya, Spain    | P1 Lewis Hamilton (Ferrari)  | P2 George Russell (Mercedes)        | P3 Lando Norris (McLaren)
    # R08 |            Red Bull Ring, Austria            | P1 George Russell (Mercedes) | P2 Max Verstappen (Red Bull Racing) | P3 Kimi Antonelli (Mercedes)
    # R09 |      Silverstone Circuit, United Kingdom     | P1 Charles Leclerc (Ferrari) | P2 George Russell (Mercedes)        | P3 Lewis Hamilton (Ferrari)
    # R10 |     Circuit de Spa-Francorchamps, Belgium    | P1 Kimi Antonelli (Mercedes) | P2 Charles Leclerc (Ferrari)        | P3 Max Verstappen (Red Bull Racing)
    # R11 |             Hungaroring, Hungary             | P1 Lando Norris (McLaren)    | P2 Max Verstappen (Red Bull Racing) | P3 Kimi Antonelli (Mercedes)
    # R12 |        Circuit Zandvoort, Netherlands        | TBD |
    print(f"FPR           : {false_positive_rate:.5f}")
    print(f"FNR           : {false_negative_rate:.5f}")
    print(f"TP/FP/FN/TN   : {tp}/{fp}/{fn}/{tn}")
    print(f"Elapsed Time  : {elapsed_time:.2f}s")
    print(f"Process Speed : {len(validation_dataset) / elapsed_time:.2f}person/second")

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"Peak GPU Memory: {peak_memory:.2f}GiB")

    RESULT.mkdir(parents=True, exist_ok=True)  # I don't want to re-inference!

    zone_results = []

    print()
    print("Zone Validation Result")

    for zone in range(17):
        section = zone + 1

        zone_labels = labels[:, zone]
        zone_probabilities = probabilities[:, zone]
        zone_predictions = predictions[:, zone]

        zone_tn, zone_fp, zone_fn, zone_tp = confusion_matrix(zone_labels, zone_predictions, labels=[0, 1]).ravel()

        zone_false_positive_rate = zone_fp / (zone_fp + zone_tn)
        zone_false_negative_rate = zone_fn / (zone_fn + zone_tp)

        zone_log_loss = log_loss(zone_labels, zone_probabilities, labels=[0, 1])
        zone_roc_auc = roc_auc_score(zone_labels, zone_probabilities)
        zone_pr_auc = average_precision_score(zone_labels, zone_probabilities)
        zone_precision = precision_score(zone_labels, zone_predictions, zero_division=0)
        zone_recall = recall_score(zone_labels, zone_predictions, zero_division=0)
        zone_f1 = f1_score(zone_labels, zone_predictions, zero_division=0)

        zone_result = {
            "zone": section,
            "positive_count": int(zone_labels.sum()),
            "negative_count": int(len(zone_labels) - zone_labels.sum()),
            "bce_log_loss": zone_log_loss,
            "roc_auc": zone_roc_auc,
            "pr_auc": zone_pr_auc,
            "precision": zone_precision,
            "recall": zone_recall,
            "f1_score": zone_f1,
            "false_positive_rate": zone_false_positive_rate,
            "false_negative_rate": zone_false_negative_rate,
        }

        zone_results.append(zone_result)

        print(f"Zone {section:02d} | ", end="")
        print(f"Loss: {zone_log_loss:.5f} | ", end="")
        print(f"ROC-AUC: {zone_roc_auc:.5f} | ", end="")
        print(f"PR-AUC: {zone_pr_auc:.5f} | ", end="")
        print(f"F1: {zone_f1:.5f}")

    zone_results = pd.DataFrame(zone_results)
    zone_output = RESULT / f"phase{phase}_zone_metrics.csv"
    zone_results.to_csv(zone_output, index=False)

    result = {"scan_id": scan_ids}

    for zone in range(17):
        section = zone + 1

        result[f"zone_{section}_label"] = labels[:, zone]
        result[f"zone_{section}_probability"] = probabilities[:, zone]

    result = pd.DataFrame(result)
    output = RESULT / f"phase{phase}_validation_predictions.csv"
    result.to_csv(output, index=False)

    print(f"Prediction results are successfully saved in {output}!")
    print(f"Zone metrics are successfully saved in {zone_output}!")


if __name__ == "__main__":
    main(phase=13)
