"""탐색적 데이터 분석(Exploratory Data Analysis)을 수행합니다.

知彼知己 百戰不殆 (지피지기 백전불태).
데이터를 이해해야 이 데이터를 적절히 활용할 수 있는 법입니다. 이 모듈은 EDA를 통해 데이터의 특성을 간략히 파악합니다.

References
----------
이 파일은 2025학년도 2학기에 이정인 교수님께서 강의하신 데이터분석기초 과목의 교안을 참고하여 만들었습니다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.dataset import show_2d_views

ZONE_COLUMNS = [f"zone_{zone}" for zone in range(1, 18)]

def summarize_zones(manifest):
    """Zone별 위험물 양성 데이터 개수와 비율을 확인합니다."""
    danger_zone_counts = manifest[ZONE_COLUMNS].sum()  # Zone별 위험물 양성 데이터 개수를 계산합니다.
    danger_zone_rates = manifest[ZONE_COLUMNS].mean()  # Zone별 위험물 양성 데이터 비율을 계산합니다.

    summary = pd.DataFrame({
        "danger_zone_count": danger_zone_counts,
        "danger_zone_rates": danger_zone_rates
    })

    return summary


def summarize_scans(manifest):
    danger_zone_per_scan = manifest[ZONE_COLUMNS].sum(axis=1)
    danger_zone_distribution = danger_zone_per_scan.value_counts().sort_index()

    return danger_zone_distribution


def zone_cooccurrence(manifest):
    """구역별 동시 발생 행렬을 산출합니다."""

    # 위험 구역 양성 동시 발생 행렬을 시각화해 데이터의 특성을 파악합니다.
    # 인공지능 모델이 동시 발생 행렬에 나타나는 잘못된 패턴을 학습하는 경향이 있는지 확인하기 위해 사용합니다.

    zone_labels = manifest[ZONE_COLUMNS]

    cooccurrence = zone_labels.T.dot(zone_labels)

    return cooccurrence


def plot_zone_cooccurrence(cooccurrence):
    data = cooccurrence.copy()

    for zone in data.columns:  # 자기 자신과의 동시 발생은 제외합니다.
        data.loc[zone, zone] = 0

    figure, axis = plt.subplots(figsize=(10, 8))

    image = axis.imshow(data, cmap="Blues")
    zone_labels = [f"Zone {zone}" for zone in range(1, 18)]

    axis.set_xticks(range(17), labels=zone_labels, rotation=90)
    axis.set_yticks(range(17), labels=zone_labels)

    axis.set_title("Zone Co-occurence")

    figure.colorbar(image, ax=axis, label="Number of scans")

    figure.tight_layout()
    plt.show()
    plt.close(figure)


def sample_scans(manifest):
    """위험 구역 양성 개수가 0, 1, 2, 3개인 사람을 한 명씩 뽑아 시각화합니다."""
    sample_data = manifest.copy()

    sample_data["danger_zone_count"] = sample_data[ZONE_COLUMNS].sum(axis=1)

    samples = []
    for count in range(4):
        suspect = sample_data[sample_data["danger_zone_count"] == count]

        sample = suspect.sample(n=1, random_state=42)
        samples.append(sample)

    samples = pd.concat(samples)

    return samples


if __name__ == "__main__":
    manifest = pd.read_csv(Path("data/manifests/manifest.csv"))

    summary = summarize_zones(manifest)
    distribution = summarize_scans(manifest)
    cooccurrence = zone_cooccurrence(manifest)

    print("구역별 위험 구역 양성 데이터 수와 비율")
    print(summary)
    print()

    # Zone 간 위험 구역 양성 비율은 약 7.8% ~ 11.6%로 큰 차이가 나타나지 않습니다.
    # 하지만 모든 Zone에 대해 위험 구역 음성 비율이 약 88% ~ 92%정도이므로 전체적으로는 위험 구역 양성과 음성 간 불균형이 존재합니다.
    # 따라서 모델이 0만 예측해도 90%에 가까운 정확도가 나타날 수 있기 때문에 모델의 성능을 측정할 때에는 정확도만으로 판단하지 않고,
    # Binary Log Loss 및 Zone별 AUPRC를 함께 확인할 필요가 있습니다.

    print("한 사람당 위험 구역 양성 개수")
    print(distribution)
    print()

    # 한 사람의 스캔에서 위험 구역 양성인 Zone은 0개에서 3개까지 나타납니다.
    # 따라서 train과 validation을 분리할 때 사람별 위험 구역 양성 Zone 수와 Zone별 양성 분포를 함께 고려할 필요가 있습니다.

    print("구역별 동시 발생 위험 구역 양성 데이터 수")
    print(cooccurrence)
    print()

    plot_zone_cooccurrence(cooccurrence)

    samples = sample_scans(manifest)

    print(samples[["scan_id", "aps_path", "danger_zone_count"]])

    for _, sample in samples.iterrows():
        aps_path = Path("data") / sample["aps_path"]
        danger_zone = [zone for zone in ZONE_COLUMNS if sample[zone] == 1]

        print(f"Scan ID: {sample["scan_id"]}")
        print(danger_zone)
        print()

        show_2d_views(aps_path)
