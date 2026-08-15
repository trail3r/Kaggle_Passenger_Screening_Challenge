from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


def main():
    data = pd.read_csv(Path("data/manifests/manifest.csv"))

    train_data, validation_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

    print("학습 데이터와 검증 데이터 분할 개수 확인")
    print(f"전체 데이터 크기: {data.shape}")
    print(f"학습 데이터 크기: {train_data.shape}")
    print(f"검증 데이터 크기: {validation_data.shape}")
    print()

    # Data Leakage가 존재하는지 확인합니다.
    train_ids = set(train_data["scan_id"])
    validation_ids = set(validation_data["scan_id"])

    if len((train_ids & validation_ids)):
        print("Warning: 데이터 누수가 의심됩니다. 데이터를 확인해 주세요.")
        print()
    else:
        print(f"Pass: 데이터 누수가 없습니다. {len(train_ids) + len(validation_ids)}개 데이터가 분할되었습니다.")
        print()

    train_data["type"] = "train"
    validation_data["type"] = "validation"

    split_data = pd.concat([train_data, validation_data])
    split_data = split_data.sort_values(by="scan_id").reset_index(drop=True)

    output = Path("data/splits/dataset.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    split_data.to_csv(output, index=False)

    print("학습용 데이터를 저장하였습니다.")


if __name__ == "__main__":
    main()
