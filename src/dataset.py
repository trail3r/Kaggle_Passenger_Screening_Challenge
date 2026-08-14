""" APS Parser, Visualization Utilities and Manifest Builder.

이 모듈은 Kaggle Passenger Screening Algorithm Challenge에서 제공한 `*.aps` 파일을 읽고
시각화하기 위한 함수, 정답 레이블 핸들링 함수, 학습용 manifest 생성을 위한 함수들을 포함합니다.

하나의 APS 파일에는 한 사람을 여러 방향에서 촬영한 밀리미터파 스캔 데이터가 저장되어 있습니다.
하나의 APS 파일은 512 bytes 크기의 고정 배치 헤더와 그 뒤에 이어지는 1차원 이미지 데이터로 구성됩니다.

512 bytes 크기를 가진 고정 배치 헤더의 이진 데이터가 어떤 의미를 가지는지는 주최 측에서 비공식 함수를 제공함으로써
일부 공개하였습니다. 비공식 함수는 "https://www.kaggle.com/code/wcukierski/reading-images"에서
확인할 수 있습니다. 이 모듈은 주최 측의 비공식 함수 코드와 10위 공개 솔루션을 참고하여 작성되었으며, 프로젝트 목적에
맞게 수정되었습니다.

헤더에 저장된 주요 데이터는 이미지의 자료형을 나타내는 "word_type", 픽셀값에 적용할 "data_scale_factor",
이미지 배열의 크기를 나타내는 "num_x_pts" (이미지의 가로 픽셀 크기), "num_y_pts" (이미지의 세로 픽셀 크기),
"num_t_pts" (이미지 촬영 view의 개수) 등의 데이터가 저장되어 있습니다.

이미지는 헤더 뒤에 1차원 배열로 읽은 뒤 Fortran order로 "(num_x_pts, num_y_pts, num_t_pts)" 형태로
재배열합니다. 데이터셋의 APS 데이터는 일반적으로 (512, 660, 16)의 형태이며, 한 사람당 16개의 2차원 view를
포함합니다.

각 view는 올바른 방향으로 렌더링하기 위해 축을 전치하고, 상하 반전을 수행합니다. 복원된 view는 단순 이미지 또는
회전 애니메이션 GIF 파일로 시각화할 수 있습니다.

References
----------
Kaggle Competition Notebook - Reading Images (Unofficial Functions):
https://www.kaggle.com/code/wcukierski/reading-images

Kaggle Passenger Screening Algorithm Challenge 10th place's solution:
https://github.com/ShayanPersonal/Kaggle-Passenger-Screening-Challenge-Solution

Matplotlib API Reference matplotlib.animation:
https://matplotlib.org/stable/api/animation_api.html

Matplotlib Animation Example Codes:
파이썬 그래프 - 애니메이션 그래프 동영상 파일 저장: https://blog.naver.com/ahn_ss75/222671709830
[python] GIF 그래프 만들어 보기: https://yobbicorgi.tistory.com/33
Matplotlib Animation: https://tech.jehyunlee.dev/2022/08/05/Python-DS-110-anim/

"""


from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomAffine

COLUMNS = [f"zone_{zone}" for zone in range(1, 18)]
FLIPFLOP = {1: 3, 2: 4, 3: 1, 4: 2, 5: 5, 6: 7, 7: 6, 8: 10, 9: 9, 10: 8, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15, 17: 17}


# 학습에 사용할 "*.aps" 데이터를 읽습니다.
def read_header(infile):
    """이미지 데이터의 첫 512 bytes 공간에 저장된 헤더 데이터를 읽습니다."""

    header = {}

    with open(infile, "rb") as aps:
        header["filename"] = b"".join(np.fromfile(aps, dtype="S1", count=20))
        header["parent_filename"] = b"".join(np.fromfile(aps, dtype="S1", count=20))
        header["comments1"] = b"".join(np.fromfile(aps, dtype="S1", count=80))
        header["comments2"] = b"".join(np.fromfile(aps, dtype="S1", count=80))
        header["energy_type"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["config_type"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["file_type"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["trans_type"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["scan_type"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["data_type"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["date_modified"] = b"".join(np.fromfile(aps, dtype="S1", count=16))
        header["frequency"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["mat_velocity"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["num_pts"] = np.fromfile(aps, dtype=np.int32, count=1)
        header["num_polarization_channels"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["spare00"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["adc_min_voltage"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["adc_max_voltage"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["band_width"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["spare01"] = np.fromfile(aps, dtype=np.int16, count=5)
        header["polarization_type"] = np.fromfile(aps, dtype=np.int16, count=4)
        header["record_header_size"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["word_type"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["word_precision"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["min_data_value"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["max_data_value"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["avg_data_value"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["data_scale_factor"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["data_units"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["surf_removal"] = np.fromfile(aps, dtype=np.uint16, count=1)
        header["edge_weighting"] = np.fromfile(aps, dtype=np.uint16, count=1)
        header["x_units"] = np.fromfile(aps, dtype=np.uint16, count=1)
        header["y_units"] = np.fromfile(aps, dtype=np.uint16, count=1)
        header["z_units"] = np.fromfile(aps, dtype=np.uint16, count=1)
        header["t_units"] = np.fromfile(aps, dtype=np.uint16, count=1)
        header["spare02"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["x_return_speed"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["y_return_speed"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["z_return_speed"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["scan_orientation"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["scan_direction"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["data_storage_order"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["scanner_type"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["x_inc"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["y_inc"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["z_inc"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["t_inc"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["num_x_pts"] = np.fromfile(aps, dtype=np.int32, count=1)
        header["num_y_pts"] = np.fromfile(aps, dtype=np.int32, count=1)
        header["num_z_pts"] = np.fromfile(aps, dtype=np.int32, count=1)
        header["num_t_pts"] = np.fromfile(aps, dtype=np.int32, count=1)
        header["x_speed"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["y_speed"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["z_speed"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["x_acc"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["y_acc"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["z_acc"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["x_motor_res"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["y_motor_res"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["z_motor_res"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["x_encoder_res"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["y_encoder_res"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["z_encoder_res"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["date_processed"] = b"".join(np.fromfile(aps, dtype="S1", count=8))
        header["time_processed"] = b"".join(np.fromfile(aps, dtype="S1", count=8))
        header["depth_recon"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["x_max_travel"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["y_max_travel"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["elevation_offset_angle"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["roll_offset_angle"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["z_max_travel"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["azimuth_offset_angle"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["adc_type"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["spare06"] = np.fromfile(aps, dtype=np.int16, count=1)
        header["scanner_radius"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["x_offset"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["y_offset"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["z_offset"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["t_delay"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["range_gate_start"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["range_gate_end"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["ahis_software_version"] = np.fromfile(aps, dtype=np.float32, count=1)
        header["spare_end"] = np.fromfile(aps, dtype=np.float32, count=5)

    return header


def read_data(infile, scale=True):
    """파일 유형에 따른 4가지 이미지 데이터를 읽어 NumPy 배열 형태로 반환합니다."""

    extension = Path(infile).suffix
    header = read_header(infile)

    nx = int(header["num_x_pts"][0])
    ny = int(header["num_y_pts"][0])
    nt = int(header["num_t_pts"][0])

    word_type = int(header["word_type"][0])
    data_scale_factor = header["data_scale_factor"][0]

    with open(infile, "rb") as aps:
        aps.seek(512)  # Skip header

        if extension == ".aps" or extension == ".a3daps":
            if word_type == 7:  # float32
                data = np.fromfile(aps, dtype=np.float32, count=nx * ny * nt)
            elif word_type == 4:  # uint16
                data = np.fromfile(aps, dtype=np.uint16, count=nx * ny * nt)

            data = data.reshape(nx, ny, nt, order="F").copy()  # Make N-d image

            if scale:
                data = data * data_scale_factor  # Scaling factor
            else:
                data = (data, data_scale_factor)
        elif extension == ".a3d":
            if word_type == 7:  # float32
                data = np.fromfile(aps, dtype=np.float32, count=nx * ny * nt)
            elif word_type == 4:  # uint16
                data = np.fromfile(aps, dtype=np.uint16, count=nx * ny * nt)

            data = data * data_scale_factor  # Scaling factor
            data = data.reshape(nx, nt, ny, order="F").copy()  # Make N-d image
        elif extension == ".ahi":
            data = np.fromfile(aps, dtype=np.float32, count=2 * nx * ny * nt)
            data = data.reshape(2, ny, nx, nt, order="F").copy()
            real = data[0, :, :, :].copy()
            imaginary = data[1, :, :, :].copy()

    if extension != ".ahi":
        return data
    else:
        return real, imaginary


# 데이터를 직관적으로 확인하기 위해 시각화합니다.
def get_2d_views(filename, perspectives=16):
    """이미지 데이터를 2d 뷰로 변형합니다."""
    data = read_data(filename)

    # 이미지를 올바른 방향으로 정렬합니다.
    # "*.aps" 파일은 이미지를 (x 좌표, y 좌표, view) 순서로 저장하며, 이를 포트란 배열 순서로 복원해야 합니다.
    # 단순한 시각화에는 matplotlib을 사용하기 때문에 matplotlib의 순서에 맞게 (y 좌표, x 좌표) 순서로 재정렬합니다.
    # 또한, 스캐너 y축 방향과 화면상 렌더링되는 이미지의 세로축 방향이 반대이므로 이미지가 반대로 보입니다.
    # 따라서 flipud() 함수를 사용하여 이미지가 정상적으로 렌더링될 수 있게 상하 반전을 수행합니다.
    views = data.shape[2]
    return [np.flipud(data[:, :, view_index].T) for view_index in range(0, views, views // perspectives)]


def show_2d_views(infile, perspectives=16):
    """matplotlib를 활용해 이미지를 렌더링합니다."""
    images = get_2d_views(infile, perspectives)

    grid = {1: (1, 1), 2: (1, 2), 4: (2, 2), 8: (2, 4), 16: (4, 4)}
    rows, columns = grid[perspectives]

    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.5, rows * 2), squeeze=False)

    step = 16 // perspectives  # Select views at uniform intervals.
    for index, (axis, image) in enumerate(zip(axes.flat, images)):
        index = index * step

        axis.imshow(image, cmap="gray")
        axis.set_title(f"View {index}")
        axis.axis("off")

    figure.tight_layout()
    plt.show()
    plt.close(figure)


def animate(infile, outfile, perspectives=16):
    """matplotlib을 활용해 view별 이미지를 순차적으로 렌더링하는 GIF 파일을 생성합니다."""
    images = get_2d_views(infile, perspectives)

    figure, axis = plt.subplots(figsize=(5, 6))

    image = axis.imshow(images[0], cmap="gray")
    title = axis.set_title("View 0")
    axis.axis("off")

    step = 16 // perspectives

    def update(frame):
        index = frame * step

        image.set_data(images[frame])
        title.set_text(f"View {index}")

        return image, title

    animation = FuncAnimation(figure, update, frames=len(images), interval=250, blit=False)
    animation.save(outfile, writer=PillowWriter(fps=4), dpi=100)
    plt.close(figure)


# 데이터의 정답 레이블을 읽습니다.
def read_labels(infile):
    labels = pd.read_csv(infile)

    labels[["scan_id", "zone"]] = labels["Id"].str.split("_", expand=True)
    labels["zone"] = labels["zone"].str[4:].astype(int)

    label = labels.pivot(index="scan_id", columns="zone", values="Probability")

    return label


# 학습에 사용할 데이터의 메타데이터(manifest)를 생성합니다.
def build_manifest(aps_directory, label_file):
    """학습 데이터의 메타데이터(manifest)를 생성합니다."""
    aps_directory = Path(aps_directory)
    label_file = Path(label_file)

    aps_files = sorted(aps_directory.glob("*.aps"))  # "*.aps" 파일을 탐색합니다.

    records = []
    for aps_file in aps_files:
        data = {
            "scan_id": aps_file.stem,  # 파일명에서 스캔 ID를 추출합니다.
            "aps_path": aps_file.relative_to(aps_directory.parent).as_posix()  # 학습용 파일의 파일 경로를 저장합니다.
        }

        records.append(data)

    aps_dataframe = pd.DataFrame(records)

    labels = read_labels(label_file)
    labels.columns = [f"zone_{zone}" for zone in labels.columns]  # 컬럼명을 직관적으로 변경합니다.
    labels_dataframe = labels.reset_index()

    manifest_dataframe = aps_dataframe.merge(labels_dataframe, on="scan_id", how="inner")
    manifest_dataframe = manifest_dataframe.sort_values(by="scan_id").reset_index(drop=True)

    return manifest_dataframe


def flip(scan, labels):
    flipped_scan = torch.flip(scan, dims=[3])
    flipped_scan = torch.flip(flipped_scan, dims=[0])
    flipped_scan = torch.roll(flipped_scan, shifts=1, dims=0)

    flipped_labels = torch.empty_like(labels)

    for target_zone, source_zone in FLIPFLOP.items():
        flipped_labels[target_zone - 1] = labels[source_zone - 1]

    return flipped_scan, flipped_labels


# For PyTorch
class APSDataset(Dataset):
    def __init__(self, dataset, data_directory, type, augment=False):
        data = pd.read_csv(dataset)
        self.data = data[data["type"] == type].reset_index(drop=True)
        self.data_directory = Path(data_directory)

        self.augment = augment
        self.random_affine = RandomAffine(degrees=15, translate=(0.01, 0.01), scale=(0.95, 1.05), interpolation=InterpolationMode.NEAREST)

    def __getitem__(self, index):
        data = self.data.iloc[index]

        scan_id = data["scan_id"]
        aps_path = self.data_directory / data["aps_path"]  # Path Object

        views = get_2d_views(aps_path)
        scan = np.stack(views)

        scan = np.pad(scan, ((0, 0), (0, 1), (0, 0)), mode="constant")

        scan = torch.from_numpy(scan).float()
        scan = scan.unsqueeze(1)

        labels = data[COLUMNS].to_numpy(dtype=np.float32)
        labels = torch.from_numpy(labels)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                # 50% 확률로 Flip
                scan, labels = flip(scan, labels)

            # Circular Roll
            shift = torch.randint(0, scan.shape[0], (1, )).item()
            scan = torch.roll(scan, shifts=shift, dims=0)

            scan = self.random_affine(scan)

        return scan, labels, scan_id

    def __len__(self):  # Data 길이 확인
        return len(self.data)


if __name__ == "__main__":
    aps_directory = Path("data/stage1_aps")
    sample_data = sorted(aps_directory.glob("*.aps"))[0]
    stage1_labels_csv = Path("data/stage1_labels.csv")

    header = read_header(sample_data)

    print(f"샘플 데이터에 저장된 헤더 정보:")
    print(f"word_type: {header["word_type"][0]}")
    print(f"data_scale_factor: {header["data_scale_factor"][0]}")
    print(f"num_x_pts: {header["num_x_pts"][0]}")
    print(f"num_y_pts: {header["num_y_pts"][0]}")
    print(f"num_t_pts: {header["num_t_pts"][0]}")
    print()

    data = read_data(sample_data)  # 샘플 데이터에 저장된 이미지 데이터를 확인합니다.

    print("샘플 데이터 이미지 정보 확인")
    print(data.shape)
    print(data.dtype)
    print(data.min(), data.max())
    print()

    # show_2d_views(sample_data)  # 샘플 데이터를 시각화합니다.
    # animate(sample_data, "aps_rotation.gif")  # 샘플 데이터를 GIF로 만들어 view별로 시각화합니다.

    # 정답 레이블을 읽습니다.
    labels = read_labels(stage1_labels_csv)

    print("정답 레이블")
    print(labels.head())
    print()
    print(labels.shape)
    print()
    print(labels.columns)
    print()

    sample_scan_id = labels.index[0]  # 첫 번째 데이터의 정답을 확인합니다.
    sample_data_label = labels.loc[sample_scan_id]

    print("샘플 데이터 정답 레이블")
    print(sample_scan_id)
    print(sample_data_label.to_list())
    print()

    scan_id = sample_data.stem
    scan_labels = labels.loc[scan_id].to_numpy()

    print(scan_id)
    print(scan_labels)
    print(scan_labels.shape)
    print()

    # Manifest 생성
    output = Path("data/manifests/manifest.csv")

    manifest = build_manifest(aps_directory, stage1_labels_csv)

    print(manifest.head())
    print(manifest.shape)
    print(manifest["scan_id"].is_unique)  # 중복된 ID가 있는지 확인합니다.
    print(manifest.isna().sum().sum())  # 결측치를 확인합니다.

    zone_columns = [f"zone_{zone}" for zone in range(1, 18)]
    print(manifest[zone_columns].stack().unique())  # zone별 정답에 0과 1 이외의 값이 있는지 확인합니다.

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)

    print(f"Manifest를 {output}에 저장하였습니다.")
    print()

    dataset = Path("data/splits/dataset.csv")
    data_directory = Path("data")

    train_dataset = APSDataset(dataset=dataset, data_directory=data_directory, type="train")
    validation_dataset = APSDataset(dataset=dataset, data_directory=data_directory, type="validation")

    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(validation_dataset)}")
    print()

    scan, labels, scan_id = train_dataset[0]

    print(f"Scan ID: {scan_id}")
    print(f"Scan shape: {scan.shape}")
    print(f"Scan dtype: {scan.dtype}")
    print(f"Scan range: {scan.min()} ~ {scan.max()}")
    print(f"Label shape: {labels.shape}")
    print(f"Label dtype: {labels.dtype}")
    print(f"Labels: {labels}")
