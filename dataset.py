from torcheeg import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torcheeg.datasets import DEAPDataset, SEEDDataset, DREAMERDataset
from torcheeg.datasets.constants.emotion_recognition.deap import (
    DEAP_CHANNEL_LOCATION_DICT,
)
from torcheeg.datasets.constants.emotion_recognition.seed import (
    SEED_CHANNEL_LOCATION_DICT,
)
from torcheeg.datasets.constants.emotion_recognition.dreamer import (
    DREAMER_CHANNEL_LOCATION_DICT,
)
from torcheeg.model_selection import LeaveOneSubjectOut

import torch
from utils import set_seed

NUM_WORKER = 8

# ---------- DISCRETIZER ----------
from typing import Dict, List, Union, Tuple
from torcheeg.transforms.base_transform import LabelTransform

class Discretizer(LabelTransform):
    def __init__(self, thresholds: List[float]):
        super(Discretizer, self).__init__()
        self.thresholds = sorted(thresholds)  # Ensure thresholds are sorted

    def __call__(
        self, *args, y: Union[int, float, List[Union[int, float]]], **kwargs
    ) -> Union[int, List[int]]:
        return super().__call__(*args, y=y, **kwargs)

    def apply(
        self, y: Union[int, float, List[Union[int, float]]], **kwargs
    ) -> Union[int, List[int]]:
        if isinstance(y, list):
            return [self.classify_value(value) for value in y]
        return self.classify_value(y)

    def classify_value(self, value: Union[int, float]) -> int:
        # Iterate through the thresholds to determine the bin
        for i, threshold in enumerate(self.thresholds):
            if value < threshold:
                return i
        return len(self.thresholds)  # Value is greater than the last threshold

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{"thresholds": self.thresholds})

# ---------- BINARY DATASET ----------

import os
from typing import Any, Callable, Union
from torcheeg.datasets import SEEDDataset
import scipy.io as scio


class SEEDBinaryDataset(SEEDDataset):
    def process_record(
        self,
        file: Any = None,
        root_path: str = "./Preprocessed_EEG",
        chunk_size: int = 200,
        overlap: int = 0,
        num_channel: int = 62,
        before_trial: Union[None, Callable] = None,
        offline_transform: Union[None, Callable] = None,
        **kwargs,
    ):
        file_name = file

        subject = int(
            os.path.basename(file_name).split(".")[0].split("_")[0]
        )  # subject (15)
        date = int(
            os.path.basename(file_name).split(".")[0].split("_")[1]
        )  # period (3)

        samples = scio.loadmat(
            os.path.join(root_path, file_name), verify_compressed_data_integrity=False
        )  # trial (15), channel(62), timestep(n*200)
        # label file
        labels = scio.loadmat(
            os.path.join(root_path, "label.mat"), verify_compressed_data_integrity=False
        )["label"][0]

        trial_ids = [key for key in samples.keys() if "eeg" in key]

        write_pointer = 0
        # loop for each trial
        for trial_id in trial_ids:
            # Skip trials with label == 0 (neutral)
            trial_label = int(labels[int(trial_id.split("_")[-1][3:]) - 1])
            if trial_label == 0:
                continue  # Skip this trial

            trial_samples = samples[trial_id]  # channel(62), timestep(n*200)
            if before_trial:
                trial_samples = before_trial(trial_samples)

            trial_meta_info = {
                "subject_id": subject,
                "trial_id": trial_id,
                "emotion": trial_label,
                "date": date,
            }

            start_at = 0
            if chunk_size <= 0:
                dynamic_chunk_size = trial_samples.shape[1] - start_at
            else:
                dynamic_chunk_size = chunk_size

            end_at = dynamic_chunk_size
            step = dynamic_chunk_size - overlap

            while end_at <= trial_samples.shape[1]:
                clip_sample = trial_samples[:num_channel, start_at:end_at]

                t_eeg = clip_sample
                if offline_transform is not None:
                    t_eeg = offline_transform(eeg=clip_sample)["eeg"]

                clip_id = f"{file_name}_{write_pointer}"
                write_pointer += 1

                record_info = {
                    "start_at": start_at,
                    "end_at": end_at,
                    "clip_id": clip_id,
                }
                record_info.update(trial_meta_info)
                yield {"eeg": t_eeg, "key": clip_id, "info": record_info}

                start_at = start_at + step
                end_at = start_at + dynamic_chunk_size


class BinaryDataset(Dataset):
    def __init__(self, dataset, with_subject_id=False):
        self.dataset = dataset
        try:
            self.dataset_name = dataset.dataset.__class__.__name__.lower().replace(
                "dataset", ""
            )
        except:
            self.dataset_name = dataset.__class__.__name__.lower().replace(
                "dataset", ""
            )

        if self.dataset_name == "deap":
            self.threshold = 5
        elif self.dataset_name == "seedbinary":
            self.threshold = 0
        elif self.dataset_name == "dreamer":
            self.threshold = 3

        self.with_subject_id = with_subject_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]

        # convert X into 3D for topomap
        if len(X.shape) == 4:
            n_bands, w, h, color_channels = X.shape
            X = X.transpose(0, 3).reshape(-1, w, h)

        # convert y into discrete
        subject_ids, valences = y
        processed_y = self.preprocess_y(valences)

        if self.with_subject_id:
            return X, processed_y, subject_ids
        else:
            return X, processed_y

    def preprocess_y(self, y):
        transform = transforms.Binary(threshold=self.threshold)
        return transform(y=y)["y"]


# ---------- TERNARY DATASET ----------


class TernaryDataset(Dataset):
    def __init__(self, dataset, with_subject_id=False):
        self.dataset = dataset
        try:
            self.dataset_name = dataset.dataset.__class__.__name__.lower().replace(
                "dataset", ""
            )
        except:
            self.dataset_name = dataset.__class__.__name__.lower().replace(
                "dataset", ""
            )

        if self.dataset_name == "deap":
            self.thresholds = [4, 6]
        elif self.dataset_name == "seed":
            self.thresholds = [0, 1]
        elif self.dataset_name == "dreamer":
            self.thresholds = [3, 4]

        self.with_subject_id = with_subject_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]

        # convert X into 3D for topomap
        if len(X.shape) == 4:
            n_bands, w, h, color_channels = X.shape
            X = X.transpose(0, 3).reshape(-1, w, h)

        # convert y into discrete
        subject_ids, valences = y
        processed_y = self.preprocess_y(valences)

        if self.with_subject_id:
            return X, processed_y, subject_ids
        else:
            return X, processed_y

    def preprocess_y(self, y):
        transform = Discretizer(thresholds=self.thresholds)
        return transform(y=y)["y"]


def prepare_dataset(
    feature_type: str = "raw_normalized",  # raw_normalized, de_grid, de_interpolated_grid, psd_grid, de_psd_grid
    class_type: str = "binary",  # binary, ternary
    overlap_percent: float = 0,  # 0, 25, 50, 75
):

    assert overlap_percent in [0, 25, 50, 75], "Overlap must be one of [0, 25, 50, 75]"
    overlap_name = f"{overlap_percent}_percent_overlap"
    overlap = overlap_percent / 100

    if feature_type == "raw_normalized":

        deap = DEAPDataset(
            io_path=f"dataset/processed_data/deap_raw_normalized_{overlap_name}",
            root_path="../../../../EEG/Dataset/DEAP/data_preprocessed_python",
            overlap=int(overlap * 128),  # 0, 32, 64, 96
            offline_transform=transforms.Compose(
                [
                    transforms.BaselineRemoval(),
                    transforms.MeanStdNormalize(),
                ]
            ),
            online_transform=transforms.ToTensor(),
            label_transform=transforms.Compose(
                [transforms.Select(["subject_id", "valence"])]
            ),
            num_worker=NUM_WORKER,
        )

        if class_type == "binary":
            seed = SEEDBinaryDataset(
                io_path=f"dataset/processed_data/seed_binary_raw_normalized_{overlap_name}",
                root_path="../../../../EEG/Dataset/SEED/SEED/SEED_EEG/Preprocessed_EEG/",
                overlap=int(overlap * 200),  # 0, 50, 100, 150
                offline_transform=transforms.Compose(
                    [
                        transforms.MeanStdNormalize(),
                    ]
                ),
                online_transform=transforms.ToTensor(),
                label_transform=transforms.Compose(
                    [transforms.Select(["subject_id", "emotion"])]
                ),
                num_worker=NUM_WORKER,
            )

        elif class_type == "ternary":
            seed = SEEDDataset(
                io_path=f"dataset/processed_data/seed_raw_normalized_{overlap_name}",
                root_path="../../../../EEG/Dataset/SEED/SEED/SEED_EEG/Preprocessed_EEG/",
                overlap=int(overlap * 200),  # 0, 50, 100, 150
                offline_transform=transforms.Compose(
                    [
                        transforms.MeanStdNormalize(),
                    ]
                ),
                online_transform=transforms.ToTensor(),
                label_transform=transforms.Compose(
                    [transforms.Select(["subject_id", "emotion"])]
                ),
                num_worker=NUM_WORKER,
            )

        dreamer = DREAMERDataset(
            io_path=f"dataset/processed_data/dreamer_raw_normalized_{overlap_name}",
            mat_path="../../../../EEG/Dataset/DREAMER/DREAMER.mat",
            overlap=int(overlap * 128),  # 0, 32, 64, 96
            offline_transform=transforms.Compose(
                [
                    transforms.MeanStdNormalize(),
                ]
            ),
            online_transform=transforms.Compose([
                transforms.ToTensor(), 
                transforms.BaselineRemoval()
                ]),
            label_transform=transforms.Compose(
                [transforms.Select(["subject_id", "valence"])]
            ),
            num_worker=NUM_WORKER,
        )

    elif feature_type == "de_grid":
        deap = DEAPDataset(
            io_path=f"dataset/processed_data/deap_de_grid_{overlap_name}",
            root_path="../../../../EEG/Dataset/DEAP/data_preprocessed_python",
            overlap=int(overlap * 128),  # 0, 32, 64, 96
            offline_transform=transforms.Compose(
                [
                    transforms.BandDifferentialEntropy(apply_to_baseline=True),
                    transforms.ToGrid(
                        DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True
                    ),
                ]
            ),
            online_transform=transforms.Compose(
                [
                    transforms.BaselineRemoval(),
                    transforms.ToTensor(),
                ]
            ),
            label_transform=transforms.Compose(
                [transforms.Select(["subject_id", "valence"])]
            ),
            num_worker=NUM_WORKER,
        )

        if class_type == "binary":
            seed = SEEDBinaryDataset(
                io_path=f"dataset/processed_data/seed_binary_de_grid_{overlap_name}",
                root_path="../../../../EEG/Dataset/SEED/SEED/SEED_EEG/Preprocessed_EEG/",
                overlap=int(overlap * 200),  # 0, 50, 100, 150
                offline_transform=transforms.Compose(
                    [
                        transforms.BandDifferentialEntropy(),
                        transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT),
                    ]
                ),
                online_transform=transforms.ToTensor(),
                label_transform=transforms.Compose(
                    [transforms.Select(["subject_id", "emotion"])]
                ),
                num_worker=NUM_WORKER,
            )

        elif class_type == "ternary":
            seed = SEEDDataset(
                io_path=f"dataset/processed_data/seed_de_grid_{overlap_name}",
                root_path="../../../../EEG/Dataset/SEED/SEED/SEED_EEG/Preprocessed_EEG/",
                overlap=int(overlap * 200),  # 0, 50, 100, 150
                offline_transform=transforms.Compose(
                    [
                        transforms.BandDifferentialEntropy(),
                        transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT),
                    ]
                ),
                online_transform=transforms.ToTensor(),
                label_transform=transforms.Compose(
                    [transforms.Select(["subject_id", "emotion"])]
                ),
                num_worker=NUM_WORKER,
            )

        dreamer = DREAMERDataset(
            io_path=f"dataset/processed_data/dreamer_de_grid_{overlap_name}",
            mat_path="../../../../EEG/Dataset/DREAMER/DREAMER.mat",
            overlap=int(overlap * 128),  # 0, 32, 64, 96
            offline_transform=transforms.Compose(
                [
                    transforms.BandDifferentialEntropy(apply_to_baseline=True),
                    transforms.ToGrid(DREAMER_CHANNEL_LOCATION_DICT, apply_to_baseline=True),
                ]
            ),
            online_transform=transforms.Compose([
                transforms.ToTensor(), 
                transforms.BaselineRemoval()
                ]),
            label_transform=transforms.Compose(
                [transforms.Select(["subject_id", "valence"])]
            ),
            num_worker=NUM_WORKER,
        )

    elif feature_type == "de_interpolated_grid":
        deap = DEAPDataset(
            io_path="dataset/processed_data/deap_de_interpolated_grid",
            root_path="../../../../EEG/Dataset/DEAP/data_preprocessed_python",
            offline_transform=transforms.Compose(
                [
                    transforms.BandDifferentialEntropy(apply_to_baseline=True),
                    transforms.ToInterpolatedGrid(
                        DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True
                    ),
                ]
            ),
            online_transform=transforms.Compose(
                [
                    transforms.BaselineRemoval(),
                    transforms.ToTensor(),
                ]
            ),
            label_transform=transforms.Compose(
                [transforms.Select(["subject_id", "valence"])]
            ),
            num_worker=NUM_WORKER,
        )

        if class_type == "binary":
            seed = SEEDBinaryDataset(
                io_path="dataset/processed_data/seed_binary_de_interpolated_grid",
                root_path="../../../../EEG/Dataset/SEED/SEED/SEED_EEG/Preprocessed_EEG/",
                offline_transform=transforms.Compose(
                    [
                        transforms.BandDifferentialEntropy(),
                        transforms.ToInterpolatedGrid(SEED_CHANNEL_LOCATION_DICT),
                    ]
                ),
                online_transform=transforms.ToTensor(),
                label_transform=transforms.Compose(
                    [transforms.Select(["subject_id", "emotion"])]
                ),
                num_worker=NUM_WORKER,
            )

        elif class_type == "ternary":
            seed = SEEDDataset(
                io_path="dataset/processed_data/seed_de_interpolated_grid",
                root_path="../../../../EEG/Dataset/SEED/SEED/SEED_EEG/Preprocessed_EEG/",
                offline_transform=transforms.Compose(
                    [
                        transforms.BandDifferentialEntropy(),
                        transforms.ToInterpolatedGrid(SEED_CHANNEL_LOCATION_DICT),
                    ]
                ),
                online_transform=transforms.ToTensor(),
                label_transform=transforms.Compose(
                    [transforms.Select(["subject_id", "emotion"])]
                ),
                num_worker=NUM_WORKER,
            )

        dreamer = DREAMERDataset(
            io_path="dataset/processed_data/dreamer_de_interpolated_grid",
            mat_path="../../../../EEG/Dataset/DREAMER/DREAMER.mat",
            offline_transform=transforms.Compose(
                [
                    transforms.BandDifferentialEntropy(),
                    transforms.ToInterpolatedGrid(DREAMER_CHANNEL_LOCATION_DICT),
                ]
            ),
            online_transform=transforms.Compose([
                transforms.ToTensor(), 
                transforms.BaselineRemoval()
                ]),
            label_transform=transforms.Compose(
                [transforms.Select(["subject_id", "valence"])]
            ),
            num_worker=NUM_WORKER,
        )

    elif feature_type == "psd_grid":
        deap = DEAPDataset(
            io_path="dataset/processed_data/deap_psd_grid",
            root_path="../../../../EEG/Dataset/DEAP/data_preprocessed_python",
            offline_transform=transforms.Compose(
                [
                    transforms.BandPowerSpectralDensity(apply_to_baseline=True),
                    transforms.ToGrid(
                        DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True
                    ),
                ]
            ),
            online_transform=transforms.Compose(
                [
                    transforms.BaselineRemoval(),
                    transforms.ToTensor(),
                ]
            ),
            label_transform=transforms.Compose(
                [transforms.Select(["subject_id", "valence"])]
            ),
            num_worker=NUM_WORKER,
        )

        if class_type == "binary":
            seed = SEEDBinaryDataset(
                io_path="dataset/processed_data/seed_binary_psd_grid",
                root_path="../../../../EEG/Dataset/SEED/SEED/SEED_EEG/Preprocessed_EEG/",
                offline_transform=transforms.Compose(
                    [
                        transforms.BandPowerSpectralDensity(),
                        transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT),
                    ]
                ),
                online_transform=transforms.ToTensor(),
                label_transform=transforms.Compose(
                    [transforms.Select(["subject_id", "emotion"])]
                ),
                num_worker=NUM_WORKER,
            )

        elif class_type == "ternary":
            seed = SEEDDataset(
                io_path="dataset/processed_data/seed_psd_grid",
                root_path="../../../../EEG/Dataset/SEED/SEED/SEED_EEG/Preprocessed_EEG/",
                offline_transform=transforms.Compose(
                    [
                        transforms.BandPowerSpectralDensity(),
                        transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT),
                    ]
                ),
                online_transform=transforms.ToTensor(),
                label_transform=transforms.Compose(
                    [transforms.Select(["subject_id", "emotion"])]
                ),
                num_worker=NUM_WORKER,
            )

        dreamer = DREAMERDataset(
            io_path="dataset/processed_data/dreamer_psd_grid",
            mat_path="../../../../EEG/Dataset/DREAMER/DREAMER.mat",
            offline_transform=transforms.Compose(
                [
                    transforms.BandPowerSpectralDensity(),
                    transforms.ToGrid(DREAMER_CHANNEL_LOCATION_DICT),
                ]
            ),
            online_transform=transforms.Compose([
                transforms.ToTensor(), 
                transforms.BaselineRemoval()
                ]),
            label_transform=transforms.Compose(
                [transforms.Select(["subject_id", "valence"])]
            ),
            num_worker=NUM_WORKER,
        )

    elif feature_type == "de_psd_grid":
        deap = DEAPDataset(
            io_path="dataset/processed_data/deap_de_psd_grid",
            root_path="../../../../EEG/Dataset/DEAP/data_preprocessed_python",
            offline_transform=transforms.Compose(
                [
                    transforms.Concatenate(
                        [
                            transforms.BandDifferentialEntropy(),
                            transforms.BandPowerSpectralDensity(),
                        ]
                    ),
                    transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT),
                ]
            ),
            online_transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
            label_transform=transforms.Compose(
                [transforms.Select(["subject_id", "valence"])]
            ),
            num_worker=NUM_WORKER,
        )

        if class_type == "binary":
            seed = SEEDBinaryDataset(
                io_path="dataset/processed_data/seed_binary_de_psd_grid",
                root_path="../../../../EEG/Dataset/SEED/SEED/SEED_EEG/Preprocessed_EEG/",
                offline_transform=transforms.Compose(
                    [
                        transforms.Concatenate(
                            [
                                transforms.BandDifferentialEntropy(),
                                transforms.BandPowerSpectralDensity(),
                            ]
                        ),
                        transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT),
                    ]
                ),
                online_transform=transforms.ToTensor(),
                label_transform=transforms.Compose(
                    [transforms.Select(["subject_id", "emotion"])]
                ),
                num_worker=NUM_WORKER,
            )

        elif class_type == "ternary":
            seed = SEEDDataset(
                io_path="dataset/processed_data/seed_de_psd_grid",
                root_path="../../../../EEG/Dataset/SEED/SEED/SEED_EEG/Preprocessed_EEG/",
                offline_transform=transforms.Compose(
                    [
                        transforms.Concatenate(
                            [
                                transforms.BandDifferentialEntropy(),
                                transforms.BandPowerSpectralDensity(),
                            ]
                        ),
                        transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT),
                    ]
                ),
                online_transform=transforms.ToTensor(),
                label_transform=transforms.Compose(
                    [transforms.Select(["subject_id", "emotion"])]
                ),
                num_worker=NUM_WORKER,
            )

        dreamer = DREAMERDataset(
            io_path="dataset/processed_data/dreamer_de_psd_grid",
            mat_path="../../../../EEG/Dataset/DREAMER/DREAMER.mat",
            offline_transform=transforms.Compose(
                [
                    transforms.Concatenate(
                        [
                            transforms.BandDifferentialEntropy(),
                            transforms.BandPowerSpectralDensity(),
                        ]
                    ),
                    transforms.ToGrid(DREAMER_CHANNEL_LOCATION_DICT),
                ]
            ),
            online_transform=transforms.Compose([
                transforms.ToTensor(), 
                transforms.BaselineRemoval()
                ]),
            label_transform=transforms.Compose(
                [transforms.Select(["subject_id", "valence"])]
            ),
            num_worker=NUM_WORKER,
        )

    datasets = [deap, seed, dreamer]

    return datasets


def prepare_subject_dataset(feature_type, dataset_names=["deap", "seed", "dreamer"]):
    deap, seed, dreamer = prepare_dataset(feature_type)
    datasets = {"deap": deap, "seed": seed, "dreamer": dreamer}

    subject_dataset = []
    for name in dataset_names:
        if name in datasets:
            dataset = datasets[name]
            loso = LeaveOneSubjectOut(split_path=f"loso_split/{name}")

            for _, subject in loso.split(dataset):
                subject_dataset.append(subject)

    return subject_dataset


def prepare_dataloaders(
    datasets,
    batch_size: int = 32,
    test_ratio: float = 0.2,
    seed: int = 333,
    with_subject_id: bool = False,
):
    # reproducibility
    set_seed(seed)

    trainloaders = []
    valloaders = []
    testloaders = []

    # determine class type
    class_type = "ternary"
    for dataset_ in datasets:
        dataset_name = dataset_.__class__.__name__.lower()
        if "binary" in dataset_name:
            class_type = "binary"
            break

    # split datasets
    for dataset_ in datasets:
        # calculate split sizes
        num_total = len(dataset_)
        num_train = int((1 - test_ratio) ** 2 * num_total)
        num_val = int((1 - test_ratio) * test_ratio * num_total)
        num_test = num_total - num_train - num_val

        # generate random shuffled indices
        indices = torch.randperm(num_total).tolist()

        # split into train-val-test
        train_indices = indices[:num_train]
        val_indices = indices[num_train : (num_train + num_val)]
        test_indices = indices[(num_train + num_val) :]

        # create subsets
        subset_train = Subset(dataset_, train_indices)
        subset_val = Subset(dataset_, val_indices)
        subset_test = Subset(dataset_, test_indices)

        # dataloaders
        CustomDataset = BinaryDataset if class_type == "binary" else TernaryDataset

        trainloader = DataLoader(
            CustomDataset(subset_train, with_subject_id),
            batch_size=batch_size,
            num_workers=NUM_WORKER,
        )
        valloader = DataLoader(
            CustomDataset(subset_val, with_subject_id),
            batch_size=batch_size,
            num_workers=NUM_WORKER,
        )
        testloader = DataLoader(
            CustomDataset(subset_test, with_subject_id),
            batch_size=batch_size,
            num_workers=NUM_WORKER,
        )

        trainloaders.append(trainloader)
        valloaders.append(valloader)
        testloaders.append(testloader)

    return trainloaders, valloaders, testloaders


if __name__ == "__main__":
    # raw normalized datasets
    raw_datasets = prepare_dataset(
        feature_type="raw_normalized",
        class_type="binary",
        overlap_percent=0,
    )

    # grid (differential entropy) datasets
    grid_datasets = prepare_dataset(
        feature_type="de_grid",
        class_type="binary",
        overlap_percent=0,
    )