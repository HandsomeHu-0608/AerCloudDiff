"""Optional training-window filter based on a missing-observation manifest."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.data.final_target_policy import missing_leads, validate_source_record


class KnownMissingTargetDataset(Dataset):
    def __init__(self, dataset, policy):
        self.dataset = dataset
        self.policy = policy
        if list(dataset.cascade_start_hours) != [0, 4, 8, 12, 16] or dataset.cascade_horizon != 8:
            raise ValueError("Missing-target training requires the five fixed eight-hour windows")
        self.indices = []
        self.excluded_windows = []
        patches = dataset.patches_per_file
        for file_index, path in enumerate(dataset.files):
            validate_source_record(policy, path)
            absent = set(missing_leads(policy, path))
            for block, start in enumerate(dataset.cascade_start_hours):
                original = (file_index * 5 + block) * patches
                if all(hour in absent for hour in range(start + 1, start + 9)):
                    self.excluded_windows.append((path.name, start + 1, start + 8))
                    continue
                self.indices.extend(range(original, original + patches))
        if not self.indices:
            raise ValueError("Missing-target filter left no supervised training windows")

    def __getattr__(self, name):
        # files, current_epoch, set_epoch and read_block preserve base behavior;
        # importantly read_block still emits all eight hours for inference.
        if name in {"dataset", "policy", "indices", "excluded_windows"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = self.dataset[self.indices[index]]
        validate_source_record(self.policy, item["path"])
        absent = set(missing_leads(self.policy, item["path"]))
        hours = item["future_hours"]
        known = torch.tensor([int(hour) in absent for hour in hours], dtype=torch.bool)
        mask = item["loss_mask"] > 0
        counts = mask.sum(dim=(0, 2, 3))
        if counts.shape != known.shape:
            raise ValueError("Missing-target flags and loss-mask time axes disagree")
        if torch.any((counts == 0) & ~known):
            raise ValueError("Unexpected empty target frame/crop outside the approved missing list")
        if torch.any((counts != 0) & known):
            raise ValueError("An approved missing target now contains scoring pixels")
        truth = item["target_k"]
        if known.any() and (torch.isfinite(truth[:, known]).any()
                            or (item["target_valid_mask"][:, known] > 0).any()):
            raise ValueError("An approved missing frame no longer has all-missing truth/mask")
        if bool(known.all()):
            raise ValueError("An all-empty training window escaped the fixed index filter")
        item["known_missing_target"] = known
        return item
