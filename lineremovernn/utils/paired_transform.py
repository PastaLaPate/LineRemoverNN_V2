import random

import torch


class PairedTransform:
    def __init__(self, augment: bool = False) -> None:
        self.augment = augment

    def __call__(
        self, ruled: torch.Tensor, blank: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ruled = ruled.float() / 255.0
        blank = blank.float() / 255.0
        if self.augment:
            if random.random() > 0.5:
                ruled = torch.flip(ruled, dims=[2])
                blank = torch.flip(blank, dims=[2])
            if random.random() > 0.5:
                ruled = torch.flip(ruled, dims=[1])
                blank = torch.flip(blank, dims=[1])
        return ruled, blank
