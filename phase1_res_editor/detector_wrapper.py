# detector_wrapper.py
from typing import Any, Dict, Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenDetectorScorer(nn.Module):
    """
    Wrap a frozen detector into a differentiable scorer.

    Assumptions:
    - input x is float tensor in [0, 1], shape [B, 3, H, W]
    - output score is shape [B], larger means "more AI / more fake"
    """

    def __init__(
        self,
        raw_detector: nn.Module,
        input_size: int = 224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        fake_index: int = 1,
        dict_score_keys: Iterable[str] = ("logits", "pred", "score", "scores", "output"),
    ):
        super().__init__()
        self.detector = raw_detector.eval()
        for p in self.detector.parameters():
            p.requires_grad = False

        self.input_size = input_size
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.fake_index = fake_index
        self.dict_score_keys = tuple(dict_score_keys)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self.mean) / self.std
        return x

    def _extract_score(self, out: Any) -> torch.Tensor:
        # Case 1: dict
        if isinstance(out, dict):
            for k in self.dict_score_keys:
                if k in out:
                    return self._extract_score(out[k])
            # fallback: take first tensor-like value
            for v in out.values():
                if torch.is_tensor(v):
                    return self._extract_score(v)
            raise ValueError("Detector output dict has no tensor values.")

        # Case 2: tuple/list
        if isinstance(out, (tuple, list)):
            for item in out:
                if torch.is_tensor(item):
                    return self._extract_score(item)
            raise ValueError("Detector output tuple/list has no tensor values.")

        # Case 3: tensor
        if torch.is_tensor(out):
            # [B]
            if out.ndim == 1:
                return out

            # [B, 1]
            if out.ndim == 2 and out.shape[1] == 1:
                return out[:, 0]

            # [B, 2] => assume binary logits/probs, pick fake_index
            if out.ndim == 2 and out.shape[1] >= 2:
                return out[:, self.fake_index]

            # Unexpected tensor shape
            raise ValueError(f"Unsupported detector tensor output shape: {tuple(out.shape)}")

        raise TypeError(f"Unsupported detector output type: {type(out)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.preprocess(x)
        out = self.detector(x_in)
        score = self._extract_score(out)
        return score