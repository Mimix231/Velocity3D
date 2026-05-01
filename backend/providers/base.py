from __future__ import annotations

import threading
from abc import ABC
from pathlib import Path
from typing import Optional


class ProviderError(Exception):
    pass


class ProviderDependencyError(ProviderError):
    pass


class ProviderConfigurationError(ProviderError):
    pass


class ProviderCapabilityError(ProviderError):
    pass


class ProviderExecutionError(ProviderError):
    pass


class GenerationProvider(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    def generate_text(
        self,
        prompt: str,
        output_dir: Path,
        cancellation_event: threading.Event,
    ) -> str:
        raise ProviderCapabilityError(f"{self.model_id} does not support text-to-3D generation")

    def generate_image(
        self,
        image_bytes: bytes,
        output_dir: Path,
        cancellation_event: threading.Event,
        prompt: Optional[str] = None,
    ) -> str:
        raise ProviderCapabilityError(f"{self.model_id} does not support image-to-3D generation")
