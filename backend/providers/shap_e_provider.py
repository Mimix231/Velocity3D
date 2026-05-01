from __future__ import annotations

import threading
from pathlib import Path

from backend.pipelines.text_to_3d import TextTo3DPipeline
from backend.providers.base import GenerationProvider


class ShapEProvider(GenerationProvider):
    _pipeline: TextTo3DPipeline | None = None
    _lock = threading.Lock()

    def _get_pipeline(self) -> TextTo3DPipeline:
        if self._pipeline is not None:
            return self._pipeline
        with self._lock:
            if self._pipeline is None:
                ShapEProvider._pipeline = TextTo3DPipeline()
        return ShapEProvider._pipeline

    def generate_text(self, prompt: str, output_dir: Path, cancellation_event: threading.Event) -> str:
        pipeline = self._get_pipeline()
        return pipeline.generate(prompt, output_dir, cancellation_event)
