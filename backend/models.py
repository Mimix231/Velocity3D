from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class TextureOptions(BaseModel):
    enabled: bool = False
    checkpoint: Optional[str] = None


class GenerationRequest(BaseModel):
    type: Literal["text", "image"]
    prompt: Optional[str] = None
    image_base64: Optional[str] = None
    reference_image_base64: Optional[str] = None
    model_id: Optional[str] = None
    texture_options: Optional[TextureOptions] = None
    request_id: str

    @model_validator(mode="after")
    def validate_fields(self) -> "GenerationRequest":
        if self.type == "text":
            if not self.prompt or not self.prompt.strip():
                raise ValueError("prompt must be non-empty for text generation requests")
        elif self.type == "image":
            if not self.image_base64:
                raise ValueError("image_base64 is required for image generation requests")
        return self


class GenerationMetadata(BaseModel):
    vertex_count: int
    face_count: int
    generation_time_ms: int
    pipeline: str
    model_id: str
    model_name: str
    texture_applied: bool = False
    texture_checkpoint: Optional[str] = None
    material_texture_dir: Optional[str] = None
    material_textures: list[str] = Field(default_factory=list)


class GenerationResponse(BaseModel):
    request_id: str
    model_path: str
    metadata: GenerationMetadata


class ErrorResponse(BaseModel):
    error: str
    details: str


class ExportRequest(BaseModel):
    model_path: str
    output_path: str
    format: Literal["glb", "obj", "fbx"]


class ExportResponse(BaseModel):
    output_path: str
    format: str


class CancelRequest(BaseModel):
    request_id: str


class ModelCatalogItem(BaseModel):
    id: str
    name: str
    family: str
    role: Literal["generator", "assistant"]
    summary: str
    description: str
    selection_modes: list[Literal["text", "image"]]
    library_modes: list[Literal["text", "image", "multiview"]]
    recommended: bool
    repo_url: Optional[str] = None
    docs_url: Optional[str] = None
    huggingface_url: Optional[str] = None
    license_name: Optional[str] = None
    vram_hint: Optional[str] = None
    size_hint: Optional[str] = None
    platform_note: Optional[str] = None
    preferred_python: Optional[str] = None
    supported_python: list[str]
    current_python: Optional[str] = None
    python_compatible: Optional[bool] = None
    python_status_detail: Optional[str] = None
    install_steps: list[str]
    downloaded: bool
    generation_ready: bool
    status: Literal["ready", "downloaded", "setup_required", "library_only"]
    status_detail: str


class ModelCatalogResponse(BaseModel):
    models: list[ModelCatalogItem]


class ModelDownloadRequest(BaseModel):
    model_id: str


class ModelDownloadResponse(BaseModel):
    model_id: str
    destination: str


class ModelInstallRequest(BaseModel):
    model_id: str


class ModelInstallStartResponse(BaseModel):
    job_id: str
    model_id: str


class ModelInstallStatusResponse(BaseModel):
    job_id: str
    model_id: str
    model_name: str
    status: Literal["running", "complete", "error", "manual_required"]
    current_step: int
    step_count: int
    active_step: Optional[str] = None
    logs: list[str]
    status_detail: str
    error: Optional[str] = None
