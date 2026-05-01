import pytest
from pydantic import ValidationError
from backend.models import GenerationRequest


def test_text_request_valid():
    req = GenerationRequest(type="text", prompt="a red cube", request_id="abc-123")
    assert req.type == "text"
    assert req.prompt == "a red cube"


def test_text_request_empty_prompt_raises():
    with pytest.raises(ValidationError):
        GenerationRequest(type="text", prompt="   ", request_id="abc-123")


def test_text_request_missing_prompt_raises():
    with pytest.raises(ValidationError):
        GenerationRequest(type="text", request_id="abc-123")


def test_image_request_valid():
    req = GenerationRequest(type="image", image_base64="abc123==", request_id="xyz-456")
    assert req.type == "image"
    assert req.image_base64 == "abc123=="


def test_image_request_missing_image_raises():
    with pytest.raises(ValidationError):
        GenerationRequest(type="image", request_id="xyz-456")


def test_image_request_with_optional_prompt():
    req = GenerationRequest(
        type="image", image_base64="abc123==", prompt="a car", model_id="hunyuan3d-2.1", request_id="xyz-789"
    )
    assert req.prompt == "a car"
    assert req.model_id == "hunyuan3d-2.1"
