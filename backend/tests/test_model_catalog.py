import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


@pytest.mark.asyncio
async def test_model_catalog_lists_core_entries():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    ids = {item["id"] for item in payload["models"]}

    assert "shap-e" in ids
    assert "hunyuan3d-2.1" in ids
    assert "trellis" in ids
    assert "trellis.2" in ids
    assert "stable-fast-3d" in ids
    assert "triposr" in ids
    assert "zero123++" in ids


@pytest.mark.asyncio
async def test_model_catalog_marks_zero123_as_library_only():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    zero123 = next(item for item in payload["models"] if item["id"] == "zero123++")
    assert zero123["role"] == "assistant"
    assert zero123["status"] == "library_only"


@pytest.mark.asyncio
async def test_model_catalog_exposes_trellis_for_text_and_image():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    trellis = next(item for item in payload["models"] if item["id"] == "trellis")
    assert sorted(trellis["selection_modes"]) == ["image", "text"]
