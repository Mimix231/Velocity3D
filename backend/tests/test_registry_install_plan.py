from backend.providers.catalog import MODEL_CATALOG_BY_ID
from backend.providers.installers import INSTALLER_MAP
from backend.providers.registry import _build_install_plan, _entry_python_status


def test_hunyuan21_catalog_marks_python_313_as_supported_cu130(monkeypatch):
    entry = MODEL_CATALOG_BY_ID["hunyuan3d-2.1"]

    monkeypatch.setattr("backend.providers.registry.current_python_series", lambda: "3.13")
    monkeypatch.setattr("backend.providers.registry.current_python_version", lambda: "3.13.13")

    current_python, compatible, detail = _entry_python_status(entry)

    assert current_python == "3.13.13"
    assert compatible is True
    assert "torch 2.10.0+cu130" in detail


def test_hunyuan21_install_plan_uses_existing_torch_on_python_313(monkeypatch, tmp_path):
    entry = MODEL_CATALOG_BY_ID["hunyuan3d-2.1"]

    monkeypatch.setattr("backend.providers.registry.repo_path", lambda _: tmp_path / "hunyuan3d-2.1")
    monkeypatch.setattr("backend.providers.registry.current_python_series", lambda: "3.13")
    monkeypatch.setattr("backend.providers.registry.current_python_version", lambda: "3.13.13")
    monkeypatch.setattr("backend.providers.registry._has_module", lambda _: True)
    monkeypatch.setattr(
        "backend.providers.registry.installed_package_version",
        lambda name: {
            "torch": "2.10.0+cu130",
            "torchvision": "0.25.0+cu130",
            "torchaudio": "2.10.0+cu130",
        }.get(name),
    )

    plan = _build_install_plan(entry)

    assert any(step.note for step in plan)
    assert any("keeping current backend torch runtime" in step.label for step in plan if step.note)
    assert not any("torch==2.5.1" in (step.command or "") for step in plan)
    assert not any("numpy>=2.1,<2.2" in (step.command or "") for step in plan)
    assert not any("numpy==" in (step.command or "") for step in plan)
    assert not any("pyyaml==6.0.2" in (step.command or "").lower() for step in plan)
    assert not any("trimesh==4.4.7" in (step.command or "") for step in plan)
    assert any("numpy>=2.1" in (step.command or "") for step in plan)


def test_hunyuan21_install_plan_uses_upstream_stack_on_python_311(monkeypatch, tmp_path):
    entry = MODEL_CATALOG_BY_ID["hunyuan3d-2.1"]

    monkeypatch.setattr("backend.providers.registry.repo_path", lambda _: tmp_path / "hunyuan3d-2.1")
    monkeypatch.setattr("backend.providers.registry.current_python_series", lambda: "3.11")
    monkeypatch.setattr("backend.providers.registry.current_python_version", lambda: "3.11.11")
    monkeypatch.setattr("backend.providers.registry._has_module", lambda _: False)
    monkeypatch.setattr("backend.providers.registry.installed_package_version", lambda _: None)

    plan = _build_install_plan(entry)

    assert any("torch==2.10.0" in (step.command or "") for step in plan)
    assert any("download.pytorch.org/whl/cu130" in (step.command or "") for step in plan)


def test_every_catalog_model_has_a_specific_installer():
    assert set(INSTALLER_MAP) == set(MODEL_CATALOG_BY_ID)


def test_python_313_external_installers_use_curated_dependency_sets(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.providers.registry.repo_path", lambda model: tmp_path / model)
    monkeypatch.setattr("backend.providers.registry.current_python_series", lambda: "3.13")
    monkeypatch.setattr("backend.providers.registry.current_python_version", lambda: "3.13.13")
    monkeypatch.setattr("backend.providers.registry._has_module", lambda _: True)
    monkeypatch.setattr(
        "backend.providers.registry.installed_package_version",
        lambda name: {
            "torch": "2.10.0+cu130",
            "torchvision": "0.25.0+cu130",
            "torchaudio": "2.10.0+cu130",
        }.get(name),
    )

    for model_id in ("stable-fast-3d", "triposr", "zero123++"):
        plan = _build_install_plan(MODEL_CATALOG_BY_ID[model_id])

        assert any("curated" in step.label.lower() for step in plan)
        assert not any("requirements.txt" in (step.command or "") for step in plan)
