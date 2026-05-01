from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from backend.providers.catalog import CatalogEntry
from backend.providers.helpers import python_executable


@dataclass(frozen=True)
class InstallerContext:
    current_python_series: str
    current_python_version: str
    installed_packages: dict[str, str | None]
    torch_stack_ready: bool
    repo_exists: bool


@dataclass(frozen=True)
class InstallPlanStep:
    label: str
    command: str | None = None
    manual: bool = False
    note: bool = False
    action: str | None = None
    optional: bool = False
    hf_repo_id: str | None = None
    hf_allow_patterns: tuple[str, ...] = ()
    hf_local_dir: str | None = None
    download_url: str | None = None
    download_path: str | None = None


@dataclass(frozen=True)
class PackageGroup:
    name: str
    packages: tuple[str, ...]
    index_url: str | None = None
    extra_args: tuple[str, ...] = ()
    optional: bool = False


def is_manual_install_step(step: str) -> bool:
    lowered = step.strip().lower()
    return lowered.startswith(("follow ", "read ", "see ", "open "))


def is_note_step(step: str) -> bool:
    return step.strip().lower().startswith("velocity3d ")


def normalize_install_command(command: str) -> str:
    stripped = command.strip()
    lowered = stripped.lower()
    for prefix in ("python ", "python3 ", "py "):
        if lowered.startswith(prefix):
            return f'"{python_executable()}" {stripped[len(prefix):]}'
    return stripped


def clone_step(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    if entry.repo_url and entry.vendor_dir_name and not context.repo_exists:
        return [InstallPlanStep(label=f"Clone {entry.name} repository")]
    return []


def note(label: str) -> InstallPlanStep:
    return InstallPlanStep(label=label, note=True)


def manual(label: str) -> InstallPlanStep:
    return InstallPlanStep(label=label, manual=True)


def action(label: str, action_name: str) -> InstallPlanStep:
    return InstallPlanStep(label=label, action=action_name)


def huggingface_download_step(
    label: str,
    repo_id: str,
    *,
    allow_patterns: tuple[str, ...] = (),
    local_dir: str | None = None,
    optional: bool = False,
) -> InstallPlanStep:
    return InstallPlanStep(
        label=label,
        action="download_huggingface_snapshot",
        optional=optional,
        hf_repo_id=repo_id,
        hf_allow_patterns=allow_patterns,
        hf_local_dir=local_dir,
    )


def download_file_step(label: str, url: str, destination: str, *, optional: bool = False) -> InstallPlanStep:
    return InstallPlanStep(
        label=label,
        action="download_file",
        optional=optional,
        download_url=url,
        download_path=destination,
    )


def pip_install_command(packages: Iterable[str], *, index_url: str | None = None, extra_args: Iterable[str] = ()) -> str:
    parts = ["python", "-m", "pip", "install"]
    parts.extend(extra_args)
    if index_url:
        parts.extend(["--index-url", index_url])
    parts.extend(packages)
    return normalize_install_command(" ".join(parts))


def pip_install_group(group: PackageGroup) -> InstallPlanStep:
    return InstallPlanStep(
        label=f"Install {group.name}",
        command=pip_install_command(group.packages, index_url=group.index_url, extra_args=group.extra_args),
        optional=group.optional,
    )


def pip_install_groups(groups: Iterable[PackageGroup]) -> list[InstallPlanStep]:
    return [pip_install_group(group) for group in groups]


def pip_requirements(label: str = "Install upstream requirements.txt") -> InstallPlanStep:
    return InstallPlanStep(
        label=label,
        command=normalize_install_command("python -m pip install -r requirements.txt"),
    )


def keep_existing_torch_note(model_name: str, context: InstallerContext, reason: str) -> InstallPlanStep:
    return note(
        f"Keeping current backend torch runtime for {model_name} on Python {context.current_python_version}: "
        f"torch {context.installed_packages.get('torch')}, "
        f"torchvision {context.installed_packages.get('torchvision')}, "
        f"torchaudio {context.installed_packages.get('torchaudio')}. {reason}"
    )


def torch_or_reuse_step(
    model_name: str,
    context: InstallerContext,
    *,
    packages: tuple[str, ...] = ("torch", "torchvision", "torchaudio"),
    index_url: str = "https://download.pytorch.org/whl/cu130",
    reuse_reason: str,
) -> InstallPlanStep:
    if context.torch_stack_ready:
        return keep_existing_torch_note(model_name, context, reuse_reason)
    return pip_install_group(
        PackageGroup(
            name=f"{model_name} compatible PyTorch runtime",
            packages=packages,
            index_url=index_url,
        )
    )


def torch_step_for_runtime(
    label: str,
    context: InstallerContext,
    *,
    supported_series: set[str],
    upstream_packages: tuple[str, ...],
    upstream_index_url: str,
    fallback_packages: tuple[str, ...] = ("torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0"),
    fallback_index_url: str = "https://download.pytorch.org/whl/cu130",
) -> InstallPlanStep:
    if context.current_python_series in supported_series:
        if context.torch_stack_ready:
            return note(
                f"{label}: keeping current backend torch runtime on Python {context.current_python_version}: "
                f"torch {context.installed_packages.get('torch')}, "
                f"torchvision {context.installed_packages.get('torchvision')}, "
                f"torchaudio {context.installed_packages.get('torchaudio')}."
            )
        return InstallPlanStep(
            label=label,
            command=pip_install_command(upstream_packages, index_url=upstream_index_url),
        )
    if context.current_python_series == "3.13" and context.torch_stack_ready:
        return note(
            f"{label}: keeping current backend torch runtime on Python {context.current_python_version}: "
            f"torch {context.installed_packages.get('torch')}, "
            f"torchvision {context.installed_packages.get('torchvision')}, "
            f"torchaudio {context.installed_packages.get('torchaudio')}. "
            "Older upstream PyTorch wheels are unavailable for Windows Python 3.13."
        )
    return InstallPlanStep(
        label=f"Install Python {context.current_python_series}-compatible PyTorch CUDA 13.0 runtime",
        command=pip_install_command(fallback_packages, index_url=fallback_index_url),
    )


def requirements_or_curated(
    label: str,
    context: InstallerContext,
    groups: Iterable[PackageGroup],
    *,
    python313_note: str,
) -> list[InstallPlanStep]:
    if context.current_python_series == "3.13":
        plan = [
            note(python313_note),
        ]
        plan.extend(pip_install_groups(groups))
        return plan
    return [pip_requirements(label)]


def generic_install_plan(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    plan: list[InstallPlanStep] = []

    if entry.repo_url and entry.vendor_dir_name and not context.repo_exists:
        plan.append(InstallPlanStep(label=f"Clone {entry.name} repository"))

    for raw_step in entry.install_steps:
        step = raw_step.strip()
        lowered = step.lower()
        if lowered.startswith("git clone "):
            continue
        if lowered.startswith("cd "):
            continue
        if is_note_step(step):
            plan.append(InstallPlanStep(label=step, note=True))
            continue
        if is_manual_install_step(step):
            plan.append(InstallPlanStep(label=step, manual=True))
            continue
        plan.append(InstallPlanStep(label=step, command=normalize_install_command(step)))

    return plan
