# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for TrainingHost + HostRegistry + workstation auto-detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from lerobot.gui.training.hosts import (
    WORKSTATION_HOST_ID,
    HostRegistry,
    TrainingHost,
    _detect_nvidia_gpu,
    profile_to_training_host,
    workstation_host,
)
from lerobot.gui.training.jobs import HostProfile
from lerobot.gui.training.transport import SshTransport, SubprocessTransport

# ── GPU detection ─────────────────────────────────────────────────────────────


def test_detect_nvidia_gpu_no_nvidia_smi_returns_none() -> None:
    """When nvidia-smi isn't on PATH, detection returns None."""
    with patch("lerobot.gui.training.hosts.shutil.which", return_value=None):
        assert _detect_nvidia_gpu() is None


def test_detect_nvidia_gpu_parses_output() -> None:
    """A typical nvidia-smi output gives us gpu_name + vram_mb."""
    fake_output = "NVIDIA GeForce RTX 5090, 32768\n"
    with patch("lerobot.gui.training.hosts.shutil.which", return_value="/usr/bin/nvidia-smi"):
        with patch("lerobot.gui.training.hosts.subprocess.check_output", return_value=fake_output):
            caps = _detect_nvidia_gpu()
    assert caps is not None
    assert caps["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert caps["vram_mb"] == 32768
    assert caps["gpu_count_detected"] == 1


def test_detect_nvidia_gpu_multi_gpu_counts() -> None:
    fake_output = "NVIDIA A100, 40960\nNVIDIA A100, 40960\n"
    with patch("lerobot.gui.training.hosts.shutil.which", return_value="/usr/bin/nvidia-smi"):
        with patch("lerobot.gui.training.hosts.subprocess.check_output", return_value=fake_output):
            caps = _detect_nvidia_gpu()
    assert caps is not None
    assert caps["gpu_count_detected"] == 2


def test_detect_nvidia_gpu_subprocess_failure_returns_none() -> None:
    import subprocess

    with patch("lerobot.gui.training.hosts.shutil.which", return_value="/usr/bin/nvidia-smi"):
        with patch(
            "lerobot.gui.training.hosts.subprocess.check_output",
            side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
        ):
            assert _detect_nvidia_gpu() is None


def test_detect_nvidia_gpu_malformed_memory_falls_back() -> None:
    fake_output = "Some GPU, not-a-number\n"
    with patch("lerobot.gui.training.hosts.shutil.which", return_value="/usr/bin/nvidia-smi"):
        with patch("lerobot.gui.training.hosts.subprocess.check_output", return_value=fake_output):
            caps = _detect_nvidia_gpu()
    assert caps is not None
    assert caps["vram_mb"] == 0


# ── workstation_host ───────────────────────────────────────────────────────────


def test_workstation_host_none_when_no_gpu() -> None:
    with patch("lerobot.gui.training.hosts._detect_nvidia_gpu", return_value=None):
        assert workstation_host() is None


def test_workstation_host_returns_host_when_gpu_present(tmp_path: Path) -> None:
    fake_caps = {"gpu_name": "Test GPU", "vram_mb": 16384, "gpu_count_detected": 1}
    with patch("lerobot.gui.training.hosts._detect_nvidia_gpu", return_value=fake_caps):
        h = workstation_host(workdir=tmp_path)
    assert h is not None
    assert h.id == WORKSTATION_HOST_ID
    assert h.display_name == "This server"
    assert isinstance(h.transport, SubprocessTransport)
    assert h.transport.workdir == tmp_path
    assert h.capabilities == fake_caps


# ── HostRegistry ──────────────────────────────────────────────────────────────


def _ssh_host(id_: str) -> TrainingHost:
    return TrainingHost(id=id_, display_name=id_, transport=SshTransport(host="x"))


def test_registry_list_and_get() -> None:
    h1 = _ssh_host("a")
    h2 = _ssh_host("b")
    reg = HostRegistry(hosts=[h1, h2])
    listed = reg.list_hosts()
    assert {h.id for h in listed} == {"a", "b"}
    assert reg.get("a") is h1
    assert reg.get("missing") is None


def test_registry_add_collision_asserts() -> None:
    h1 = _ssh_host("a")
    reg = HostRegistry(hosts=[h1])
    with pytest.raises(AssertionError, match="collision"):
        reg.add(_ssh_host("a"))


def test_registry_add_appends() -> None:
    reg = HostRegistry(hosts=[_ssh_host("a")])
    reg.add(_ssh_host("b"))
    assert {h.id for h in reg.list_hosts()} == {"a", "b"}


def test_registry_auto_with_gpu_present(tmp_path: Path) -> None:
    fake_caps = {"gpu_name": "G", "vram_mb": 1024, "gpu_count_detected": 1}
    with patch("lerobot.gui.training.hosts._detect_nvidia_gpu", return_value=fake_caps):
        # hosts_dir=tmp_path: never read the developer's real saved hosts
        reg = HostRegistry.auto(workdir=tmp_path, hosts_dir=tmp_path)
    hosts = reg.list_hosts()
    assert len(hosts) == 1
    assert hosts[0].id == WORKSTATION_HOST_ID


def test_registry_auto_no_gpu_empty(tmp_path: Path) -> None:
    with patch("lerobot.gui.training.hosts._detect_nvidia_gpu", return_value=None):
        reg = HostRegistry.auto(workdir=tmp_path, hosts_dir=tmp_path)
    assert reg.list_hosts() == []


# ── TrainingHost frozen ────────────────────────────────────────────────────────


def test_training_host_frozen() -> None:
    import dataclasses

    h = _ssh_host("a")
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.display_name = "other"  # type: ignore[misc]


# ── Mutation: remove / replace ────────────────────────────────────────────────


def test_registry_remove_evicts() -> None:
    reg = HostRegistry(hosts=[_ssh_host("a"), _ssh_host("b")])
    assert reg.remove("a") is True
    assert reg.get("a") is None
    assert {h.id for h in reg.list_hosts()} == {"b"}


def test_registry_remove_missing_returns_false() -> None:
    reg = HostRegistry(hosts=[_ssh_host("a")])
    assert reg.remove("nope") is False
    assert reg.get("a") is not None


def test_registry_replace_swaps_entry() -> None:
    reg = HostRegistry(hosts=[_ssh_host("a")])
    new_a = TrainingHost(id="a", display_name="renamed", transport=SshTransport(host="y"))
    reg.replace("a", new_a)
    fetched = reg.get("a")
    assert fetched is new_a
    assert fetched.display_name == "renamed"


# ── profile_to_training_host adapter ──────────────────────────────────────────


def test_profile_to_training_host_builds_ssh_transport() -> None:
    p = HostProfile(name="lab-h100", ssh_user="feit", ssh_host="10.0.0.5", ssh_port=2222)
    th = profile_to_training_host(p)
    assert th.id == "lab-h100"
    assert th.display_name == "lab-h100"  # falls back to name
    assert isinstance(th.transport, SshTransport)
    assert th.transport.host == "10.0.0.5"
    assert th.transport.port == 2222
    assert th.transport.user == "feit"


def test_profile_to_training_host_respects_display_name() -> None:
    p = HostProfile(name="lab", ssh_user="u", ssh_host="h", display_name="Lab GPU")
    th = profile_to_training_host(p)
    assert th.display_name == "Lab GPU"


# ── auto() loads disk profiles ────────────────────────────────────────────────


def test_registry_auto_loads_disk_profiles(tmp_path: Path) -> None:
    HostProfile(name="lab-a", ssh_user="u", ssh_host="a.lan").save(dir_=tmp_path)
    HostProfile(name="lab-b", ssh_user="u", ssh_host="b.lan").save(dir_=tmp_path)
    with patch("lerobot.gui.training.hosts._detect_nvidia_gpu", return_value=None):
        reg = HostRegistry.auto(workdir=tmp_path, hosts_dir=tmp_path)
    assert {h.id for h in reg.list_hosts()} == {"lab-a", "lab-b"}


def test_registry_auto_mixes_workstation_and_disk(tmp_path: Path) -> None:
    HostProfile(name="lab-h100", ssh_user="u", ssh_host="h").save(dir_=tmp_path)
    fake_caps = {"gpu_name": "G", "vram_mb": 1024, "gpu_count_detected": 1}
    with patch("lerobot.gui.training.hosts._detect_nvidia_gpu", return_value=fake_caps):
        reg = HostRegistry.auto(workdir=tmp_path, hosts_dir=tmp_path)
    assert {h.id for h in reg.list_hosts()} == {WORKSTATION_HOST_ID, "lab-h100"}
