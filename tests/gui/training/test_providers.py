# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for src/lerobot/gui/training/providers/.

Covers:
  - Protocol conformance via runtime_checkable
  - PersistentSshProvider degenerate behavior (every lifecycle method
    raises; handle_from_profile constructs the right HostHandle)
  - NebiusProvider SKU tables + defensive defaults (spawn/destroy
    stubbed for C2). No cost math by design — see DESIGN.md § What we
    don't try to do.
"""

from __future__ import annotations

import getpass
import time

import pytest

from lerobot.gui.training.jobs import HostProfile
from lerobot.gui.training.providers import (
    NebiusProvider,
    PersistentSshProvider,
    get_provider,
    list_providers,
)
from lerobot.gui.training.providers.nebius import NEBIUS_DISK_WARN_GIB, NebiusConfigError
from lerobot.gui.training.providers.protocol import (
    HostHandle,
    HostProvider,
    SpawnSpec,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def _example_spec(**overrides) -> SpawnSpec:
    defaults: dict = {
        "gpu": "L40S",
        "gpu_count": 1,
        "preemptible": True,
        "disk_gib": 100,
        "image": "ghcr.io/thewisp/lerobot-training:latest",
        "ttl_seconds": 24 * 3600,
    }
    defaults.update(overrides)
    return SpawnSpec(**defaults)


def _example_handle(**overrides) -> HostHandle:
    defaults: dict = {
        "provider": "persistent",
        "provider_resource_id": "test-id",
        "ssh_host": "10.0.0.5",
        "ssh_port": 22,
        "ssh_user": "feit",
        "region": "user-managed",
        "expires_at_unix": int(time.time()) + 3600,
    }
    defaults.update(overrides)
    return HostHandle(**defaults)


# ── Protocol conformance ────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_persistent_satisfies_protocol(self):
        assert isinstance(PersistentSshProvider(), HostProvider)

    def test_nebius_satisfies_protocol(self):
        assert isinstance(NebiusProvider(), HostProvider)

    def test_all_registered_providers_satisfy_protocol(self):
        for pid in list_providers():
            provider = get_provider(pid)
            assert isinstance(provider, HostProvider), (
                f"registered provider {pid!r} does not satisfy HostProvider protocol"
            )

    def test_provider_has_id_and_display_name(self):
        for pid in list_providers():
            provider = get_provider(pid)
            assert provider.id == pid
            assert provider.display_name  # non-empty


# ── Registry ────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_get_provider_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown provider id"):
            get_provider("not-a-vendor")

    def test_get_provider_returns_correct_type(self):
        assert isinstance(get_provider("persistent"), PersistentSshProvider)
        assert isinstance(get_provider("nebius"), NebiusProvider)

    def test_list_providers_includes_v1_set(self):
        providers = list_providers()
        assert "persistent" in providers
        assert "nebius" in providers

    def test_list_providers_sorted(self):
        providers = list_providers()
        assert providers == sorted(providers)


# ── PersistentSshProvider ───────────────────────────────────────────────────


class TestPersistentProvider:
    def test_spawn_raises_not_implemented(self):
        # Persistent hosts are added by the user, not spawned by the GUI.
        with pytest.raises(NotImplementedError, match="Persistent hosts are added"):
            PersistentSshProvider().spawn(_example_spec())

    def test_destroy_raises(self):
        """User owns the VM; the GUI never destroys it. Raising (instead
        of a silent no-op) means a caller that reaches for destroy() on
        the wrong host type fails loudly rather than believing resources
        were freed."""
        with pytest.raises(NotImplementedError, match="user-owned"):
            PersistentSshProvider().destroy(_example_handle())

    def test_verify_destroyed_raises(self):
        """Same fail-fast rationale as destroy(): a vacuous True would
        let a buggy teardown sweep report all-clean on a host type it
        should never have been asked about."""
        with pytest.raises(NotImplementedError, match="user-owned"):
            PersistentSshProvider().verify_destroyed(_example_handle())

    def test_handle_from_profile(self):
        """HostProfile (BYO SSH) → HostHandle, with no private-key fields
        on either end. Auth is resolved by the user's local SSH setup
        (~/.ssh/config / agent / default keys); see DESIGN.md §
        Authentication."""
        profile = HostProfile(
            name="lab-h100",
            ssh_user="feit",
            ssh_host="192.168.1.50",
            ssh_port=22,
            persistent_volume="/data",
        )
        handle = PersistentSshProvider.handle_from_profile(profile)
        assert handle.provider == "persistent"
        assert handle.ssh_host == "192.168.1.50"
        assert handle.ssh_user == "feit"
        assert handle.ssh_port == 22
        assert handle.persistent_volume_id == "/data"
        # Persistent hosts effectively never expire (user owns lifecycle).
        # Sanity check: at least 50 years in the future.
        assert handle.expires_at_unix > int(time.time()) + 50 * 365 * 24 * 3600


# ── NebiusProvider — defensive helpers (the lessons from the bill) ──────────


class TestNebiusDefensiveDefaults:
    def test_disk_warning_fires_above_threshold(self):
        warning = NebiusProvider.disk_warning(640)  # the 640 GB that bit us
        assert warning is not None
        assert "640" in warning
        assert "regardless of VM state" in warning

    def test_disk_warning_silent_below_threshold(self):
        assert NebiusProvider.disk_warning(100) is None
        assert NebiusProvider.disk_warning(NEBIUS_DISK_WARN_GIB) is None

    def test_disk_warning_fires_at_default_terraform_disk(self):
        # The 1.28 TB Terraform default that caused yesterday's bill
        warning = NebiusProvider.disk_warning(1280)
        assert warning is not None

    def test_platform_lookup(self):
        # L40S is the VERIFIED value (from the GPU-smoke Terraform).
        assert NebiusProvider.platform_for("L40S") == "gpu-l40s-d"
        assert NebiusProvider.platform_for("H100") == "gpu-h100-sxm5"

    def test_platform_lookup_unsupported_raises(self):
        with pytest.raises(ValueError, match="no platform mapping"):
            NebiusProvider.platform_for("RTX_4090")

    def test_preset_lookup(self):
        assert NebiusProvider.preset_for("L40S") == "1gpu-16vcpu-96gb"
        assert NebiusProvider.preset_for("H100") == "1gpu-16vcpu-200gb"


# ── NebiusProvider — lifecycle via an injected fake SDK service ─────────────
#
# No credentials/network here, so the provider runs against a fake async
# InstanceServiceClient. The CreateInstanceRequest is built with the REAL
# SDK message classes (so these tests exercise actual message construction)
# — hence importorskip("nebius"). Same loopback-first spirit as SshClient;
# the live spawn→train→destroy smoke is run on the user's Nebius account.

nebius_sdk = pytest.importorskip("nebius.api.nebius.compute.v1")


class _FakeOp:
    def __init__(self, resource_id="computeinstance-abc", ok=True):
        self._rid = resource_id
        self._ok = ok

    async def wait(self):
        return None

    def successful(self):
        return self._ok

    @property
    def resource_id(self):
        return self._rid


class _FakeState:
    def __init__(self, name):
        self.name = name


class _FakePubIP:
    def __init__(self, address):
        self.address = address


class _FakeNic:
    def __init__(self, address):
        self.public_ip_address = _FakePubIP(address)


class _FakeStatus:
    def __init__(self, state, ip):
        self.state = _FakeState(state)
        self.network_interfaces = [_FakeNic(ip)] if ip else []


class _FakeMeta:
    def __init__(self, parent_id):
        self.parent_id = parent_id


class _FakeInstance:
    def __init__(self, state, ip, parent_id="eu-north1"):
        self.status = _FakeStatus(state, ip)
        self.metadata = _FakeMeta(parent_id)


class _FakeService:
    """Async InstanceServiceClient stand-in. Records the requests it's
    handed and replays a scripted sequence of `get` instances so a spawn
    can progress CREATING → RUNNING. ``create``/``delete`` can be made to
    raise to exercise error paths."""

    def __init__(self, get_sequence=None, create_exc=None, get_exc=None, delete_exc=None):
        self.created: list = []
        self.deleted: list = []
        self.gotten: list = []
        self._get_sequence = list(get_sequence or [])
        self._create_exc = create_exc
        self._get_exc = get_exc
        self._delete_exc = delete_exc

    async def create(self, request):
        self.created.append(request)
        if self._create_exc:
            raise self._create_exc
        return _FakeOp()

    async def get(self, request):
        self.gotten.append(request)
        if self._get_exc:
            raise self._get_exc
        if self._get_sequence:
            return self._get_sequence.pop(0)
        raise AssertionError("unexpected extra get() call")

    async def delete(self, request):
        self.deleted.append(request)
        if self._delete_exc:
            raise self._delete_exc
        return _FakeOp()


def _nebius(service=None, **kw):
    kw.setdefault("ssh_public_key", "ssh-ed25519 AAAAFAKEKEY test@host")
    kw.setdefault("project_id", "project-TEST")
    kw.setdefault("subnet_id", "vpcsubnet-TEST")
    kw.setdefault("poll_interval_s", 0)
    kw.setdefault("sleep", lambda _s: None)
    return NebiusProvider(instance_service=service, **kw)


class TestNebiusSpawn:
    def test_spawn_happy_path_returns_handle(self):
        svc = _FakeService(
            get_sequence=[
                _FakeInstance("CREATING", None),
                _FakeInstance("RUNNING", "195.242.29.74"),
            ]
        )
        handle = _nebius(svc).spawn(_example_spec(ttl_seconds=3600))
        assert handle.provider == "nebius"
        assert handle.provider_resource_id == "computeinstance-abc"
        assert handle.ssh_host == "195.242.29.74"
        assert handle.ssh_user == getpass.getuser()  # Fix A: matches GUI server user (#11 workaround)
        assert handle.expires_at_unix > 0

    def test_spawn_strips_cidr_suffix_from_public_ip(self):
        """Nebius returns the public IP in CIDR form (e.g. "195.242.28.4/32").
        ssh_host must be the bare address — otherwise ssh fails with
        "Could not resolve hostname 195.242.28.4/32" before training starts."""
        svc = _FakeService(get_sequence=[_FakeInstance("RUNNING", "195.242.28.4/32")])
        handle = _nebius(svc).spawn(_example_spec(ttl_seconds=3600))
        assert handle.ssh_host == "195.242.28.4"

    def test_public_ip_normalizes_address_forms(self):
        """_public_ip yields a bare host from any form Nebius might return:
        CIDR-suffixed IPv4, a bare IP, or bracketed/CIDR IPv6; None when absent."""
        from lerobot.gui.training.providers.nebius import _public_ip

        assert _public_ip(_FakeInstance("RUNNING", "195.242.28.4/32")) == "195.242.28.4"
        assert _public_ip(_FakeInstance("RUNNING", "10.0.0.5")) == "10.0.0.5"
        assert _public_ip(_FakeInstance("RUNNING", "[2001:db8::1]/128")) == "2001:db8::1"
        assert _public_ip(_FakeInstance("RUNNING", None)) is None

    def test_spawn_builds_request_with_verified_sku(self):
        """Exercises REAL SDK message construction: platform/preset/disk/
        cloud-init must land on the CreateInstanceRequest."""
        svc = _FakeService(get_sequence=[_FakeInstance("RUNNING", "1.2.3.4")])
        _nebius(svc).spawn(_example_spec(gpu="L40S", disk_gib=50, ttl_seconds=3600))
        req = svc.created[0]
        assert req.spec.resources.platform == "gpu-l40s-d"
        assert req.spec.resources.preset == "1gpu-16vcpu-96gb"
        assert req.spec.boot_disk.managed_disk.spec.size_gibibytes == 50
        assert req.metadata.parent_id == "project-TEST"
        nic = req.spec.network_interfaces[0]
        assert nic.subnet_id == "vpcsubnet-TEST"
        # Server-required NIC fields (a real spawn 400s without them):
        # name + an ip_address (empty = auto-allocate the internal IPv4).
        assert nic.name, "network interface name is server-required"
        assert nic.check_presence("ip_address"), "ip_address is server-required"
        assert "ssh-ed25519 AAAAFAKEKEY" in req.spec.cloud_init_user_data

    def test_spawn_cloud_init_arms_ttl_poweroff(self):
        svc = _FakeService(get_sequence=[_FakeInstance("RUNNING", "1.2.3.4")])
        _nebius(svc).spawn(_example_spec(ttl_seconds=2 * 3600))
        ud = svc.created[0].spec.cloud_init_user_data
        assert "shutdown -P +120" in ud  # 2h → 120 min

    def test_spawn_rejects_nonpositive_ttl(self):
        with pytest.raises(ValueError, match="ttl_seconds"):
            _nebius(_FakeService()).spawn(_example_spec(ttl_seconds=0))

    def test_spawn_requires_project_and_subnet(self):
        from lerobot.gui.training.providers.nebius import NebiusConfigError

        with pytest.raises(NebiusConfigError, match="project_id"):
            _nebius(_FakeService(), project_id=None).spawn(_example_spec())
        with pytest.raises(NebiusConfigError, match="subnet_id"):
            _nebius(_FakeService(), subnet_id=None).spawn(_example_spec())

    def test_spawn_create_failure_raises(self):
        svc = _FakeService(create_exc=RuntimeError("quota exceeded"))
        with pytest.raises(RuntimeError, match="quota exceeded"):
            _nebius(svc).spawn(_example_spec())

    def test_spawn_times_out_when_never_ready(self):
        svc = _FakeService(get_sequence=[_FakeInstance("CREATING", None) for _ in range(50)])
        clock = iter([0.0, 0.0, 100.0, 1000.0])  # last exceeds the 900s default
        with pytest.raises(Exception, match="not reachable"):
            _nebius(svc, clock=lambda: next(clock)).spawn(_example_spec())


class TestNebiusDestroyVerify:
    def test_destroy_issues_delete(self):
        svc = _FakeService()
        _nebius(svc).destroy(_example_handle(provider="nebius", provider_resource_id="i-9"))
        assert svc.deleted[0].id == "i-9"

    def test_destroy_idempotent_on_not_found(self):
        svc = _FakeService(delete_exc=RuntimeError("instance not found"))
        _nebius(svc).destroy(_example_handle(provider="nebius"))  # no raise

    def test_destroy_real_failure_raises(self):
        svc = _FakeService(delete_exc=RuntimeError("permission denied"))
        with pytest.raises(RuntimeError, match="permission denied"):
            _nebius(svc).destroy(_example_handle(provider="nebius"))

    def test_verify_destroyed_true_on_not_found(self):
        svc = _FakeService(get_exc=RuntimeError("instance does not exist"))
        assert _nebius(svc).verify_destroyed(_example_handle(provider="nebius")) is True

    def test_verify_destroyed_false_when_still_running(self):
        svc = _FakeService(get_sequence=[_FakeInstance("RUNNING", "1.2.3.4")])
        assert _nebius(svc).verify_destroyed(_example_handle(provider="nebius")) is False

    def test_verify_destroyed_true_when_deleting(self):
        svc = _FakeService(get_sequence=[_FakeInstance("DELETING", None)])
        assert _nebius(svc).verify_destroyed(_example_handle(provider="nebius")) is True

    def test_verify_destroyed_false_on_unknown_error(self):
        """Conservative: an error we can't classify as not-found reports
        NOT destroyed, so the user keeps looking."""
        svc = _FakeService(get_exc=RuntimeError("transient API 503"))
        assert _nebius(svc).verify_destroyed(_example_handle(provider="nebius")) is False


class TestNebiusCloudInit:
    def test_cloud_init_grants_key_sudo_and_ttl(self):
        from lerobot.gui.training.providers.nebius import build_cloud_init

        doc = build_cloud_init("lerobot", "ssh-ed25519 AAAAKEY u@h", ttl_seconds=3600)
        assert doc.startswith("#cloud-config")
        assert "name: lerobot" in doc
        assert "NOPASSWD:ALL" in doc
        assert "ssh-ed25519 AAAAKEY u@h" in doc
        assert "shutdown -P +60" in doc
        # SSH user must join the docker group or `docker run` over SSH gets
        # "permission denied" on /var/run/docker.sock (round-3 ephemeral bug).
        assert "groups: [docker]" in doc

    def test_cloud_init_rejects_key_path(self):
        from lerobot.gui.training.providers.nebius import build_cloud_init

        with pytest.raises(AssertionError, match="not a path"):
            build_cloud_init("lerobot", "/home/u/.ssh/id_ed25519.pub", ttl_seconds=3600)


class TestNebiusAuthError:
    def test_missing_credential_raises_instructive_auth_error(self, monkeypatch):
        """No token + no ambient creds → a NebiusAuthError whose message
        tells the user how to log in (not a raw SDK NoTokenInEnvError)."""
        from lerobot.gui.training.providers.nebius import NebiusAuthError

        monkeypatch.delenv("NEBIUS_IAM_TOKEN", raising=False)
        # No credentials_file and no instance_service → real SDK() build,
        # which fails credential resolution with no token/profile.
        prov = NebiusProvider(
            project_id="project-x",
            subnet_id="vpcsubnet-y",
            ssh_public_key="ssh-ed25519 AAAA t@h",
        )
        with pytest.raises(NebiusAuthError) as ei:
            prov.spawn(_example_spec(ttl_seconds=3600))
        msg = str(ei.value)
        assert "Nebius is not connected" in msg
        assert "service account" in msg
        assert "Nebius connection" in msg


# ── NebiusProvider.list_subnets (subnet discovery for the connection form) ──


class _FakeSubnet:
    def __init__(self, sid, name):
        self.metadata = type("M", (), {"id": sid, "name": name})()


class _FakeSubnetList:
    def __init__(self, items):
        self.items = items


class _FakeSubnetService:
    def __init__(self, items):
        self._items = items
        self.listed: list = []

    async def list(self, request):
        self.listed.append(request)
        return _FakeSubnetList(self._items)


class TestNebiusListSubnets:
    def test_returns_id_and_name_scoped_to_project(self):
        svc = _FakeSubnetService([_FakeSubnet("vpcsubnet-1", "default"), _FakeSubnet("vpcsubnet-2", "lab")])
        out = _nebius(subnet_service=svc).list_subnets("project-TEST")
        assert out == [
            {"id": "vpcsubnet-1", "name": "default"},
            {"id": "vpcsubnet-2", "name": "lab"},
        ]
        assert svc.listed[0].parent_id == "project-TEST"

    def test_empty_project_id_raises(self):
        with pytest.raises(NebiusConfigError):
            _nebius(subnet_service=_FakeSubnetService([])).list_subnets("  ")
