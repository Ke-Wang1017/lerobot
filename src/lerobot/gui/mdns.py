# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""mDNS / Zeroconf advertising for the GUI host service.

Goal: from any device on the LAN (laptop, phone, tablet) you can open
``http://lerobot.local:<port>/`` and hit your robot host without
knowing its IP. The standard, browser-native mechanism for this is
multicast DNS — modern macOS, Windows 10+, Linux (with avahi-daemon),
iOS, and Android all resolve ``*.local`` hostnames natively.

This module is import-guarded: if ``zeroconf`` isn't installed, or if
multicast is blocked on the network, the advertise call returns None
and the rest of the GUI keeps working. The raw LAN IP printed in the
startup banner is the always-available fallback.

Two consequences worth documenting:

  - Advertising only makes sense when the server binds to a non-loopback
    interface (i.e. ``--host 0.0.0.0`` or a LAN IP). With ``--host
    127.0.0.1`` we'd publish a hostname that resolves to the LAN IP but
    the server wouldn't be reachable from outside the loopback — a
    confusing dead-end for the user. ``advertise()`` checks this and
    declines to publish in the loopback case.

  - If multiple robot hosts on the same LAN both call ``advertise()``
    with the same name, zeroconf's name-conflict logic appends a
    counter (``lerobot-2.local``, ``lerobot-3.local``, ...). The
    actually-used name is returned so the startup banner can show it.
"""

from __future__ import annotations

import logging
import socket
from typing import Any

logger = logging.getLogger(__name__)


def detect_lan_ip() -> str | None:
    """Return this machine's primary LAN IP (the one a client on the same
    network would address us at), or None if we couldn't determine it.

    Uses the standard "connect a UDP socket to a known external address
    and read back our local address" trick. We don't actually send
    anything — the kernel just picks the route. Works even without
    internet access since UDP connect() is local-only.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # The 8.8.8.8 address is convenient; any non-loopback target works.
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        return ip if not ip.startswith("127.") else None
    except OSError:
        return None
    finally:
        s.close()


class _MdnsHandle:
    """Opaque handle returned by ``advertise``. Use ``unregister()``
    on server shutdown to send a goodbye packet.

    Stored as a class so the server module can hold one reference and
    not have to know about zeroconf types directly.
    """

    def __init__(self, zc: Any, info: Any, hostname: str, port: int):
        self._zc = zc
        self._info = info
        self.hostname = hostname  # e.g. "lerobot.local" or "lerobot-2.local"
        self.port = port

    def unregister(self) -> None:
        try:
            self._zc.unregister_service(self._info)
        except Exception:
            logger.warning("mDNS unregister failed", exc_info=True)
        try:
            self._zc.close()
        except Exception:
            logger.warning("zeroconf close failed", exc_info=True)


def advertise(
    *,
    host: str,
    port: int,
    base_name: str = "lerobot",
    service_type: str = "_http._tcp.local.",
    properties: dict[str, str] | None = None,
) -> _MdnsHandle | None:
    """Advertise this host as ``<base_name>.local`` over mDNS.

    Args:
        host: the interface the server is bound to (passed as ``--host``).
            Loopback values cause this function to no-op since the
            advertised hostname would resolve to an unreachable address.
        port: the TCP port the server is listening on.
        base_name: the desired short hostname. ``zeroconf`` will append
            a counter (``-2``, ``-3``, ...) if the name is already in use.
        service_type: the mDNS service type to register under. Defaults
            to ``_http._tcp.local.`` (browsers discover this for hosts
            advertising a web GUI). Pass ``_lerobot-mcp._tcp.local.`` for
            the MCP daemon so MCP clients can filter for it cleanly.
        properties: extra TXT-record key/value pairs to attach to the
            service registration. Defaults to ``{"path": "/"}`` so a
            generic mDNS browser can show the GUI root URL.

    Returns:
        An ``_MdnsHandle`` (call ``.unregister()`` on shutdown), or
        ``None`` if the bind interface is loopback-only (so advertising
        would point at an unreachable address).

    Raises:
        RuntimeError: when the caller binds to a non-loopback interface
            but ``zeroconf`` isn't installed. Silently skipping in that
            case is a footgun — the user explicitly opted into LAN
            access and would only discover the missing dependency when
            ``lerobot.local`` fails to resolve from another device.
            Better to fail loudly at startup with an actionable message.
    """
    # Loopback-only bind → don't advertise. The advertised A-record
    # would point to our LAN IP, but the server wouldn't accept
    # connections to it. Better to be silent than misleading.
    if host in ("127.0.0.1", "::1", "localhost"):
        logger.debug("mDNS advertise skipped: server bound to loopback only")
        return None

    try:
        from zeroconf import IPVersion, ServiceInfo, Zeroconf
    except ImportError as e:
        raise RuntimeError(
            "Cannot advertise lerobot.local: the `zeroconf` package is not "
            "installed in this Python environment.\n\n"
            "You bound the server to a non-loopback interface "
            f"(--host {host}), which means mDNS advertising is needed so "
            "the GUI is reachable as http://lerobot.local:<port>/ from "
            "other devices on the LAN.\n\n"
            "Install with one of:\n"
            "    pip install zeroconf\n"
            "    uv pip install zeroconf\n"
            "    pip install -e '.[gui]'   # also picks up zeroconf via the gui extra\n\n"
            "Or, if you don't need LAN access right now, bind to loopback "
            "instead: --host 127.0.0.1"
        ) from e

    lan_ip = detect_lan_ip()
    if lan_ip is None:
        logger.warning("mDNS advertise skipped: could not determine LAN IP")
        return None

    zc = Zeroconf(ip_version=IPVersion.V4Only)
    if properties is None:
        properties = {"path": "/"}

    # Pick a unique short hostname. zeroconf will reject collisions; we
    # retry with a counter suffix so two robot hosts on the same LAN can
    # coexist without manual intervention.
    server_name: str = ""
    info: Any = None
    for attempt in range(1, 20):
        candidate = base_name if attempt == 1 else f"{base_name}-{attempt}"
        server_name = f"{candidate}.local."
        info = ServiceInfo(
            type_=service_type,
            name=f"{candidate}.{service_type}",
            addresses=[socket.inet_aton(lan_ip)],
            port=port,
            server=server_name,
            properties=properties,
        )
        try:
            zc.register_service(info, allow_name_change=False)
            break
        except Exception as e:
            # NonUniqueNameException or similar — try the next suffix.
            logger.debug("mDNS name %s taken (%s), trying next", candidate, e)
            info = None
            continue

    if info is None:
        logger.warning("mDNS advertise: every name from %s..%s-19 was taken", base_name, base_name)
        import contextlib

        with contextlib.suppress(Exception):
            zc.close()
        return None

    # Strip the trailing dot for the user-facing form.
    hostname = server_name.rstrip(".")
    logger.info("mDNS: advertised %s → %s:%d", hostname, lan_ip, port)
    return _MdnsHandle(zc=zc, info=info, hostname=hostname, port=port)
