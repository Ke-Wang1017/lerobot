# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SSH transport client — runs training on a remote host via ssh + tmux.

Implements the :class:`TransportClient` Protocol (see ``transport.py``)
end-to-end, exclusively shelling out to the system ``ssh`` / ``scp``
binaries. No paramiko / asyncssh dependency — the host has them anyway,
and the existing ``ControlMaster``/``ControlPersist`` multiplexing is
sufficient for the GUI's polling cadence (typically << 1 op/sec/run).

Design decisions documented in ``src/lerobot/gui/training/DESIGN.md`` § Transport
and re-derived in the implementation workflow ``ssh-client-design``:

  1. **Detached worker via tmux**: every ``launch`` wraps the worker in
     ``tmux new-session -d -s lerobot-<run_id>``. Survives the SSH channel
     closing AND the GUI server restarting. Uniform across docker /
     fake-recipe / future trainer entrypoints.
  2. **session_id is opaque**: encoded as ``<tmux-session-name>|<workdir>``.
     The orchestrator never parses it; only this client does. Lets
     :meth:`exit_code` recover the path to the ``.exit_code`` artefact
     without adding a workdir parameter to the Protocol.
  3. **Exit code via wrapper artefact**: the tmux command is wrapped in
     ``bash -c '<cmd>; echo $? > <workdir>/.exit_code'`` so a process
     exit value survives the container teardown. :meth:`exit_code` cats
     the file. The orchestrator's existing "code=None + ckpt_count>0 →
     completed" fallback covers the brief post-exit window.
  4. **ControlMaster multiplexing**: per-client control socket under
     ``/tmp/lerobot-ssh-cm-<pid>-<host>``; subsequent ssh/scp ops reuse
     the TCP connection. ``ControlPersist=600`` keeps it warm 10 min;
     ``ServerAliveInterval=30`` defeats NAT idle timeouts.
  5. **Path semantics**: ``Path`` arguments are treated as opaque
     absolute remote POSIX paths. Guarded by ``assert path.is_absolute()``
     at every entry point.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from lerobot.gui.training.transport import SshTransport

logger = logging.getLogger(__name__)

# Default timeout for "should be fast" remote ops (read_text, list_dir,
# is_alive, etc.). On loopback these complete in ~30 ms; over a real
# network we'd expect 100-500 ms with ControlMaster multiplexing. 30 s
# leaves room for slow first-connection handshakes. Tunable in Phase 2.
_DEFAULT_TIMEOUT_S = 30.0

# Long timeout for transfer ops (image_pull, fetch_file). The caller
# already runs these on a background thread; we just need to not abort
# a legitimate multi-GB transfer. 24 h is "effectively no timeout."
_LONG_TIMEOUT_S = 24 * 3600.0

# Timeout for the host prereqs install (apt install of Docker +
# nvidia-container-toolkit on a bare host can take a few minutes).
_PREREQS_TIMEOUT_S = 600.0

# The idempotent host-setup script, run over SSH at launch.
# TODO(packaging): ship this inside the package (package-data) so it resolves
# in wheel installs too; today the GUI runs from a source checkout.
_PREREQS_SCRIPT_PATH = Path(__file__).resolve().parents[4] / "scripts" / "training" / "install_prereqs.sh"


def _load_prereqs_script() -> str:
    assert _PREREQS_SCRIPT_PATH.is_file(), f"prereqs script not found at {_PREREQS_SCRIPT_PATH}"
    return _PREREQS_SCRIPT_PATH.read_text(encoding="utf-8")


# Encoded session_id separator. Splits ``<tmux-name>|<workdir>``.
# Chosen because POSIX paths can't contain ``|`` unescaped in a typical
# lerobot run dir layout. If we ever want to allow it, switch to a NUL
# separator and base64-encode the workdir.
_SESSION_SEP = "|"


class SshClient:
    """TransportClient impl for :class:`SshTransport`.

    Construction is cheap; the SSH connection is established lazily on
    the first remote op and reused via ControlMaster for all subsequent
    ops until :meth:`close` or process exit.
    """

    def __init__(self, transport: SshTransport, *, control_path_dir: Path | None = None) -> None:
        self._transport = transport
        # Per-process control socket. ``%C`` template would also work but
        # ssh treats ``%C`` lazily and our tests + tmp-cleanup are tidier
        # with a fully-resolved path.
        #
        # The socket key MUST cover the full connection identity —
        # user + host + port — not just the host. With a host-only key,
        # two saved hosts on the same address with different users shared
        # one authenticated master, and OpenSSH happily executed the
        # second user's commands over the first user's session (found
        # live: a 'smoketest@vm' run silently ran as 'feit'). The digest
        # keeps the identity complete while bounding the path under
        # AF_UNIX's 108-char limit (control sockets are AF_UNIX); the
        # host prefix keeps `ls /tmp` human-readable. pid distinguishes
        # two GUI servers on one workstation.
        base = control_path_dir or Path(tempfile.gettempdir())
        identity = f"{transport.user}@{transport.host}:{transport.port}"
        digest = hashlib.sha256(identity.encode()).hexdigest()[:8]
        self._control_path = base / f"lerobot-ssh-cm-{os.getpid()}-{transport.host[:24]}-{digest}"

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def workdir(self) -> Path:
        """SSH transport has no fixed workdir at construction time — each
        run passes its own via :meth:`launch`. Returns a sentinel path
        that's intentionally unopenable locally; the orchestrator never
        calls this property today (it uses ``RunPaths.root`` directly)."""
        return Path("/__ssh_remote_no_fixed_workdir__")

    # ── ssh / scp argv builders ───────────────────────────────────────────

    def _ssh_options(self) -> list[str]:
        """Common SSH options for both ``ssh`` and ``scp``. The
        ControlMaster socket is set up by the FIRST op that calls ssh;
        subsequent ops with the same options reuse it. ``BatchMode=yes``
        disables interactive prompts so a missing key fails fast instead
        of hanging the GUI's poll thread."""
        return [
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={self._control_path}",
            "-o",
            "ControlPersist=600",
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "StrictHostKeyChecking=accept-new",
            # Bound the initial connect so a not-yet-ready host fails fast
            # (ssh exits 255) instead of hanging to the subprocess timeout —
            # this is what makes the wait_until_ready retry loop snappy.
            "-o",
            "ConnectTimeout=10",
        ]

    def _ssh_argv(self, *remote_argv: str) -> list[str]:
        # No ``-i <key>`` flag by design. ``ssh`` resolves the identity
        # from the user's setup: ``~/.ssh/config`` Host blocks, ssh-agent,
        # then default-path keys. Matches VS Code Remote-SSH /
        # JetBrains Gateway / ``gh codespace ssh`` — and means GUI
        # server compromise cannot exfiltrate user SSH keys because
        # their bytes never enter this process.
        t = self._transport
        argv = ["ssh", *self._ssh_options(), "-p", str(t.port)]
        argv.append(f"{t.user}@{t.host}")
        argv.extend(remote_argv)
        return argv

    def _scp_argv(self) -> list[str]:
        t = self._transport
        return ["scp", *self._ssh_options(), "-P", str(t.port)]

    def _exec(
        self,
        remote_cmd: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT_S,
        stdin: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run a single remote shell command. Returns the CompletedProcess
        verbatim; callers branch on returncode + stdout / stderr.

        Prepends ``~/.local/bin`` to PATH because SSH non-interactive
        non-login sessions get a minimal PATH (just ``/usr/bin`` +
        ``/usr/local/bin`` style entries) — user-installed binaries
        (``tmux``, rootless docker, ``uv``) land in ``~/.local/bin`` by
        convention and would otherwise be invisible. Login shells via
        ``bash -lc`` would also work but pull in heavier startup costs
        (full profile sourcing) per op; a simple PATH prepend is faster
        and more predictable.
        """
        cmd_with_path = f'export PATH="$HOME/.local/bin:$PATH"; {remote_cmd}'
        return subprocess.run(
            self._ssh_argv(cmd_with_path),
            capture_output=True,
            input=stdin,
            timeout=timeout,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def launch(
        self,
        command: list[str],
        env: dict[str, str],
        workdir: Path,
        log_path: Path,
    ) -> str:
        """Spawn ``command`` inside a detached tmux session on the remote.

        Pre: ``workdir`` and ``log_path`` are absolute remote POSIX paths.
        Post: returned session_id is ``<tmux-name>|<workdir-string>``.
        The tmux name encodes the workdir's basename (typically the
        run_id) so ``tmux ls`` on the remote shows human-readable entries.
        """
        assert workdir.is_absolute(), f"SshClient requires absolute remote paths, got {workdir}"
        assert log_path.is_absolute(), f"SshClient requires absolute remote paths, got {log_path}"

        tmux_name = f"lerobot-{workdir.name}"
        # Build the inner shell command: cd, env exports, command, then
        # echo the exit code to a sibling file so we can recover it later.
        env_exports = " ".join(f"{shlex.quote(k)}={shlex.quote(v)}" for k, v in env.items())
        cmd_quoted = shlex.join(command)
        workdir_q = shlex.quote(str(workdir))
        log_q = shlex.quote(str(log_path))
        exit_code_q = shlex.quote(str(workdir / ".exit_code"))

        # Two-stage wrapping:
        # 1. ``env_part cmd_part >> log 2>&1; echo $? > .exit_code`` — the
        #    semicolon ensures echo runs even on worker failure, so the
        #    artefact exists for both success and crash paths.
        # 2. ``bash -lc '<above>'`` — login shell so the worker inherits
        #    the remote user's PATH (matters for docker, uv, etc.).
        inner = f"cd {workdir_q} && {env_exports} {cmd_quoted} >> {log_q} 2>&1; echo $? > {exit_code_q}"
        bash_invocation = f"bash -lc {shlex.quote(inner)}"

        # Ensure remote dirs exist, then create the tmux session.
        log_parent_q = shlex.quote(str(log_path.parent))
        remote_cmd = (
            f"mkdir -p {workdir_q} {log_parent_q} && "
            f"tmux new-session -d -s {shlex.quote(tmux_name)} {shlex.quote(bash_invocation)}"
        )
        r = self._exec(remote_cmd)
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace")[-400:]
            raise RuntimeError(f"SshClient.launch failed: rc={r.returncode} stderr={err}")
        return f"{tmux_name}{_SESSION_SEP}{workdir}"

    @staticmethod
    def _parse_session(session_id: str) -> tuple[str, str | None]:
        """Split the encoded session_id. Pre-restart and well-formed:
        returns (tmux_name, workdir). Malformed (e.g. an old plain PID
        from a subprocess run): returns (session_id, None)."""
        if _SESSION_SEP not in session_id:
            return session_id, None
        name, _, workdir = session_id.partition(_SESSION_SEP)
        return name, workdir or None

    def is_alive(self, session_id: str) -> bool:
        tmux_name, _ = self._parse_session(session_id)
        r = self._exec(f"tmux has-session -t {shlex.quote(tmux_name)} 2>/dev/null")
        return r.returncode == 0

    def exit_code(self, session_id: str) -> int | None:
        """Read the worker's exit code from the ``.exit_code`` artefact
        the launch wrapper writes. Returns None if:
          - the artefact doesn't exist (worker still running, OR just
            exited and the write hasn't flushed yet, OR pre-restart
            session_id with no workdir component)
          - the file content isn't an integer (truncated write)
        The orchestrator's ``_write_terminal_event_from_exit`` treats
        ``None`` as "fall back to checkpoint count," so a transient
        post-exit None resolves correctly on the next poll cycle.
        """
        _, workdir = self._parse_session(session_id)
        if workdir is None:
            return None
        path = shlex.quote(f"{workdir}/.exit_code")
        r = self._exec(f"cat {path} 2>/dev/null")
        if r.returncode != 0 or not r.stdout.strip():
            return None
        try:
            return int(r.stdout.strip())
        except ValueError:
            return None

    def stop(self, session_id: str, *, force: bool = False) -> None:
        """Signal the worker. Tmux's kill-session sends SIGHUP to all
        panes by default; we explicitly TERM (or KILL with ``force``)
        every pane's process group first so docker run propagates the
        signal to the container, then kill the tmux session to clean up
        the wrapping shell. Both sub-commands are idempotent — a
        non-existent session yields a benign non-zero exit we ignore."""
        tmux_name, _ = self._parse_session(session_id)
        sig = "KILL" if force else "TERM"
        name_q = shlex.quote(tmux_name)
        # list-panes prints one pane_pid per line; xargs sends the signal
        # to each pane's process group ("-<pid>" arg to kill). All sub-
        # ops are stderr-suppressed; the final ``|| true`` swallows
        # tmux kill-session's "no such session" error after the panes
        # are already dead.
        remote = (
            f"tmux list-panes -t {name_q} -F '#{{pane_pid}}' 2>/dev/null "
            f"| xargs -r -I{{}} kill -{sig} -{{}} 2>/dev/null; "
            f"tmux kill-session -t {name_q} 2>/dev/null || true"
        )
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._exec(remote)

    # ── File ops (remote paths) ───────────────────────────────────────────

    def read_text(self, path: Path) -> str | None:
        assert path.is_absolute(), f"absolute remote path required, got {path}"
        r = self._exec(f"cat {shlex.quote(str(path))} 2>/dev/null")
        if r.returncode != 0:
            return None
        return r.stdout.decode("utf-8", errors="replace")

    def read_bytes_from_offset(self, path: Path, offset: int) -> tuple[bytes, int]:
        assert offset >= 0, f"offset must be non-negative, got {offset}"
        assert path.is_absolute()
        # tail -c +N is 1-indexed (byte N is the FIRST byte returned).
        # We want bytes starting at index ``offset``, so pass ``offset+1``.
        # On missing file, tail prints to stderr; suppressed via 2>/dev/null.
        r = self._exec(f"tail -c +{offset + 1} {shlex.quote(str(path))} 2>/dev/null")
        if r.returncode != 0:
            return b"", offset
        return r.stdout, offset + len(r.stdout)

    def fetch_file(self, src: Path, dst: Path) -> None:
        assert src.is_absolute()
        dst.parent.mkdir(parents=True, exist_ok=True)
        t = self._transport
        argv = self._scp_argv() + [f"{t.user}@{t.host}:{src}", str(dst)]
        r = subprocess.run(argv, capture_output=True, timeout=_LONG_TIMEOUT_S)
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace")[-400:]
            raise RuntimeError(f"scp {src} → {dst} failed: rc={r.returncode} stderr={err}")

    def read_tail(self, path: Path, n_bytes: int) -> bytes:
        assert n_bytes >= 0
        assert path.is_absolute()
        if n_bytes == 0:
            return b""
        r = self._exec(f"tail -c {n_bytes} {shlex.quote(str(path))} 2>/dev/null")
        return r.stdout if r.returncode == 0 else b""

    def list_dir(self, path: Path) -> list[Path]:
        assert path.is_absolute()
        # ``find -print0`` survives filenames with newlines or spaces;
        # ``ls`` parsing is fragile. ``-mindepth 1 -maxdepth 1`` makes
        # find behave like Path.iterdir() — children only, no recursion,
        # and excludes the parent dir from output.
        r = self._exec(f"find {shlex.quote(str(path))} -mindepth 1 -maxdepth 1 -print0 2>/dev/null")
        if r.returncode != 0 or not r.stdout:
            return []
        return [Path(p.decode("utf-8")) for p in r.stdout.split(b"\x00") if p]

    def sha256_of(self, path: Path) -> str | None:
        assert path.is_absolute()
        # sha256sum's output format is ``<hex>  <path>\n``. Take the
        # first whitespace-separated token. Non-zero exit (file missing)
        # → None; matches SubprocessClient's behavior.
        r = self._exec(f"sha256sum {shlex.quote(str(path))} 2>/dev/null")
        if r.returncode != 0 or not r.stdout:
            return None
        token = r.stdout.split(maxsplit=1)
        if not token:
            return None
        return token[0].decode("ascii")

    def append_text(self, path: Path, text: str) -> None:
        assert path.is_absolute()
        # Stream the text via stdin (``cat >> file``) to avoid argv
        # length limits and shell quoting on the payload. mkdir -p is
        # idempotent and runs in the same SSH call.
        remote = f"mkdir -p {shlex.quote(str(path.parent))} && cat >> {shlex.quote(str(path))}"
        r = self._exec(remote, stdin=text.encode("utf-8"))
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace")[-200:]
            raise RuntimeError(f"append_text({path}) failed: rc={r.returncode} stderr={err}")

    def host_identity(self) -> tuple[int, int, str]:
        """``(uid, gid, home)`` of the REMOTE user — queried once over
        SSH and cached for the client's lifetime (a user's uid/home don't
        change mid-session). This is the launch-time truth the recipe
        placeholders resolve against; the GUI server's own uid is
        irrelevant here (remote users are not reliably uid 1000)."""
        cached = getattr(self, "_host_identity", None)
        if cached is not None:
            return cached
        r = self._exec('echo "$(id -u) $(id -g) $HOME"')
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace")[-200:]
            raise RuntimeError(f"host_identity failed: rc={r.returncode} stderr={err}")
        uid_s, gid_s, home = r.stdout.decode().strip().split(" ", 2)
        identity = (int(uid_s), int(gid_s), home)
        if not home.startswith("/"):
            raise RuntimeError(
                f"remote $HOME is not an absolute path ({home!r}) — check the remote shell environment"
            )
        self._host_identity = identity
        return identity

    def wait_until_ready(
        self,
        *,
        timeout_s: float = 300.0,
        probe_timeout_s: float = 15.0,
        poll_interval_s: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Block until the host accepts SSH, or raise after ``timeout_s``.

        A freshly-spawned cloud VM reports RUNNING before sshd / cloud-init
        finish, so the first remote op races the boot. Poll a trivial command
        until it answers. ``sleep``/``clock`` are injectable for tests.

        Pre: the transport's host/port/user are set.
        Post: returns iff SSH answered (rc 0); otherwise raises ``RuntimeError``
        naming the two real causes — still booting, or inbound TCP/22 blocked.
        """
        assert timeout_s > 0 and probe_timeout_s > 0
        deadline = clock() + timeout_s
        last = "no attempt made"
        while True:
            try:
                r = self._exec("true", timeout=probe_timeout_s)
                if r.returncode == 0:
                    return
                last = f"rc={r.returncode} stderr={r.stderr.decode('utf-8', 'replace').strip()[-200:]}"
            except subprocess.TimeoutExpired:
                last = f"connect timed out after {probe_timeout_s:.0f}s"
            except OSError as e:
                last = f"{type(e).__name__}: {e}"
            if clock() >= deadline:
                t = self._transport
                raise RuntimeError(
                    f"{t.user}@{t.host}:{t.port} did not accept SSH within {timeout_s:.0f}s "
                    f"(last: {last}). The VM may still be booting, or inbound TCP/22 may be "
                    f"blocked by the cloud security group."
                )
            sleep(poll_interval_s)

    def ensure_prereqs(self) -> None:
        """Install/verify Docker + nvidia-container-toolkit + docker-group on the
        remote host, then reset the control socket so the user's (possibly new)
        docker-group membership applies to subsequent ops.

        Runs the idempotent ``install_prereqs.sh`` over SSH as root. The
        container GPU smoke is skipped (the real training run is the end-to-end
        GPU test); the script still does a cheap host-side ``nvidia-smi`` check.

        Pre: the host accepts SSH (call ``wait_until_ready`` first for a fresh
        VM) and the SSH user has passwordless sudo.
        Post: ``docker`` is usable by the SSH user, or ``RuntimeError`` is raised.
        """
        script = _load_prereqs_script()
        r = self._exec(
            "sudo LEROBOT_PREREQS_SKIP_CONTAINER_SMOKE=1 bash -s",
            stdin=script.encode("utf-8"),
            timeout=_PREREQS_TIMEOUT_S,
        )
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", "replace").strip()[-400:]
            raise RuntimeError(f"host prereqs setup failed: rc={r.returncode}\n{err}")
        # The script may have just added the SSH user to the docker group;
        # group membership is fixed at login, so drop the persistent
        # ControlMaster — the next op re-logs in and picks up the new group.
        self.close()

    def ensure_dir(self, path: Path) -> None:
        assert path.is_absolute()
        r = self._exec(f"mkdir -p {shlex.quote(str(path))}")
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace")[-200:]
            raise RuntimeError(f"ensure_dir({path}) failed: rc={r.returncode} stderr={err}")

    # ── Docker ops ────────────────────────────────────────────────────────

    def image_inspect(self, tag: str) -> bool:
        r = self._exec(
            f"docker image inspect {shlex.quote(tag)} >/dev/null 2>&1",
            timeout=_DEFAULT_TIMEOUT_S,
        )
        return r.returncode == 0

    def image_pull(self, tag: str) -> tuple[bool, str]:
        # Long timeout: docker pull on a multi-GB image over a slow link
        # can legitimately take 10+ min. The caller (orchestrator's
        # _prepare_and_launch thread) already runs us off the request
        # thread so the GUI isn't blocked.
        r = self._exec(f"docker pull {shlex.quote(tag)}", timeout=_LONG_TIMEOUT_S)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).decode("utf-8", errors="replace")[-1000:]
            return False, tail
        return True, ""

    def image_size(self, tag: str) -> int | None:
        # Note the doubled braces: `{{.Size}}` in the docker format
        # string lands as `{.Size}` after Python's f-string escapes it.
        r = self._exec(
            f"docker image inspect -f '{{{{.Size}}}}' {shlex.quote(tag)} 2>/dev/null",
            timeout=10.0,
        )
        if r.returncode != 0:
            return None
        try:
            return int(r.stdout.strip())
        except ValueError:
            return None

    # ── Connection teardown ───────────────────────────────────────────────

    def close(self) -> None:
        """Tear down the ControlMaster socket. Idempotent; safe to call
        from a finalizer. ssh -O exit returns immediately."""
        with contextlib.suppress(Exception):
            subprocess.run(
                self._ssh_argv("-O", "exit"),
                capture_output=True,
                timeout=5.0,
            )
