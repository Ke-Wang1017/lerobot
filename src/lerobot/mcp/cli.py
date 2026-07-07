"""Command-line entry point for the LeRobot MCP server.

Usage:
    lerobot-mcp serve [--transport {stdio,http}] [--host H] [--port P]
                      [--dataset-root PATH] [--db PATH] [--token-store PATH]
    lerobot-mcp issue-token --name NAME --scope SCOPES
    lerobot-mcp revoke-token --name NAME
    lerobot-mcp list-tokens [--include-revoked]

For Claude Code integration:
    # stdio (local dev):
    claude mcp add lerobot --command 'uv run lerobot-mcp serve'
    # http (LAN, requires a token issued via `issue-token`):
    claude mcp add lerobot --transport http \\
      --url http://lerobot.local:7861/mcp --token sk-lr-...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .auth import ALL_SCOPES, TokenStore, default_token_store_path
from .bridge_tools import configure_bridge
from .server import build_server


def _add_serve(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("serve", help="Run the MCP server.")
    p.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    p.add_argument("--host", default="127.0.0.1", help="Bind host for HTTP transport.")
    p.add_argument("--port", type=int, default=7861, help="Bind port for HTTP transport.")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override the dataset cache root (defaults to $HF_LEROBOT_HOME).",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Annotation sidecar SQLite path (defaults to <dataset-root>/_mcp_annotations.sqlite).",
    )
    p.add_argument(
        "--token-store",
        type=Path,
        default=None,
        help="Bearer-token store path (required for http transport; "
        "defaults to $LEROBOT_MCP_TOKEN_STORE or $HF_LEROBOT_HOME/_mcp_tokens.sqlite).",
    )
    p.add_argument(
        "--no-mdns",
        action="store_true",
        help="Skip mDNS advertisement (default: advertise _lerobot-mcp._tcp.local on http transport).",
    )
    p.add_argument(
        "--cors-origin",
        action="append",
        default=None,
        help="Restrict CORS to a specific origin (repeatable). Default: allow any origin "
        "since auth is bearer-token-based.",
    )
    p.add_argument(
        "--gui-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the LeRobot GUI on this host (default: http://127.0.0.1:8000). "
        "Bridge tools (navigate_to / notify_user / ...) POST to "
        "<gui-url>/api/bridge/_dispatch. Pass an empty string to disable the bridge.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )


def _add_token_subcommands(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("issue-token", help="Issue a new bearer token.")
    p.add_argument("--name", required=True, help="Unique device/client name (e.g. alice-laptop).")
    p.add_argument(
        "--scope",
        required=True,
        help=f"Comma-separated scopes from {{{','.join(ALL_SCOPES)}}} (e.g. 'read,annotate').",
    )
    p.add_argument("--token-store", type=Path, default=None)

    p = sub.add_parser("revoke-token", help="Revoke a token by name.")
    p.add_argument("--name", required=True)
    p.add_argument("--token-store", type=Path, default=None)

    p = sub.add_parser("list-tokens", help="List issued tokens (does not show bearers).")
    p.add_argument("--include-revoked", action="store_true")
    p.add_argument("--token-store", type=Path, default=None)
    p.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")


def _resolve_token_store_path(arg: Path | None) -> Path:
    return arg if arg is not None else default_token_store_path()


def _cmd_serve(args: argparse.Namespace) -> int:
    log = logging.getLogger("lerobot.mcp")
    # Bridge tools are configured once per process. Empty string disables.
    bridge_url = (getattr(args, "gui_url", None) or "").strip() or None
    configure_bridge(bridge_url)
    if args.transport == "http":
        ts_path = _resolve_token_store_path(args.token_store)
        token_store = TokenStore(ts_path)
        log.info("HTTP transport: bind=%s:%d token-store=%s", args.host, args.port, ts_path)
        mcp = build_server(
            dataset_root=args.dataset_root,
            db_path=args.db,
            token_store=token_store,
            host=args.host,
            port=args.port,
        )
        mdns_handle = None
        if not args.no_mdns:
            try:
                from lerobot.gui.mdns import advertise

                mdns_handle = advertise(
                    host=args.host,
                    port=args.port,
                    base_name="lerobot",
                    service_type="_lerobot-mcp._tcp.local.",
                    properties={"path": "/mcp", "scopes": "read,annotate,operate"},
                )
                if mdns_handle is not None:
                    log.info("mDNS: %s:%d advertised as %s", args.host, args.port, mdns_handle.hostname)
            except Exception:  # noqa: BLE001 — advertisement is best-effort
                log.warning("mDNS advertise raised; continuing without it", exc_info=True)
        # Wrap the FastMCP starlette app with CORS so the browser-side
        # embedded agent (different origin from this MCP daemon) can reach
        # it. Auth is bearer-token based — CORS does not weaken it; it
        # just lets the browser send the request in the first place.
        import uvicorn
        from starlette.middleware.cors import CORSMiddleware

        app = mcp.streamable_http_app()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=args.cors_origin or ["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=False,
            expose_headers=["mcp-session-id"],
        )
        try:
            config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
            uvicorn.Server(config).run()
        finally:
            if mdns_handle is not None:
                mdns_handle.unregister()
    else:
        mcp = build_server(dataset_root=args.dataset_root, db_path=args.db)
        mcp.run("stdio")
    return 0


def _cmd_issue_token(args: argparse.Namespace) -> int:
    store = TokenStore(_resolve_token_store_path(args.token_store))
    scopes = [s.strip() for s in args.scope.split(",") if s.strip()]
    token = store.issue(name=args.name, scopes=scopes)
    # Print to stdout for piping/copy; warn on stderr that it's not recoverable.
    print(token)
    print(
        f"Issued token {args.name!r} with scopes {scopes}. "
        f"Store it now — it is NOT recoverable after this command exits.",
        file=sys.stderr,
    )
    return 0


def _cmd_revoke_token(args: argparse.Namespace) -> int:
    store = TokenStore(_resolve_token_store_path(args.token_store))
    if store.revoke(args.name):
        print(f"Revoked token {args.name!r}.", file=sys.stderr)
        return 0
    print(f"No active token named {args.name!r}.", file=sys.stderr)
    return 1


def _cmd_list_tokens(args: argparse.Namespace) -> int:
    store = TokenStore(_resolve_token_store_path(args.token_store))
    rows = store.list_tokens(include_revoked=args.include_revoked)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("No tokens.")
        return 0
    headers = ("NAME", "SCOPES", "CREATED", "LAST USED", "REVOKED")
    print("  ".join(f"{h:<24}" if i == 0 else f"{h:<22}" for i, h in enumerate(headers)))
    for r in rows:
        print(
            "  ".join(
                [
                    f"{r['name']:<24}",
                    f"{','.join(r['scopes']):<22}",
                    f"{(r['created_at'] or '-'):<22}",
                    f"{(r['last_used_at'] or '-'):<22}",
                    f"{(r['revoked_at'] or '-'):<22}",
                ]
            )
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lerobot-mcp", description="MCP server for LeRobot.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_serve(sub)
    _add_token_subcommands(sub)
    args = parser.parse_args(argv)

    log_level = getattr(args, "log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    handlers = {
        "serve": _cmd_serve,
        "issue-token": _cmd_issue_token,
        "revoke-token": _cmd_revoke_token,
        "list-tokens": _cmd_list_tokens,
    }
    handler = handlers.get(args.cmd)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
