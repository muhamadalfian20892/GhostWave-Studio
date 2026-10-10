"""
Support Ticket Client for GhostWave Studio v1.4.0.

Communicates with the secure Cloudflare Worker serverless relay to create tickets,
retrieve discussion threads, and submit user replies to the private GitHub repository.
"""

from __future__ import annotations

import json
import platform
import secrets
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_WORKER_URL = "https://ghostwave-studio-private.hafiyanajah.workers.dev"
REQUEST_TIMEOUT_SECONDS = 15.0


def generate_ticket_key() -> str:
    """Generates a secure random 32-character hex key to protect ticket privacy."""
    return secrets.token_hex(16)


def get_system_diagnostics(app_version: str = "1.5.1") -> str:
    """Collects system details for diagnostic ticket reports."""
    try:
        os_info = f"{platform.platform()} ({platform.architecture()[0]})"
    except Exception:
        os_info = "Windows (Unknown)"
    try:
        py_ver = platform.python_version()
    except Exception:
        py_ver = "Unknown"
    try:
        mach = platform.machine()
    except Exception:
        mach = "x86_64"

    lines = [
        f"OS: {os_info}",
        f"Python: {py_ver}",
        f"App Version: {app_version}",
        f"Machine: {mach}"
    ]
    return "\n".join(lines)


def create_ticket(
    category: str,
    title: str,
    description: str,
    client_version: str = "1.5.1",
    attach_sys_info: bool = True,
    worker_url: str = DEFAULT_WORKER_URL
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Submits a new support ticket to the serverless relay.
    Returns: (success_bool, result_dict, error_message_or_none)
    """
    ticket_key = generate_ticket_key()
    full_description = description.strip()

    if attach_sys_info:
        diagnostics = get_system_diagnostics(client_version)
        full_description += f"\n\n### System Diagnostics\n```\n{diagnostics}\n```"

    payload = {
        "title": title.strip(),
        "description": full_description,
        "category": category.strip() or "Support",
        "client_version": client_version,
        "ticket_key": ticket_key
    }

    endpoint = f"{worker_url.rstrip('/')}/api/tickets"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"GhostWave-Studio/{client_version}"
    }

    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status_code = response.getcode()
            body_bytes = response.read()
            data = json.loads(body_bytes.decode("utf-8"))

            if status_code in (200, 201) and data.get("success"):
                data["ticket_key"] = ticket_key
                data["category"] = category
                return True, data, None
            return False, {}, data.get("error", "Unknown server response")

    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            return False, {}, err_body.get("error", f"HTTP Error {e.code}")
        except Exception:
            return False, {}, f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return False, {}, f"Connection failed: {e.reason}"
    except Exception as e:
        return False, {}, str(e)


def fetch_ticket(
    ticket_id: int,
    ticket_key: str,
    worker_url: str = DEFAULT_WORKER_URL
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Fetches ticket status and comment thread for a specific ticket.
    Returns: (success_bool, ticket_data, error_message_or_none)
    """
    query = urllib.parse.urlencode({
        "id": str(ticket_id),
        "key": ticket_key
    })
    endpoint = f"{worker_url.rstrip('/')}/api/tickets?{query}"
    headers = {
        "User-Agent": "GhostWave-Studio/1.5.0"
    }

    try:
        req = urllib.request.Request(endpoint, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status_code = response.getcode()
            body_bytes = response.read()
            data = json.loads(body_bytes.decode("utf-8"))

            if status_code == 200:
                return True, data, None
            return False, {}, data.get("error", "Failed to retrieve ticket")

    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            return False, {}, err_body.get("error", f"HTTP Error {e.code}")
        except Exception:
            return False, {}, f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return False, {}, f"Connection failed: {e.reason}"
    except Exception as e:
        return False, {}, str(e)


def reply_to_ticket(
    ticket_id: int,
    ticket_key: str,
    message: str,
    worker_url: str = DEFAULT_WORKER_URL
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Submits a follow-up comment to an existing ticket thread.
    Returns: (success_bool, result_dict, error_message_or_none)
    """
    clean_msg = message.strip()
    if not clean_msg:
        return False, {}, "Reply message cannot be empty."

    payload = {
        "ticket_id": ticket_id,
        "ticket_key": ticket_key,
        "message": clean_msg
    }
    endpoint = f"{worker_url.rstrip('/')}/api/tickets/reply"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "GhostWave-Studio/1.5.0"
    }

    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status_code = response.getcode()
            body_bytes = response.read()
            data = json.loads(body_bytes.decode("utf-8"))

            if status_code in (200, 201) and data.get("success"):
                return True, data, None
            return False, {}, data.get("error", "Failed to post reply")

    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            return False, {}, err_body.get("error", f"HTTP Error {e.code}")
        except Exception:
            return False, {}, f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return False, {}, f"Connection failed: {e.reason}"
    except Exception as e:
        return False, {}, str(e)
