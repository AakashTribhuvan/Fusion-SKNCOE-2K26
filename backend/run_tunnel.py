"""Run the local demo behind a temporary Cloudflare Quick Tunnel.

Windows Job Object kill-on-close ensures the API and cloudflared processes do
not survive if this launcher is closed unexpectedly.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from ctypes import wintypes
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_URL = "http://127.0.0.1:8000"
TUNNEL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.IGNORECASE)


def tunnel_url_from_log(line: str) -> str | None:
    match = TUNNEL_PATTERN.search(line)
    if not match or match.group(0).lower() == "https://api.trycloudflare.com":
        return None
    return match.group(0)


class BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", BasicLimitInformation),
        ("IoInfo", IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class ChildProcessGroup:
    """Windows process job configured to terminate all children on close."""

    KILL_ON_JOB_CLOSE = 0x2000
    EXTENDED_LIMIT_INFORMATION = 9

    def __init__(self) -> None:
        self.handle = None
        self.enabled = False
        if os.name != "nt":
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD
        )
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        self.kernel32 = kernel32
        self.handle = kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            return
        limits = ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = self.KILL_ON_JOB_CLOSE
        self.enabled = bool(kernel32.SetInformationJobObject(
            self.handle, self.EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits), ctypes.sizeof(limits),
        ))

    def add(self, process: subprocess.Popen[str]) -> bool:
        if not self.enabled or not self.handle:
            return False
        return bool(self.kernel32.AssignProcessToJobObject(self.handle, process._handle))

    def close(self) -> None:
        if self.handle:
            self.kernel32.CloseHandle(self.handle)
            self.handle = None


def find_cloudflared() -> str | None:
    local_copy = ROOT / ".tools" / "cloudflared.exe"
    if local_copy.is_file():
        try:
            check = subprocess.run([str(local_copy), "--version"], capture_output=True, timeout=10)
            if check.returncode == 0:
                return str(local_copy)
        except (OSError, subprocess.SubprocessError):
            pass
        local_copy.unlink(missing_ok=True)
    return shutil.which("cloudflared")


def download_cloudflared() -> str:
    """Download the official Windows binary and verify the release SHA256."""
    arch = "386" if platform.architecture()[0] == "32bit" else "amd64"
    asset_name = f"cloudflared-windows-{arch}.exe"
    headers = {"User-Agent": "FusionPrototype/0.1", "Accept": "application/vnd.github+json"}
    request = urllib.request.Request(
        "https://api.github.com/repos/cloudflare/cloudflared/releases/latest",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        release = json.loads(response.read(2_000_000))
    asset = next((item for item in release.get("assets", []) if item.get("name") == asset_name), None)
    asset_url = asset.get("browser_download_url", "") if asset else ""
    if not asset_url.startswith("https://github.com/cloudflare/cloudflared/releases/download/"):
        raise RuntimeError(f"Official Cloudflare release did not include {asset_name}.")
    checksum_line = re.search(
        rf"(?m)^\s*{re.escape(asset_name)}:\s*([0-9a-fA-F]{{64}})\s*$",
        release.get("body", ""),
    )
    if not checksum_line:
        raise RuntimeError("Could not find the binary's SHA256 checksum in the official release notes.")

    tools_dir = ROOT / ".tools"
    tools_dir.mkdir(exist_ok=True)
    destination = tools_dir / "cloudflared.exe"
    partial = tools_dir / "cloudflared.exe.download"
    digest = hashlib.sha256()
    total = 0
    download_request = urllib.request.Request(
        asset_url, headers={"User-Agent": headers["User-Agent"]}
    )
    try:
        with urllib.request.urlopen(download_request, timeout=45) as response, partial.open("wb") as output:
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > 100 * 1024 * 1024:
                    raise RuntimeError("Cloudflared download exceeded the 100 MB safety limit.")
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != checksum_line.group(1).lower():
            raise RuntimeError("Cloudflared SHA256 did not match Cloudflare's release checksum.")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return str(destination)


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def wait_for_health(url: str, timeout: float, process: subprocess.Popen[str]) -> str | None:
    deadline = time.monotonic() + timeout
    last_error = "no response received"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return "the API process exited before becoming healthy"
        try:
            request = urllib.request.Request(url, headers={"cf-skip-browser-warning": "1"})
            with urllib.request.urlopen(request, timeout=2) as response:
                body = response.read(128)
                payload = json.loads(body)
                if response.status == 200 and isinstance(payload, dict) and payload.get("status") == "ok":
                    return None
                last_error = f"unexpected HTTP {response.status} health response"
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = str(error)
        time.sleep(0.5)
    return last_error


def wait_for_tunnel_connection(
    log_queue: list[str], timeout: float, process: subprocess.Popen[str]
) -> str | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return "cloudflared exited before registering a tunnel connection"
        if any("Registered tunnel connection" in line for line in log_queue):
            return None
        time.sleep(0.2)
    return "cloudflared did not register a connection with Cloudflare"


def main() -> int:
    if os.name != "nt":
        print("This launcher is intended for Windows.", file=sys.stderr)
        return 2
    cloudflared = find_cloudflared()
    if not cloudflared:
        print("cloudflared is missing. Downloading the official Windows binary and verifying its SHA256…", flush=True)
        try:
            cloudflared = download_cloudflared()
        except (OSError, urllib.error.URLError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            print(f"ERROR: Could not prepare Cloudflare Tunnel: {error}")
            print("Check your internet connection, then run StartPrototype.bat again.")
            return 2

    with socket.socket() as listener_check:
        if listener_check.connect_ex(("127.0.0.1", 8000)) == 0:
            print("ERROR: Port 8000 is already in use. Stop that service before opening a public tunnel.")
            return 2

    children = ChildProcessGroup()
    if not children.enabled:
        print("ERROR: Windows could not create a close-on-exit process job.")
        print("No public tunnel was started; check Windows security policy and try again.")
        children.close()
        return 2
    tunnel: subprocess.Popen[str] | None = None
    api: subprocess.Popen[str] | None = None
    log_queue: list[str] = []
    tunnel_url: str | None = None
    try:
        print("Starting a temporary Cloudflare Quick Tunnel…", flush=True)
        tunnel = subprocess.Popen(
            [cloudflared, "tunnel", "--no-autoupdate", "--url", LOCAL_URL],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        if not children.add(tunnel):
            stop_process(tunnel)
            print("ERROR: Could not attach cloudflared to the close-on-exit process job.")
            return 2

        def collect_logs() -> None:
            assert tunnel is not None and tunnel.stdout is not None
            for line in tunnel.stdout:
                clean = line.strip()
                if clean:
                    log_queue.append(clean)
                    url = tunnel_url_from_log(clean)
                    if url:
                        nonlocal_url[0] = url

        nonlocal_url: list[str | None] = [None]
        threading.Thread(target=collect_logs, daemon=True).start()
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline and not nonlocal_url[0]:
            if tunnel.poll() is not None:
                break
            time.sleep(0.2)
        tunnel_url = nonlocal_url[0]
        if not tunnel_url:
            print("ERROR: Cloudflare did not provide a temporary URL.")
            for line in log_queue[-12:]:
                print(f"  {line}")
            return 1

        hostname = tunnel_url.removeprefix("https://")
        environment = os.environ.copy()
        environment["FUSION_PUBLIC_ORIGIN"] = tunnel_url
        environment["FUSION_WEBAUTHN_ORIGIN"] = tunnel_url
        environment["FUSION_WEBAUTHN_RP_ID"] = hostname
        api = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=ROOT,
            env=environment,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        if not children.add(api):
            stop_process(api)
            print("ERROR: Could not attach the API to the close-on-exit process job.")
            return 2

        print("\nTemporary phone + laptop address:", tunnel_url, flush=True)
        print("Checking the local API and Cloudflare tunnel connection…", flush=True)
        local_health_error = wait_for_health(f"{LOCAL_URL}/health", 60, api)
        if local_health_error:
            print(f"ERROR: The local API did not become ready: {local_health_error}")
            return 1
        tunnel_error = wait_for_tunnel_connection(log_queue, 30, tunnel)
        if tunnel_error:
            print(f"ERROR: {tunnel_error}")
            for line in log_queue[-12:]:
                print(f"  {line}")
            return 1

        print("Checking public HTTPS access from this computer…", flush=True)
        public_health_error = wait_for_health(f"{tunnel_url}/health", 15, api)
        if public_health_error:
            print(
                "WARNING: This computer could not verify the public URL "
                f"({public_health_error}). The local API and Cloudflare connection are ready; "
                "the browser will open the URL so you can test access."
            )
        if api.poll() is not None:
            print("ERROR: The API process exited during startup.")
            return 1
        if tunnel.poll() is not None:
            print("ERROR: cloudflared exited during startup.")
            for line in log_queue[-12:]:
                print(f"  {line}")
            return 1

        print("\nPrototype ready. Open this URL on the laptop and scan its QR with your phone:")
        print(tunnel_url)
        print("The link is public while this window is open. Press Ctrl+C to end the demo and close the tunnel.")
        os.startfile(tunnel_url)  # type: ignore[attr-defined]
        while tunnel.poll() is None and api.poll() is None:
            time.sleep(0.5)
        return 0
    except KeyboardInterrupt:
        print("\nStopping the prototype and revoking the temporary tunnel…")
        return 0
    finally:
        stop_process(api)
        stop_process(tunnel)
        children.close()
        if tunnel_url:
            print("Local API and Cloudflare connector stopped; the temporary tunnel URL is no longer served.")


if __name__ == "__main__":
    raise SystemExit(main())
