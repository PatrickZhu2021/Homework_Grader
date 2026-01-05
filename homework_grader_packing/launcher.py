# launcher.py - Production launcher for Streamlit app packaged with PyInstaller (Windows-friendly)
import os
import sys
import time
import socket
import traceback
from pathlib import Path
import multiprocessing
import threading
import ctypes
from ctypes import wintypes


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pid_is_running(pid: int) -> bool:
    """Windows-friendly PID liveness check without extra dependencies."""
    if pid <= 0:
        return False

    # Try os.kill(pid, 0) first
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        pass

    # Windows: OpenProcess + GetExitCodeProcess
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        OpenProcess = kernel32.OpenProcess
        OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        OpenProcess.restype = wintypes.HANDLE

        GetExitCodeProcess = kernel32.GetExitCodeProcess
        GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        GetExitCodeProcess.restype = wintypes.BOOL

        CloseHandle = kernel32.CloseHandle
        CloseHandle.argtypes = [wintypes.HANDLE]
        CloseHandle.restype = wintypes.BOOL

        h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        try:
            code = wintypes.DWORD(0)
            ok = GetExitCodeProcess(h, ctypes.byref(code))
            if not ok:
                return False
            return code.value == STILL_ACTIVE
        finally:
            CloseHandle(h)

    return False


def _acquire_lock(lock_path: Path, log) -> bool:
    """
    PID-aware single-instance lock.
    If lock exists and PID is running -> exit.
    If lock exists but PID is dead -> remove stale lock -> acquire new lock.
    """
    if lock_path.exists():
        old_pid = -1
        try:
            txt = lock_path.read_text(encoding="utf-8").strip()
            old_pid = int(txt.splitlines()[0]) if txt else -1
        except Exception:
            old_pid = -1

        if _pid_is_running(old_pid):
            log(f"[launcher] lock exists and pid {old_pid} is running; exit.")
            return False

        # stale lock
        try:
            lock_path.unlink()
            log(f"[launcher] stale lock removed (pid {old_pid}).")
        except Exception as e:
            log(f"[launcher] failed to remove stale lock: {e!r}; exit.")
            return False

    # acquire new lock
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
        os.close(fd)
        log(f"[launcher] lock acquired pid={os.getpid()}")
        return True
    except FileExistsError:
        log("[launcher] lock race (FileExistsError); exit.")
        return False


def _release_lock(lock_path: Path, log) -> None:
    try:
        lock_path.unlink()
        log("[launcher] lock released")
    except Exception:
        pass


def _find_free_port(preferred=8501) -> int:
    for p in [preferred, 8502, 8503, 8504, 8505, 8506]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("No free port found")


def _wait_port(host: str, port: int, timeout_sec: float = 45.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _candidate_roots(exe_dir: Path) -> list[Path]:
    """
    Possible roots containing apps/ and core/:
    - onefile: sys._MEIPASS
    - onedir (newer): exe_dir/_internal
    - onedir (older): exe_dir
    - extra: parent/_internal and parent
    """
    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))

    roots += [
        exe_dir / "_internal",
        exe_dir,
        exe_dir.parent / "_internal",
        exe_dir.parent,
    ]

    uniq: list[Path] = []
    seen = set()
    for r in roots:
        try:
            rp = r.resolve()
        except Exception:
            rp = r
        key = str(rp).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(rp)
    return uniq


def _find_app_script(exe_dir: Path, log) -> Path | None:
    candidates = [root / "apps" / "batch_app.py" for root in _candidate_roots(exe_dir)]
    log("[launcher] app candidates: " + " | ".join(str(c) for c in candidates))
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    multiprocessing.freeze_support()

    exe_path = Path(sys.argv[0]).resolve()
    exe_dir = exe_path.parent
    os.chdir(exe_dir)

    logs_dir = exe_dir / "logs"
    _mkdir(logs_dir)

    launcher_log = logs_dir / "launcher.log"
    stdout_log = logs_dir / "stdout.log"
    stderr_log = logs_dir / "stderr.log"
    lock_path = exe_dir / ".app.lock"

    def log(msg: str) -> None:
        with launcher_log.open("a", encoding="utf-8") as f:
            f.write(f"{_ts()} {msg}\n")

    # Redirect stdout/stderr so errors don't disappear
    sys.stdout = stdout_log.open("a", encoding="utf-8")
    sys.stderr = stderr_log.open("a", encoding="utf-8")

    log(f"[launcher] exe={exe_path}")
    log(f"[launcher] exe_dir={exe_dir}")
    log(f"[launcher] sys.executable={sys.executable}")
    log(f"[launcher] sys._MEIPASS={getattr(sys, '_MEIPASS', None)}")

    if not _acquire_lock(lock_path, log):
        return

    try:
        # Find Streamlit app script
        app_script = _find_app_script(exe_dir, log)
        if not app_script:
            log("[launcher] FATAL: apps/batch_app.py not found.")
            print("ERROR: apps/batch_app.py not found. See logs:", launcher_log)
            time.sleep(5)
            return

        port = _find_free_port(8501)
        url = f"http://127.0.0.1:{port}"
        log(f"[launcher] port={port}")
        log(f"[launcher] url={url}")
        log(f"[launcher] app_script={app_script}")

        # Force-disable Streamlit development mode to avoid server.port conflict
        os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
        os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

        # Prepare Streamlit CLI args (in-process)
        sys.argv = [
            "streamlit",
            "run",
            str(app_script),
            "--server.address", "127.0.0.1",
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.runOnSave", "false",
            "--server.fileWatcherType", "none",
            "--browser.gatherUsageStats", "false",
        ]

        # Open browser once (in-memory flag, no persistent lock files)
        opened = {"done": False}

        def open_browser_once():
            try:
                if opened["done"]:
                    return
                if _wait_port("127.0.0.1", port, timeout_sec=45):
                    opened["done"] = True
                    import webbrowser
                    webbrowser.open(url)
                    log("[launcher] browser opened")
                else:
                    log("[launcher] server not ready in time; open manually " + url)
                    print("Server not ready yet. Open manually:", url)
            except Exception as e:
                log(f"[launcher] open_browser_once error: {e!r}")

        threading.Thread(target=open_browser_once, daemon=True).start()

        log("[launcher] starting streamlit (in-process)")
        try:
            from streamlit.web import cli as stcli
            stcli.main()
            log("[launcher] streamlit returned normally")
        except SystemExit as e:
            log(f"[launcher] streamlit SystemExit code={e.code!r}")
            raise
        except Exception as e:
            log(f"[launcher] streamlit exception: {e!r}")
            log(traceback.format_exc())
            raise

    except Exception as e:
        log(f"[launcher] FATAL: {e!r}")
        log(traceback.format_exc())
        print("FATAL:", repr(e))
        print("See logs:", launcher_log, stderr_log)
        time.sleep(5)
    finally:
        _release_lock(lock_path, log)


if __name__ == "__main__":
    main()
