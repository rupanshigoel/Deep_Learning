"""Execute a shell command on HPC via paramiko (same auth as hpc_upload.py).

Usage:
    python hpc/hpc_run.py "ls ~/scratch/col775_a2_hpc"
    python hpc/hpc_run.py --timeout 600 "bash ~/scratch/col775_a2_hpc/hpc/setup/install_vlm_deps.sh"
"""
from __future__ import annotations
import argparse, os, sys

# Force UTF-8 on stdout/stderr so we can stream non-ASCII remote output on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import paramiko

HOST = "hpc.iitd.ac.in"
USER = "mt1230785"
PASS = os.environ.get("HPC_PASS", "")
KEY = os.path.expanduser("~/.ssh/id_ed25519")


def connect():
    cli = paramiko.SSHClient()
    cli.load_system_host_keys()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        cli.connect(HOST, username=USER, key_filename=KEY,
                    look_for_keys=True, allow_agent=False, timeout=30)
    except (paramiko.AuthenticationException, paramiko.SSHException, FileNotFoundError):
        cli.connect(HOST, username=USER, password=PASS,
                    look_for_keys=False, allow_agent=False, timeout=30)
    return cli


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", help="Shell command to execute")
    ap.add_argument("--timeout", type=int, default=1800, help="Command timeout in seconds")
    args = ap.parse_args()

    cli = connect()
    try:
        tr = cli.get_transport()
        ch = tr.open_session()
        ch.settimeout(args.timeout)
        ch.exec_command(args.command)
        # Stream stdout/stderr
        while True:
            if ch.recv_ready():
                data = ch.recv(4096)
                if not data: break
                sys.stdout.write(data.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            if ch.recv_stderr_ready():
                data = ch.recv_stderr(4096)
                if not data: break
                sys.stderr.write(data.decode("utf-8", errors="replace"))
                sys.stderr.flush()
            if ch.exit_status_ready():
                while ch.recv_ready():
                    sys.stdout.write(ch.recv(4096).decode("utf-8", errors="replace"))
                while ch.recv_stderr_ready():
                    sys.stderr.write(ch.recv_stderr(4096).decode("utf-8", errors="replace"))
                break
        sys.stdout.flush(); sys.stderr.flush()
        rc = ch.recv_exit_status()
        sys.exit(rc)
    finally:
        cli.close()


if __name__ == "__main__":
    main()
