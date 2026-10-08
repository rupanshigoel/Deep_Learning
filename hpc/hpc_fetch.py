"""Fetch a single file from HPC via paramiko (same auth as hpc_upload.py).

Usage:
    python hpc/hpc_fetch.py <remote_path> <local_path>
"""
from __future__ import annotations
import os, sys
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
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    remote, local = sys.argv[1], sys.argv[2]
    cli = connect()
    try:
        sftp = cli.open_sftp()
        os.makedirs(os.path.dirname(os.path.abspath(local)) or ".", exist_ok=True)
        sftp.get(remote, local)
        sz = os.path.getsize(local)
        print(f"[fetched] {remote} -> {local} ({sz} bytes)")
        sftp.close()
    finally:
        cli.close()


if __name__ == "__main__":
    main()
