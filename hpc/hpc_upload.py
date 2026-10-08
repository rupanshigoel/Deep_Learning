"""Recursively upload a local directory to HPC over SFTP, mirroring structure.

Usage:
    python hpc_upload.py <local_dir> <remote_dir>
"""
import os, sys, stat, posixpath
import paramiko

HOST = "hpc.iitd.ac.in"
USER = "mt1230785"
PASS = os.environ.get("HPC_PASS", "")
KEY  = os.path.expanduser("~/.ssh/id_ed25519")


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


def mkdirs(sftp, path):
    if path in ("/", "", "."):
        return
    try:
        sftp.stat(path)
        return
    except IOError:
        pass
    parent = posixpath.dirname(path)
    if parent and parent != path:
        mkdirs(sftp, parent)
    try:
        sftp.mkdir(path)
    except IOError as e:
        # If mkdir truly failed (not "already exists"), re-stat to decide.
        try:
            sftp.stat(path)
        except IOError:
            raise RuntimeError(f"mkdir failed and path missing: {path}: {e}")


def upload_tree(sftp, local_dir, remote_dir):
    local_dir = os.path.abspath(local_dir)
    mkdirs(sftp, remote_dir)
    n_files = n_skip = 0
    for root, dirs, files in os.walk(local_dir):
        # Skip cache + venv-like dirs
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", ".venv", "node_modules")]
        rel = os.path.relpath(root, local_dir).replace(os.sep, "/")
        rdir = remote_dir if rel in (".", "") else posixpath.join(remote_dir, rel)
        mkdirs(sftp, rdir)
        for fn in files:
            if fn.endswith(".pyc"):
                continue
            lpath = os.path.join(root, fn)
            rpath = posixpath.join(rdir, fn)
            try:
                lst = os.stat(lpath)
                rst = sftp.stat(rpath)
                if lst.st_size == rst.st_size:
                    # Same size — skip to be cheap. (Won't catch silent edits with same size.)
                    n_skip += 1
                    continue
            except IOError:
                pass
            print(f"[put] {rel}/{fn} ({lst.st_size} bytes)" if rel != "." else f"[put] {fn} ({lst.st_size} bytes)")
            sftp.put(lpath, rpath, confirm=True)
            # Double-check: explicit stat to detect silent paramiko put failures.
            try:
                rst_after = sftp.stat(rpath)
                if rst_after.st_size != lst.st_size:
                    raise RuntimeError(f"size mismatch after put: local={lst.st_size} remote={rst_after.st_size}")
            except IOError as e:
                raise RuntimeError(f"verify failed for {rpath}: {e}")
            # Preserve executable bit for shell scripts
            if fn.endswith((".sh", ".pbs")) or os.access(lpath, os.X_OK):
                sftp.chmod(rpath, 0o755)
            n_files += 1
    print(f"[done] uploaded {n_files} files, skipped {n_skip} (same size)")


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    local_dir, remote_dir = sys.argv[1], sys.argv[2]
    cli = connect()
    try:
        sftp = cli.open_sftp()
        upload_tree(sftp, local_dir, remote_dir)
        sftp.close()
    finally:
        cli.close()


if __name__ == "__main__":
    main()
