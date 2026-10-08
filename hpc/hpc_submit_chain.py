"""Submit a sequence of PBS scripts to IITD HPC, chaining each via -W depend=afterok.

Usage:
    python hpc/hpc_submit_chain.py path/to/script1.pbs path/to/script2.pbs ...
The first script is submitted unconditionally. Each subsequent script is queued
with depend=afterok:<previous job id>. Prints job IDs.
"""
import os, sys, posixpath
import paramiko

HOST = "hpc.iitd.ac.in"
USER = "mt1230785"
PASS = os.environ.get("HPC_PASS", "")
KEY  = os.path.expanduser("~/.ssh/id_ed25519")
REMOTE_ROOT = "/home/maths/btech/mt1230785/scratch/col775_a2_hpc"


def connect():
    cli = paramiko.SSHClient()
    cli.load_system_host_keys()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        cli.connect(HOST, username=USER, key_filename=KEY,
                    look_for_keys=True, allow_agent=False, timeout=30)
    except Exception:
        cli.connect(HOST, username=USER, password=PASS,
                    look_for_keys=False, allow_agent=False, timeout=30)
    return cli


def run(cli, cmd):
    stdin, stdout, stderr = cli.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def submit(cli, remote_pbs, depend_on=None):
    parent = posixpath.dirname(remote_pbs)
    deps = f"-W depend=afterok:{depend_on}" if depend_on else ""
    cmd = f"cd {REMOTE_ROOT} && qsub {deps} {remote_pbs}"
    rc, out, err = run(cli, cmd)
    if rc != 0:
        raise RuntimeError(f"qsub failed (rc={rc}): {err or out}")
    jobid = out.strip().splitlines()[-1]
    return jobid


def main():
    scripts = sys.argv[1:]
    if not scripts:
        print(__doc__); sys.exit(1)

    cli = connect()
    print(f"connected to {HOST} as {USER}")
    prev = None
    for s in scripts:
        # Local path 'hpc/pbs/foo.pbs' maps to remote '<root>/hpc/foo.pbs'
        # (HPC layout has all PBS files directly under <root>/hpc/, not <root>/hpc/pbs/).
        s_norm = s.replace("\\", "/")
        if s_norm.startswith("/"):
            remote = s_norm
        else:
            base = posixpath.basename(s_norm)
            remote = posixpath.join(REMOTE_ROOT, "hpc", base)
        jobid = submit(cli, remote, depend_on=prev)
        print(f"submitted {s}  ->  {jobid}" + (f"  (afterok:{prev})" if prev else ""))
        prev = jobid
    cli.close()
    print("done.")


if __name__ == "__main__":
    main()
