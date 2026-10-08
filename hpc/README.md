# hpc/ — submission & setup for IITD HPC

All the PBS scripts and setup bits that are specific to `hpc.iitd.ac.in`. Orthogonal to the code in `../col775_a2/` — the Python package is HPC-agnostic; only this directory knows about the cluster.

## Contents

- [`hpc_upload.py`](hpc_upload.py) — SFTP sync (size-based skip). Positional args: `<local_dir> <remote_dir>`. Gitignored because it has a password fallback.
- [`pbs/`](pbs/) — PBS Pro submission scripts, one per runnable stage. Every script activates `col775_t25` and prints `torch.__version__` + `cuda_is_available()` in its stdout header.
- [`setup/`](setup/) — one-shot bootstrap scripts for a fresh HPC install.

## PBS scripts map

| Part | Script | Walltime | Queue | Purpose |
|---|---|---|---|---|
| A | `submit_test.pbs` | 1 h | standard | CLIP 2-epoch smoke test |
| A | `submit_test_dino.pbs` | 1 h | standard | DINO 2-epoch smoke test |
| A | `submit_clip.pbs` | 6 h | standard | CLIP 100-epoch training |
| A | `submit_dino.pbs` | 4 h | high | DINO 100-epoch training (resumable) |
| Aa | `submit_probe.pbs` | 4 h | high | extract_embeddings → linear_probe → tsne_plot → retrieval |
| C | `submit_test_vae.pbs` | 1 h | standard | VAE smoke test |
| C | `submit_vae.pbs` | 24 h | standard | VAE 100-epoch training |
| C | `submit_latent_stats.pbs` | 1 h | standard | dump mean/std of train latents |
| C | `submit_ldm.pbs` | 24 h | standard | LDM 150-epoch training (resumable) |
| C | `submit_vae_fid.pbs` | 2 h | standard | VAE recon FID on 10K val |
| C | `submit_ldm_fid.pbs` | 6 h | standard | LDM sample 10K + FID |

## Common commands

Sync code + PBS to HPC, then submit:

```
python hpc/hpc_upload.py col775_a2 /home/maths/btech/mt1230785/scratch/col775_a2_hpc/col775_a2
python hpc/hpc_upload.py hpc/pbs  /home/maths/btech/mt1230785/scratch/col775_a2_hpc/hpc
ssh hpc.iitd.ac.in 'cd $HOME/scratch/col775_a2_hpc && qsub hpc/submit_<stage>.pbs'
```

Chain a dependent job:

```
ssh hpc.iitd.ac.in 'cd $HOME/scratch/col775_a2_hpc && qsub -W depend=afterok:<PREV_ID>.pbshpc hpc/submit_<next>.pbs'
```

Inspect state:

```
ssh hpc.iitd.ac.in 'qstat -u mt1230785'
ssh hpc.iitd.ac.in 'qstat -f <jobid>.pbshpc | grep -E "estimated|exec_vnode|comment|job_state"'
```

Pull results back:

```
# Logs
python hpc/hpc_upload.py logs_pull /home/maths/btech/mt1230785/scratch/col775_a2_hpc/logs   # flip for SFTP down

# Note: hpc_upload.py is PUSH only. For PULL, use sftp/rsync or pull-specific helper.
```

## Env setup (first time only)

```
ssh hpc.iitd.ac.in 'bash $HOME/scratch/col775_a2_hpc/setup/setup_hpc.sh'          # conda env + dataset symlinks
python hpc/hpc_upload.py torch25_wheels  /home/maths/btech/mt1230785/torch25_wheels
ssh hpc.iitd.ac.in 'bash $HOME/scratch/col775_a2_hpc/setup/build_torch25_env.sh'  # torch 2.5.0+cu118
ssh hpc.iitd.ac.in 'bash $HOME/scratch/col775_a2_hpc/setup/install_partc_deps.sh' # transformers + clean-fid (offline, uses archive/hf_cache + archive/cleanfid_cache — upload those first)
```

## Queue cheat sheet

- `standard` — default, 1× cost. `Max running = 10`, `Max queued = 10` per user.
- `high` — 5× cost. Better priority, shorter wait (~1–3 h vs 3–6 h on standard). Use only when deadline-critical or for small jobs.
- `low` — cheaper, usually saturated. Not currently used.
- `scai_q`, `top`, `admin`, `workshop` — not accessible to btech/maths users.

Each icelake A100 node has 40 GB GPU memory. Jobs must include `centos=icelake` and `ngpus=1` in the resource select.

## See also

- [`../FOLDER.md`](../FOLDER.md) — overall project layout.
- Memory file `hpc_setup.md` (in Claude user memory) — credentials, proxy tricks, queue quirks.
- Memory file `col775_t25_env.md` — how `col775_t25` was built, cu11 dependency pins, the python-binary truncation gotcha.
