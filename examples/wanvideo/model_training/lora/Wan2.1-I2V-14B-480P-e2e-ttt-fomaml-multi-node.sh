#!/usr/bin/env bash
# Multi-NODE (N nodes x 4 GPUs, DDP data-parallel) FOMAML E2E-TTT meta-training for
# Wan2.1-I2V-14B-480P: one video per GPU per step, meta-gradients on phi_0 all-reduced across
# every rank on every node. All training knobs live in the YAML config; pass extra CLI flags
# after the config path to override individual values.
#
#   sbatch examples/wanvideo/model_training/lora/Wan2.1-I2V-14B-480P-e2e-ttt-fomaml-multi-node.sbatch
#   sbatch ...-multi-node.sbatch <config.yaml> --learning_rate 2e-5
#
# This script MUST run inside a Slurm allocation (it reads $SLURM_JOB_NODELIST and fans out
# with srun).
#
# Same shape as the Wan2.2-TI2V-5B multi-node launchers; the differences that matter for a
# 16.4B DiT are all in the .sbatch (node count, host RAM, shared error-recycler arena) and in
# the config (`e2e_incremental_meta_backward: true`, without which an adaptive-length clip
# does not fit -- see the config header).
#
# NOTE: do not add --enable_model_cpu_offload here -- that path skips DDP wrapping entirely
# (see diffsynth/diffusion/runner.py) and is single-GPU only.
# NOTE: FOMAML only. The Reptile surrogate does not route through the DDP-wrapped DiT forward
# and would need --find_unused_parameters (or a manual grad all-reduce). Exact second-order
# MAML is a separate matter: it would work, but it pins one memorize graph per inner step on
# top of everything else, and at this model size that is not affordable at any chunk count.
set -eo pipefail

if [ -z "${SLURM_JOB_ID}" ] || [ -z "${SLURM_JOB_NODELIST}" ]; then
    echo "ERROR: this launcher requires a Slurm allocation (SLURM_JOB_NODELIST unset)." >&2
    exit 1
fi

CONFIG="${1:-examples/wanvideo/model_training/configs/Wan2.1-I2V-14B-480P-e2e-ttt-fomaml-data-v4.1.yaml}"

NUM_NODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
NUM_PROCESSES=$(( NUM_NODES * GPUS_PER_NODE ))

# ── Rendezvous endpoint ───────────────────────────────────────────────────────
# Every node must agree on one (ip, port). The head node is the first entry of the
# allocation's nodelist. Resolve it to an IP on the node itself: `hostname --ip-address`
# returns the address NCCL/gloo will actually bind, whereas resolving the hostname from
# the submitting shell can hand back a management-network address the compute fabric
# cannot route. Falls back to getent, then to the bare hostname.
HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
MASTER_ADDR="${MASTER_ADDR:-$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname --ip-address 2>/dev/null | awk '{print $1; exit}')}"
if [ -z "${MASTER_ADDR}" ]; then
    MASTER_ADDR="$(getent hosts "${HEAD_NODE}" | awk '{print $1; exit}')"
fi
: "${MASTER_ADDR:=${HEAD_NODE}}"

# Derive the port from the job id so two concurrent jobs on the same node never collide
# on the rendezvous socket. Range 20000-39999 avoids the ephemeral range.
MASTER_PORT="${MASTER_PORT:-$(( 20000 + SLURM_JOB_ID % 20000 ))}"

echo "[E2E-TTT multi-node] nodes=${NUM_NODES} gpus/node=${GPUS_PER_NODE} world=${NUM_PROCESSES}"
echo "[E2E-TTT multi-node] head=${HEAD_NODE} rendezvous=${MASTER_ADDR}:${MASTER_PORT}"
echo "[E2E-TTT multi-node] config=${CONFIG}"

# Newer Slurm no longer propagates --cpus-per-task to srun steps automatically; without
# this each task gets 1 CPU and the dataloader workers starve.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-64}"
# Silence the accelerate/torch thread-oversubscription warning and keep the per-rank
# CPU pool sane: 4 ranks share one node's cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( SRUN_CPUS_PER_TASK / GPUS_PER_NODE ))}"

LAUNCH_ARGS=( --config "${CONFIG}" "${@:2}" )

# One srun task per node; each task runs `accelerate launch`, which forks GPUS_PER_NODE
# local ranks. $SLURM_NODEID is expanded INSIDE the task (single-quoted heredoc-style
# body), so each node passes its own --machine_rank. The user's extra flags arrive as
# "$@" via the `bash -c '<body>' _ <args...>` convention.
srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
     --cpus-per-task="${SRUN_CPUS_PER_TASK}" \
     bash -c '
        echo "[node ${SLURM_NODEID}] host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"
        # Per-NODE error-recycler arena on node-local NVMe. Set here rather than in the
        # .sbatch because $TMPDIR is only defined inside the job step, and it must be
        # node-local: a shared-filesystem path would be a correctness bug (ranks on
        # different nodes mapping the same file), and /dev/shm pages are unreclaimable with
        # ~1 GiB of swap on a 377 GiB node, so a segment leaked by an OOM-kill costs the
        # NEXT job pinned RAM that slurm cannot see.
        export E2E_TTT_SHARED_BUFFER_DIR="${E2E_TTT_SHARED_BUFFER_DIR:-${TMPDIR:-/tmp}/e2e_ttt_buffers_${SLURM_JOB_ID}}"
        mkdir -p "${E2E_TTT_SHARED_BUFFER_DIR}"
        echo "[node ${SLURM_NODEID}] shared buffer arena: ${E2E_TTT_SHARED_BUFFER_DIR}"
        # Invoke accelerate as a module of the ACTIVE python (conda `diffsynth` env), so it
        # does not fall back to a `~/.local/bin/accelerate` tied to a different interpreter
        # that cannot import `diffsynth`.
        exec python -m accelerate.commands.launch \
            --multi_gpu \
            --num_machines '"${NUM_NODES}"' \
            --num_processes '"${NUM_PROCESSES}"' \
            --machine_rank "${SLURM_NODEID}" \
            --main_process_ip '"${MASTER_ADDR}"' \
            --main_process_port '"${MASTER_PORT}"' \
            --rdzv_backend static \
            --same_network \
            examples/wanvideo/model_training/train_e2e_ttt.py "$@"
     ' _ "${LAUNCH_ARGS[@]}"
