#!/bin/bash
# Launch one data-parallel rank per node inside a Fuzzball multinode job, with diagnostics.
# Env in:  MN_NODES, MN_VARIANT (ip|c10d), PY, TRAIN_ARGS. Prints DIAG lines for post-mortem.
set -u
mkdir -p /tmp/home
SELF=$(hostname); FQDN=$(hostname -f 2>/dev/null || echo "$SELF")
HOSTS=$(echo "${MULTINODE_HOSTLIST_NOSLOTS:-$SELF}" | tr ',' '\n' | sort)
echo "DIAG self=$SELF fqdn=$FQDN hostlist=$(echo $HOSTS) variant=$MN_VARIANT"
echo "DIAG env: $(env | grep -iE 'MULTINODE|RANK|NODE' | grep -viE 'TOKEN|SECRET|KEY' | tr '\n' ' ')"
for h in $HOSTS; do echo "DIAG resolve $h -> $(getent hosts $h | awk '{print $1}' | tr '\n' ' ')"; done
echo "DIAG ifaces: $(ls /sys/class/net | tr '\n' ' ') ib=$(ls /dev/infiniband 2>/dev/null | tr '\n' ' ')"
RANK_IDX=$(echo "$HOSTS" | grep -nx -e "$SELF" -e "$FQDN" | head -1 | cut -d: -f1)
[ -z "$RANK_IDX" ] && RANK_IDX=$(echo "$HOSTS" | grep -n "^${SELF%%.*}" | head -1 | cut -d: -f1)
NODE_RANK=${MN_NODE_RANK:-$((RANK_IDX - 1))}; MASTER=${MN_MASTER:-$(echo "$HOSTS" | head -1)}
# Fuzzball's generic multinode runs this script on the head node only (see /multinode/generic-helper).
# The head starts every other rank with: "$MULTINODE_SSH_WRAPPER" "$host" "<one shell string>".
if [ -z "${MN_NODE_RANK:-}" ] && [ -n "${MULTINODE_SSH_WRAPPER:-}" ]; then
  HEAD_IP=${MULTINODE_NODE_IP:-$(getent hosts "$SELF" | awk '{print $1}' | head -1)}
  i=0
  for h in $HOSTS; do
    [ "$h" = "$SELF" ] || [ "$h" = "$FQDN" ] && continue
    i=$((i + 1))
    echo "DIAG head ($SELF, $HEAD_IP) starting rank $i on $h"
    "$MULTINODE_SSH_WRAPPER" "$h" "MN_NODE_RANK=$i MN_MASTER=$SELF MN_MASTER_IP=$HEAD_IP MN_NODES=$MN_NODES MN_VARIANT=$MN_VARIANT PY=$PY TRAIN_ARGS='$TRAIN_ARGS' MULTINODE_HOSTLIST_NOSLOTS=$MULTINODE_HOSTLIST_NOSLOTS HOME=/tmp/home bash /opt/multinode_launch.sh" > /tmp/rank$i.log 2>&1 &
  done
  NODE_RANK=0; MASTER=$SELF; MN_MASTER_IP=$HEAD_IP
  trap 'for f in /tmp/rank*.log; do echo "== worker log $f"; tail -60 "$f"; done' EXIT
fi
MASTER_IP=${MN_MASTER_IP:-$(getent hosts "$MASTER" | awk '{print $1}' | head -1)}; MASTER_IP=${MASTER_IP:-$MASTER}
echo "DIAG node_rank=$NODE_RANK master=$MASTER master_ip=$MASTER_IP"
# Plain TCP check between nodes before torch: rank 0 listens on 29511, others connect.
$PY - <<PYEOF
import socket, time, sys
r, ip = $NODE_RANK, "$MASTER_IP"
try:
    if r == 0:
        s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("0.0.0.0", 29511)); s.listen(8)
        s.settimeout(120); c, a = s.accept(); print(f"DIAG tcp_check rank0 accepted from {a}", flush=True)
    else:
        for i in range(60):
            try:
                c = socket.create_connection((ip, 29511), timeout=5); print(f"DIAG tcp_check connected to {ip}:29511 local={c.getsockname()}", flush=True); break
            except Exception as e: time.sleep(2)
        else: print(f"DIAG tcp_check FAILED to reach {ip}:29511", flush=True)
except Exception as e:
    print(f"DIAG tcp_check error {e}", flush=True)
PYEOF
[ "$MN_VARIANT" = "diag" ] && { echo "DIAG done (diagnostics only)"; exit 0; }
export NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_NCCL_ASYNC_ERROR_HANDLING=1
if [ "$MN_VARIANT" = "c10d" ]; then
  export NCCL_SOCKET_IFNAME=^lo,docker
  RDZV="--rdzv-backend c10d --rdzv-endpoint $MASTER_IP:29500 --rdzv-id dp-$$-job"
else
  export NCCL_IB_DISABLE=1
  RDZV="--node-rank $NODE_RANK --master-addr $MASTER_IP --master-port 29500"
fi
echo "DIAG launching: torch.distributed.run --nnodes $MN_NODES $RDZV"
timeout 1200 $PY -m torch.distributed.run --nnodes "$MN_NODES" --nproc-per-node 1 $RDZV $TRAIN_ARGS
rc=$?; [ $rc -eq 124 ] && echo "DIAG TIMEOUT after 1200 s (rendezvous or collective hung)"; exit $rc
