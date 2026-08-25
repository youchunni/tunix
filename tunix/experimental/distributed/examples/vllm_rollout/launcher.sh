#!/usr/bin/bash
#
# Usage:
#   To run interactively:
#     ./launcher.sh
#     ./launcher.sh --local
#
#   To run a role:
#     ./launcher.sh --command=orchestrator
#     ./launcher.sh --command=rollout
#     ./launcher.sh --command=rollout --worker_id=1
#
#   To run a role with specific docker image:
#     ./launcher.sh --command=orchestrator --image=my_awesome_image
#
#   To run a role in local mode:
#     ./launcher.sh --command=orchestrator --local

# Flag set by --local to run processes locally instead of Kubernetes.
LOCAL_MODE=false
# Command to execute ('start', 'stop', 'orchestrator' or 'rollout').
COMMAND=""
# For rollout worker. If all, start all workers, otherwise, start just one with this id (e.g. --id=1).
WORKER_ID="all"
# Docker image URI used for worker containers on Kubernetes.
TUNIX_IMAGE="us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/tunix_base_image:latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      LOCAL_MODE=true
      shift
      ;;
    --command)
      COMMAND="$2"
      shift 2
      ;;
    --command=*)
      COMMAND="${1#*=}"
      shift
      ;;
    --worker_id)
      WORKER_ID="$2"
      shift 2
      ;;
    --worker_id=*)
      WORKER_ID="${1#*=}"
      shift
      ;;
    --image)
      TUNIX_IMAGE="$2"
      shift 2
      ;;
    --image=*)
      TUNIX_IMAGE="${1#*=}"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# Locate repo root path by traversing upwards until finding tunix/experimental/distributed/runtime/main.py
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$SCRIPT_DIR"
while [[ "$REPO_ROOT" != "/" && ! -f "$REPO_ROOT/tunix/experimental/distributed/runtime/main.py" ]]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
if [[ ! -f "$REPO_ROOT/tunix/experimental/distributed/runtime/main.py" ]]; then
  echo "Error: Could not locate repo root containing tunix/experimental/distributed/runtime/main.py." >&2
  exit 1
fi

# Configures kubectl to connect to the required GKE cluster and namespace.
enter_kube_context() {
  PROJECT="cloud-tpu-multipod-dev"
  REGION="us-central1"
  ZONE="us-central1-a"
  CLUSTER="rl-scaffolding"

  export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config.$PROJECT.$REGION.$CLUSTER}"
  if ! [ -f "$KUBECONFIG" ] || ! kubectl get namespaces &>/dev/null; then
    gcloud container clusters get-credentials $CLUSTER --region=$REGION --project=$PROJECT --dns-endpoint &>/dev/null || { echo "gcloud get-credentials failed"; exit 1; }
    kubectl config use-context "gke_${PROJECT}_${REGION}_${CLUSTER}" >/dev/null || { echo "kubectl use-context failed"; exit 1; }
  fi
  kubectl config set-context --current --namespace=default >/dev/null || { echo "kubectl set-context failed"; exit 1; }
}

# Launches the orchestrator process locally or on Kubernetes (GKE).
start_orchestrator() {
  if [[ "$LOCAL_MODE" == "true" ]]; then
    cd "$REPO_ROOT"
    python -m tunix.experimental.distributed.runtime.main \
      --discovery_id=orchestrator \
      --discovery_port=12345 \
      --process_main=tunix.experimental.distributed.examples.vllm_rollout.orchestrator.main \
      --parallelism=1
  else
    cd "$REPO_ROOT"
    python tunix/experimental/distributed/deployment/yaml_generator.py \
      tunix/experimental/distributed/deployment/yamls/jobset.cpu.yaml \
      --jobset_name=orchestrator \
      --cpu_machine=n2-standard-64 \
      --worker_container_image="$TUNIX_IMAGE" \
      --worker_container_port=12345 \
      --worker_startup_command="python -m tunix.experimental.distributed.runtime.main \
        --discovery_id=orchestrator \
        --discovery_port=12345 \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.distributed.examples.vllm_rollout.orchestrator.main \
        --parallelism=256" \
      | kubectl apply -f -
  fi
}

stop_orchestrator() {
  if [[ "$LOCAL_MODE" != "true" ]]; then
    kubectl delete jobset orchestrator
  fi
}

start_rollout() {
  if [[ "$LOCAL_MODE" == "true" ]]; then
    cd "$REPO_ROOT"
    SKIP_JAX_PRECOMPILE=1 \
    $PYTHON_EXEC -m tunix.experimental.distributed.runtime.main \
      --discovery_addrs=orchestrator:12345 \
      --process_main=tunix.experimental.distributed.examples.vllm_rollout.rollout.main \
      --worker_id=rollout-0 \
      --service_port=11111
  else
    cd "$REPO_ROOT"
    if [[ "$WORKER_ID" == "all" ]]; then
      for ((i=0; i<=3; i++)); do
        python tunix/experimental/distributed/deployment/yaml_generator.py \
          tunix/experimental/distributed/deployment/yamls/leaderworkerset.ray.tpu.yaml \
          --jobset_name=rollout-$i \
          --tpu_slice=tpuv5e:4x4 \
          --worker_container_image="$TUNIX_IMAGE" \
          --worker_container_port=$((10000+i)) \
          --worker_startup_command=" \
            MODEL_IMPL_TYPE=vllm \
            SKIP_JAX_PRECOMPILE=1 \
            python -m tunix.experimental.distributed.runtime.main \
            --discovery_addrs=orchestrator:12345 \
            --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
            --process_main=tunix.experimental.distributed.examples.vllm_rollout.rollout.main \
            --tensor_parallel_size=16 \
            --worker_id=rollout-$i \
            --service_port=$((10000+i))" \
          | kubectl apply -f -
      done
    else
      python tunix/experimental/distributed/deployment/yaml_generator.py \
        tunix/experimental/distributed/deployment/yamls/leaderworkerset.ray.tpu.yaml \
        --jobset_name="rollout-${WORKER_ID}" \
        --tpu_slice=tpuv5e:4x4 \
        --worker_container_image="$TUNIX_IMAGE" \
        --worker_container_port=11111 \
        --worker_startup_command=" \
            MODEL_IMPL_TYPE=vllm \
            SKIP_JAX_PRECOMPILE=1 \
            python -m tunix.experimental.distributed.runtime.main \
            --discovery_addrs=orchestrator:12345 \
            --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
            --process_main=tunix.experimental.distributed.examples.vllm_rollout.rollout.main \
            --tensor_parallel_size=16 \
            --worker_id=rollout-${WORKER_ID} \
            --service_port=11111" \
        | kubectl apply -f -
    fi
  fi
}

stop_rollout() {
  if [[ "$LOCAL_MODE" != "true" ]]; then
    if [[ "$WORKER_ID" == "all" ]]; then
      for ((i=0; i<=3; i++)); do
        kubectl delete leaderworkerset rollout-$i
      done
    else
      kubectl delete leaderworkerset "rollout-${WORKER_ID}"
    fi
  fi
}

if [[ "$LOCAL_MODE" == "false" ]]; then
  enter_kube_context
fi

if [[ "$COMMAND" == "start" ]]; then
  stop_orchestrator
  stop_rollout
  start_orchestrator
  start_rollout
elif [[ "$COMMAND" == "stop" ]]; then
  stop_orchestrator
  stop_rollout
elif [[ "$COMMAND" == "orchestrator" ]]; then
  stop_orchestrator; start_orchestrator
elif [[ "$COMMAND" == "rollout" ]]; then
  stop_rollout; start_rollout
else
  echo "Error: Invalid command '$COMMAND'. Available commands: 'start', 'stop', 'orchestrator', 'rollout'."
  exit 1
fi
