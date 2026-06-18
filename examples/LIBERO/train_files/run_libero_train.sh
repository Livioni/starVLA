#!/usr/bin/env bash

detect_nccl_socket_ifname() {
  for iface_path in /sys/class/net/*; do
    iface=${iface_path##*/}
    case "${iface}" in
      lo|docker*|br-*|veth*|virbr*|cvd-*) continue ;;
    esac
    state="unknown"
    if [ -r "${iface_path}/operstate" ]; then
      state=$(cat "${iface_path}/operstate")
    fi
    if [ "${state}" = "up" ] || [ "${state}" = "unknown" ]; then
      printf '%s\n' "${iface}"
      return 0
    fi
  done
}

if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
  detected_ifname=$(detect_nccl_socket_ifname)
  if [ -n "${detected_ifname}" ]; then
    export NCCL_SOCKET_IFNAME="${detected_ifname}"
    echo "Using NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
  else
    echo "Warning: could not auto-detect NCCL_SOCKET_IFNAME; leaving it unset" >&2
  fi
else
  echo "Using NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
fi

# Set NCCL_IB_HCA in the environment only when your cluster requires a
# specific InfiniBand device, for example: NCCL_IB_HCA=mlx5_2,mlx5_3 bash ...

# used for check save when communication
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=outputs
run_id=libero4in1_qwen3groot_debug
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
