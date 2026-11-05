export SWANLAB_API_KEY='PIArEMJ6iFlXUxb2lIgJj'
export TARGET_MODEL="/home/jovyan/LMM/lmm_model/Qwen3.5-4B"
# export TRAIN_DATA="/home/jovyan/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm/data/finevision_qwen35_sglang_20000/train.jsonl"
export TRAIN_DATA="/home/jovyan/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm/data/finevision_qwen35_10w_img1024_relaxed/train.jsonl"
export CONFIG="$PWD/scripts/config/vlm/sft_student_aligned.yaml"
export ALGORITHM=dlite

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export REPORT_TO=swanlab
export SAVE_TRAINING_STATE=1

RUN_NAME="dlite-stage-sft-6ep-gb32-b8-data100k$(date +%Y%m%d_%H%M%S)"
export OUTPUT_DIR="$DLITE_WORKSPACE/runs/$RUN_NAME"
unset MAX_STEPS MAX_SAMPLES

mkdir -p "$OUTPUT_DIR"
set -o pipefail
bash scripts/dlite/run_training_vlm.sh \
  --swanlab-project qwen35-dlite-vlm \
  --swanlab-name "$RUN_NAME" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
