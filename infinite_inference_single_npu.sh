#!/bin/bash
# 1. 昇腾 NPU 专用环境变量
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# LiveAvatar 的原生编译和 FP8 是给 NVIDIA Triton 写的，在 NPU 上必须关掉
export ENABLE_COMPILE=true
export ENABLE_FP8=False 
export LIVEAVATAR_DISABLE_NPU_FUSED_ATTN=true

FFMPEG_DIR=$(python3 -c "import os, imageio_ffmpeg; print(os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()))")
export PATH="${FFMPEG_DIR}:${PATH}"

# 2. LiveAvatar 兼容的启动命令
PYTHONUNBUFFERED=1 TORCHDYNAMO_VERBOSE=1 TORCH_LOGS="recompiles,recompiles_verbose,guards,graph_breaks,dynamic,aot_graphs" TORCHAIR_LOG_LEVEL=DEBUG 
torchrun --nproc_per_node=1 --master_port=29101 minimal_inference/s2v_streaming_interact.py \
    --task s2v-1.3B \
    --ckpt_dir ./speed_test_1_3B_0327/FidoAvatar_1.3B_0327 \
    --size "416*240" \
    --image ./speed_test_1_3B_0327/AIjiabin/6.jpg \
    --audio ./speed_test_1_3B_0327/AIjiabin/jianggao_v2_yuanshi_0120_49_1_merged.wav \
    --prompt "A woman with glasses, wearing a yellow cloth and black bottoms, stands in front of a stone wall. She is speaking, with her hands frequently making various gestures: lifting and spreading, moving in front of her body, clenching then opening, spreading to the sides, etc. Her head and body have slight movements as she speaks, and her expression changes with the content." \
    --save_dir ./debug/ \
    --sample_steps 4 \
    --start_from_ref \
    --sample_guide_scale=1 \
    --sample_solver=euler \
    --infer_frames=16 \
    --offload_model false \
    --init_on_cpu false