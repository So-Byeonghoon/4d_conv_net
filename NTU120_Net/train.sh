
GIT_DIR="/mnt/disk2/bhso/3DV-Action/"  # 3DV git root
DATA_DIR="/mnt/disk2/bhso/data/"  # depth_path: "${DATA_DIR}/NTU_3seg_depthpoint" directory is needed
NUM_GPU=1

MODEL="3DConv_base"
RESULT_DIR="${GIT_DIR}/results/4DConv_ntu120/"
#MODEL="PointNet++"
#RESULT_DIR="${GIT_DIR}/results/3DV_motion_only/"
mkdir -p ${RESULT_DIR}

OPTIONS="--skip-appearance --batchSize 16"

#    --root_path "${DATA_DIR}/3DV_pointdata/NTU_voxelsize35_split5/" \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
CUDA_VISIBLE_DEVICES=0 python train.py \
    --root_path "${DATA_DIR}/3DV_pointdata/NTU_voxelsize35_split5_without_sampling/" \
    --depth_path "${DATA_DIR}" \
    --save_root_dir "${RESULT_DIR}" \
    --model "${MODEL}" \
    --cross-subject \
    --nepoch 80 \
    --ngpu ${NUM_GPU} ${OPTIONS} |& tee ${RESULT_DIR}/train_$(date +"%F_%H:%M:%S").log

