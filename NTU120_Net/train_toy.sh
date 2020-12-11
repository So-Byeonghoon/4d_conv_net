
GIT_DIR="/mnt/disk2/bhso/3DV-Action/"  # 3DV git root
DATA_DIR="/mnt/disk2/bhso/data/"  # depth_path: "${DATA_DIR}/NTU_3seg_depthpoint" directory is needed
NUM_GPU=8
NUM_GPU=1

MODEL="3DConv_base"
RESULT_DIR="${GIT_DIR}/results/4DConv_ntu120_toy/"
#MODEL="PointNet++"
#RESULT_DIR="${GIT_DIR}/results/3DV_motion_only/"
mkdir -p ${RESULT_DIR}

OPTIONS="--toy --batchSize 8"
#    --root_path "${DATA_DIR}/3DV_pointdata/NTU_voxelsize35_split5/" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --root_path "${DATA_DIR}/3DV_pointdata/NTU_voxelsize35_split5_without_sampling/" \
    --depth_path "${DATA_DIR}" \
    --save_root_dir "${RESULT_DIR}" \
    --model "${MODEL}" \
    --ngpu ${NUM_GPU} ${OPTIONS} |& tee ${RESULT_DIR}/train.log

