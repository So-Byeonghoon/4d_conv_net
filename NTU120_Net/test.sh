
GIT_DIR="/mnt/disk2/bhso/3DV-Action/"  # 3DV git root
DATA_DIR="/mnt/disk2/bhso/data/"  # "${DATA_DIR}/NTU_3seg_depthpoint" directory is needed
NUM_GPU=8

MODEL="PointNet++"
#RESULT_DIR="${GIT_DIR}/results/4DConv_ntu120/"
RESULT_DIR="${GIT_DIR}/results/3DV_motion_only/"

OPTIONS="--skip-appearance"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py \
    --root_path "${DATA_DIR}/3DV_pointdata/NTU_voxelsize35_split5/" \
    --depth_path "${DATA_DIR}" \
    --save_root_dir "${RESULT_DIR}" \
    --model "${MODEL}" \
    --ngpu ${NUM_GPU} \
    --nepoch 50 \
    --dataset "ntu120" ${OPTIONS} |& tee "${RESULT_DIR}/test_$(date +"%F_%H:%M:%S").log"

