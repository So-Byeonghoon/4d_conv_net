
GIT_DIR="/mnt/disk2/bhso/3DV-Action/"  # 3DV git root
DATA_DIR="/mnt/disk2/bhso/data/"  # "${DATA_DIR}/NTU_3seg_depthpoint" directory is needed
NUM_GPU=8

RESULT_DIR="${GIT_DIR}/results/4DConv_ntu120/"
mkdir -p ${RESULT_DIR}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --root_path "${DATA_DIR}/3DV_pointdata/NTU_voxelsize35_split5/" \
    --depth_path "${DATA_DIR}" \
    --save_root_dir "${RESULT_DIR}" \
    --ngpu ${NUM_GPU}

