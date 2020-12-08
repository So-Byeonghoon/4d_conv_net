
GIT_DIR="/mnt/disk2/bhso/3DV-Action/"  # 3DV git root
DATA_DIR="/mnt/disk2/bhso/data/"  # "${DATA_DIR}/NTU_3seg_depthpoint" directory is needed
NUM_GPU=8

mkdir -p "../results_ntu120/3DV_baseline/"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py \
    --root_path "${DATA_DIR}/3DV_pointdata/NTU_voxelsize35_split5/" \
    --depth_path "${DATA_DIR}" \
    --save_root_dir "${GIT_DIR}/results_ntu120/3DV_baseline/" \
    --ngpu ${NUM_GPU} \
    --nepoch 50 \
    --dataset "ntu120"

