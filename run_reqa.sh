export  CUDA_VISIBLE_DEVICES=0
DATASET=bioasq9b
POOLER=mean
BATCH_SIZE=32
NUM_EPOCHS=10
CACHE_DIR=/home/lzz/myfile/models
for PLM in bert-base-uncased dmis-lab/biobert-base-cased-v1.1
do
    for LR in 1e-5 2e-5 3e-5 4e-5 5e-5
    do
        SAVE_DIR=./output/${DATASET}/fine_tuning/${PLM#*/}_${POOLER}/bs${BATCH_SIZE}_e${NUM_EPOCHS}_lr${LR}/
        mkdir -p ${SAVE_DIR}
        nohup python3 train_reqa.py \
            --seeds 0 42 512 2023 20246\
            --main_metric mrr \
            --dataset ${DATASET} \
            --epoch ${NUM_EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --pooler ${POOLER} \
            --model_name_or_path ${PLM} \
            --matching_func dot \
            --temperature 1 \
            --cache_dir ${CACHE_DIR} \
            --learning_rate ${LR} \
            --save_dir ${SAVE_DIR} \
            --rm_saved_model True \
            --save_results True \
            > ${SAVE_DIR}/run.log 2>&1
    done 
done
