#!/bin/bash
export  CUDA_VISIBLE_DEVICES=1
DATASETS=("NQ_sample")
DATASET="NQ_sample"
CACHE_DIR=/home/lzz/myfile/models

LM_NAMES="bert-base-cased bert-base-uncased roberta-base dmis-lab/biobert-base-cased-v1.1 google/electra-base-discriminator \
          princeton-nlp/unsup-simcse-bert-base-uncased princeton-nlp/sup-simcse-bert-base-uncased facebook/bart-base \
          allenai/scibert_scivocab_cased allenai/scibert_scivocab_uncased distilbert-base-cased distilbert-base-uncased \
          nghuyong/ernie-2.0-base-en distilroberta-base distilbert-base-multilingual-cased albert-base-v2 \
          microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext michiyasunaga/BioLinkBERT-base distilgpt2 openai-gpt"

#LM_NAMES="bert-base-uncased bert-base-cased"
SEEDS="1117 1114 1027 820 905"
ALL_CANDIDATE_SIZES="4"
#METHODS="PACTran-0.1-10 Logistic GBC TransRate SFDA LogME HScore NLEEP"
#METHODS="PiLogME-whitening-0 PiLogME-whitening-0.05 PiLogME-whitening-0.1 PiLogME-whitening-0.15 PiLogME-whitening-0.2 PiLogME-whitening-0.25 \
#             PiLogME-whitening-0.3 PiLogME-whitening-0.35 PiLogME-whitening-0.4 PiLogME-whitening-0.45 PiLogME-whitening-0.5 PiLogME-whitening-0.55 \
#             PiLogME-whitening-0.6 PiLogME-whitening-0.65 PiLogME-whitening-0.7 PiLogME-whitening-0.75 PiLogME-whitening-0.8 PiLogME-whitening-0.85 \
#             PiLogME-whitening-0.9 PiLogME-whitening-0.95 PiLogME-whitening-1"
#METHODS="PiLogME-0.05 PiLogME-0.1 PiLogME-0.15 PiLogME-0.2 PiLogME-0.25 \
#             PiLogME-0.3 PiLogME-0.35 PiLogME-0.4 PiLogME-0.45 PiLogME-0.5 PiLogME-0.55 \
#             PiLogME-0.6 PiLogME-0.65 PiLogME-0.7 PiLogME-0.75 PiLogME-0.8 PiLogME-0.85 \
#             PiLogME-0.9 PiLogME-0.95 PiLogME-1 PiLogME-0 "
METHODS="PiLogME-whitening-0.15"
SAVE_RESULTS="True"
OVERWRITE_RESULTS="False"


# iterate over datasets
for DIM in 64 128 512 1024 
do
    dataset=${DATASET}
    SAVE_DIR=./output/$dataset/model_selection_white/
    mkdir -p ${SAVE_DIR}
    nohup python3 model_selection_2.py \
        --methods ${METHODS}-${DIM}-${MARGIN} \
        --all_candidate_sizes $ALL_CANDIDATE_SIZES \
        --dataset $dataset \
        --batch_size 64 \
        --model_name_or_paths ${LM_NAMES} \
        --matching_func dot \
        --pooler mean \
        --cache_dir ${CACHE_DIR} \
        --seeds ${SEEDS} \
        --save_results ${SAVE_RESULTS} \
        --overwrite_results ${OVERWRITE_RESULTS} \
        --dim ${DIM} \
        > ${SAVE_DIR}/run.log 2>&1
done
   
    
