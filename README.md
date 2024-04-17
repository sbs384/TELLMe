# TELLMe
This repository provides the PyTorch implementation of our paper "TELLMe: Teaching and Exploiting Large Language
Models for Model Selection in Text Retrieval".

## Installation
```bash
# Install huggingface transformers and pytorch
transformers==4.30.2
torch==2.1.0
```

## Dataset
The proposed TELLMe framework involves the model retrieval and the model selection. We implement experiments on 3 datasets: ReQA BioASQ 9b, SciFact and NQ. It is noted that we use 10,000 samples extracted from NQ to calculate EaSe scores for model ranking, which is named as 'NQ_sample'. All the datasets can be download from [AllNLI.tsv.gz](https://sbert.net/datasets/AllNLI.tsv.gz). The downloaded data package should be unzip to "./data/".


## Candidate Pre-trained Models for Model Retrieval
In the model retrieval stage, we first fine-tuned 50 pre-trained models as candidate pool. All the pre-trained model used can be found on huggingface according to the models' name. The list of pre-trained models and their performance on the ReQA BioASQ 9b, SciFact and NQ datasets are represented as follows.


## Candidate Pre-trained Models for Model Ranking
In the model ranking stage, we randomly selected 20 models from the aforementioned pool of 50 candidates:


## Example

We show the running cases of the baseline model (Dual-BioBERT) and the proposed RBAR-BioBERT on ReQA BioASQ 9b dataset, the other experiments can also be reproduced by using the hyper-parameters provided in the paper.

### Transforming BioASQ dataset into ReQA BioASQ dataset
First of all, run the script "run_data_processing.sh".
```bash
# Transform the BioASQ 9b dataset into ReQA BioASQ 9b.
python3 reqa_bioasq_data_processing.py --dataset 9b
```

### Dual-BioBERT
#### 1. Fine-tuning Dual-BioBERT on ReQA BioASQ dataset
run the script "run_reqa_baseline.sh".
```bash
python3 train_reqa.py \
    --seed 12345 \
    --do_train True \
    --do_test True \
    --dev_metric p1 \
    --dataset 6b \
    --max_question_len 24 \
    --max_answer_len 168 \
    --epoch 10 \
    --batch_size 32 \
    --model_type dual_encoder \
    --encoder_type biobert \
    --plm_path /workspace/baijun/models/english/biobert-base-cased-v1.1 \ # your path to the pre-trained model parameters of bioert
    --pooler_type mean \
    --matching_func cos \
    --whitening False \
    --temperature 0.001 \
    --learning_rate 5e-5 \
    --save_model_path output/6b/biobert_baseline/ \
    --rm_saved_model True \
    --save_results True \
```

### RBAR-BioBERT
#### 1. Pre-training on NLI datasets
run the script "run_nli_ranking.sh".
```bash
python3 train_nli_ranking.py \
  --max_premise_len 64 \
  --max_hypothesis_len 32 \
  --epoch 3 \
  --batch_size 32 \
  --model_type dual_encoder_wot \
  --encoder_type biobert \
  --plm_path /workspace/baijun/models/english/biobert-base-cased-v1.1 \ # your path to the pre-trained model parameters of bioert
  --matching_func cos \
  --pooler_type mean \
  --temperature 0.05 \
  --save_model_path output/nli/biobert_ranking \
  --learning_rate 2e-5 \
  --warmup_proportion 0.1 \
  --seed 12345
```

#### 2. Fine-tuning on ReQA BioASQ
run the script "run_reqa_rbar.sh".
```bash
python3 train_reqa.py \
    --seed 12345 \
    --do_train True \
    --do_test True \
    --dev_metric p1 \
    --dataset 6b \
    --max_question_len 24 \
    --max_answer_len 168 \
    --epoch 10 \
    --batch_size 32 \
    --model_type dual_encoder_wot \
    --encoder_type biobert \
    --plm_path /workspace/baijun/models/english/biobert-base-cased-v1.1 \ # your path to the pre-trained model parameters of bioert
    --pooler_type mean \
    --matching_func cos \
    --whitening True \
    --temperature 0.05 \
    --learning_rate 5e-5 \
    --save_model_path output/6b/biobert_rbar/ \
    --load_model_path output/nli/biobert_ranking/model.pt \
    --rm_saved_model True \
    --save_results True \
```

## License and Disclaimer
Please see the LICENSE file for details. Downloading data indicates your acceptance of our disclaimer.


## Contact
For help or issues using RBAR framework, please create an issue.
