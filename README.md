# TELLMe
This repository provides the PyTorch implementation of our paper "TELLMe: Teaching and Exploiting Large Language
Models for Model Selection in Text Retrieval".

## Installation
```bash
# Install huggingface transformers and pytorch
transformers==4.30.2
torch==2.1.0
```

## Datasets
The proposed TELLMe framework involves 2 stage: model retrieval and model selection. We implement experiments on 3 datasets: ReQA BioASQ 9b, SciFact and NQ. It is noted that we use 10,000 samples extracted from NQ to calculate EaSe scores for model ranking, which is named as 'NQ_sample'. The data ca be download from [data](https://www.dropbox.com/scl/fi/p06yqwpq7mu42jpwj8ojj/data.zip?rlkey=vtjxobbcy3rwoiily961s2nlf&dl=0). The downloaded data package should be unzip to "./data/.Table 1 describes the statistics of all the datasets:
### Table 1. Dataset Statistics
<table>
   <tr>
      <td rowspan="2">Datasets</td>
      <td colspan="2" align="center">Training Set</td>
      <td colspan="2" align="center">Test Set</td>
   </tr>
   <tr>
      <td align="center">#Q</td>
      <td align="center">#D</td>
      <td align="center">#D</td>
      <td align="center">#D</td>
   </tr>
   <tr>
      <td>ReQA BioASQ 9b</td>
      <td align="center">5,828</td>
      <td align="center">5,828</td>
      <td align="center">496</td>
      <td align="center">31,682</td>
   </tr>
  <tr>
      <td>SciFact</td>
      <td align="center">919</td>
      <td align="center">919</td>
      <td align="center">300</td>
      <td align="center">5,183</td>
   </tr>
   <tr>
      <td>NQ</td>
      <td align="center">58,880</td>
      <td align="center">58,880</td>
      <td align="center">10,125</td>
      <td align="center">10,125</td>
   </tr>
</table>
#Q，#D，#C represent the query number, the document number and candidate document number respectively.

## Evaluation Metrics
### Model Retrieval
For model retrieval, the top-K recall rate (Recall@K, R@K$) is used as the evaluation metric. The calculation of R@K for model retrieval is as follows:
$$R@K = \frac{|\Phi^{mr}_{top_K} \cap \Phi^{real}_{top_K}|}{K}$$
where $\Phi^{mr}_{top_K}$ is the set of top K models obtained during the model retrieval stage, and $\Phi^{real}_{top_K}$ is the set of top K models in reality.

## Example
We show the running cases of the code used for the related experiments. 

### Fine-tuning
To fine-tune all the candidate pre-trained models, run the script "run_reqa.sh". In the following example, we show a fine-tuning process of bert-base-un-cased(BERT) and dmis-lab/biobert-base-cased-v1.1(BioBERT) on bioasq9b. To obtain the best performance of each pre-trained model on different dataset, we tried several hyper-parameter combinations. For the ReQA and SciFact datasets, learning rate is set among 1e-5, 2e-5, 3e-4, 4e-5 and 5e-5; seeds are set to 0, 42 and 512. For the NQ dataset, learning rate is set among 2e-5, 3e-4, 4e-5 and 5e-5; seeds are set to 0, 42 and 512. Other args are the same as what showed in the script example.
```bash
export  CUDA_VISIBLE_DEVICES=0
DATASET=bioasq9b
POOLER=mean
BATCH_SIZE=32
NUM_EPOCHS=10
CACHE_DIR=/home/lzz/myfile/models  # your path to the pre-trained model parameters
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
```

### Model ranking
#### 1. Calculate transferability scores using EaSe.
To calculate transferability scores, run the script "run_model_selection.sh". METHODS represents different transferability estimation methods.
```bash
#!/bin/bash
export  CUDA_VISIBLE_DEVICES=0
DATASETS=("bioasq9b")
DATASET="bioasq9b"
CACHE_DIR=/home/lzz/myfile/models # your path to the pre-trained model parameters

# candidate pre-trained models for model ranking 
LM_NAMES="bert-base-cased bert-base-uncased roberta-base dmis-lab/biobert-base-cased-v1.1 google/electra-base-discriminator \
          princeton-nlp/unsup-simcse-bert-base-uncased princeton-nlp/sup-simcse-bert-base-uncased facebook/bart-base \
          allenai/scibert_scivocab_cased allenai/scibert_scivocab_uncased distilbert-base-cased distilbert-base-uncased \
          nghuyong/ernie-2.0-base-en distilroberta-base distilbert-base-multilingual-cased albert-base-v2 \
          microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext michiyasunaga/BioLinkBERT-base distilgpt2 openai-gpt"


SEEDS="1117 1114 1027 820 905"
ALL_CANDIDATE_SIZES="2 3 4 5 6 7 8 9 10"
#METHODS="PACTran-0.1-10 Logistic GBC TransRate SFDA LogME HScore NLEEP" #compared tansferability estimation methods.
METHODS="EaSe-whitening-0 EaSe-whitening-0.05 EaSe-whitening-0.1 EaSe-whitening-0.15 EaSe-whitening-0.2 EaSe-whitening-0.25 \
             EaSe-whitening-0.3 EaSe-whitening-0.35 EaSe-whitening-0.4 EaSe-whitening-0.45 EaSe-whitening-0.5 EaSe-whitening-0.55 \
             EaSe-whitening-0.6 EaSe-whitening-0.65 EaSe-whitening-0.7 EaSe-whitening-0.75 EaSe-whitening-0.8 EaSe-whitening-0.85 \
             EaSe-whitening-0.9 EaSe-whitening-0.95 EaSe-whitening-1"
#METHODS="EaSe-0.05 EaSe-0.1 EaSe-0.15 EaSe-0.2 EaSe-0.25 \
#             EaSe-0.3 EaSe-0.35 EaSe-0.4 EaSe-0.45 EaSe-0.5 EaSe-0.55 \
#             EaSe-0.6 EaSe-0.65 EaSe-0.7 EaSe-0.75 EaSe-0.8 EaSe-0.85 \
#             EaSe-0.9 EaSe-0.95 EaSe-1 EaSe-0 "
SAVE_RESULTS="True"
OVERWRITE_RESULTS="False"
dataset=${DATASET}
SAVE_DIR=./output/${DATASET}/model_selection/
mkdir -p ${SAVE_DIR}
nohup python3 model_selection.py \
    --methods ${METHODS} \
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
    --save_dir ${SAVE_DIR} \
    > ${SAVE_DIR}/run.log 2>&1
```
#### 2. Evaluating Kendalls' $\tau$ between EaSe and fine-tuning performance of pre-trained models.
run the script "eval_model_selection.py".
```python3 
eval_model_selection.py
```

## Candidate Pre-trained Models for Model Retrieval
In the model retrieval stage, we first fine-tuned 50 pre-trained models as candidate pool. All the pre-trained model used can be found on huggingface according to the models' name. The list of pre-trained models and their performance on the ReQA BioASQ 9b, SciFact and NQ datasets are represented in Table 2, Table 3 and Table 4.

### Table 2. Pre-trained model Performance on the ReQA BioASQ 9b Dataset
| Number | Pre-trained Model | MRR | P@1 | R@5 |
| :---         | :---      | :---: | :---: | :---: |
| 1 | bert-base-uncased | 0.693 | 0.588 | 0.673 |
| 2 | bert-base-cased | 0.677 | 0.572 | 0.637 |
| 3 | roberta-base | 0.625 | 0.508 | 0.617 |
| 4 | biobert-base-cased-v1.1 | 0.739 | 0.633 | 0.719 | 
| 5 | electra-base-discriminator | 0.624 | 0.525 | 0.587 | 
| 6 | unsup-simcse-bert-base-uncased | 0.696 | 0.591 | 0.680 | 
| 7 | sup-simcse-bert-base-uncased | 0.703 | 0.603 | 0.677 |
| 8 | openai-gpt | 0.590 | 0.483 | 0.557 |
| 9 | bart-base | 0.672 | 0.563 | 0.659 |
| 10 | scibert_scivocab_cased | 0.725 | 0.617 | 0.702 | 
| 11 | scibert_scivocab_uncased | 0.726 | 0.614 | 0.713 | 
| 12 | distilbert-base-cased | 0.669 | 0.567 | 0.634 |
| 13 | distilbert-base-uncased | 0.700 | 0.589 | 0.687 | 
| 14 | ernie-2.0-base-en | 0.708 | 0.596 | 0.694 |
| 15 | distilroberta-base | 0.638 | 0.517 | 0.632 |
| 16 | distilgpt2 | 0.454 | 0.350 | 0.414 |
| 17 | distilbert-base-multilingual-cased | 0.629 | 0.513 | 0.615 |
| 18 | albert-base-v2 | 0.623 | 0.516 | 0.597 |
| 19 | BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext | 0.748 | 0.633 | 0.736 |
| 20 | BioLinkBERT-base | 0.717 | 0.600 | 0.713 |
| 21 | ClinicalBERT | 0.603 | 0.494 | 0.587 |
| 22 | bluebert_pubmed_uncased_L-12_H-768_A-12 | 0.623 | 0.512 | 0.601 |
| 23 | bert-medical-ner-proj | 0.650 | 0.540 | 0.627 |
| 24 | oubiobert-base-uncased | 0.711 | 0.593 | 0.704 |
| 25 | bioelectra-base-discriminator-pubmed | 0.533 | 0.423 | 0.498 |
| 26 | BioRedditBERT-uncased | 0.679 | 0.564 | 0.670 |
| 27 | mirror-bert-base-uncased-word | 0.658 | 0.559 | 0.627 |
| 28 | SapBERT-from-PubMedBERT-fulltext | 0.702 | 0.584 | 0.695 |
| 29 | electra-medal | 0.385 | 0.298 | 0.339 |
| 30 | xlm-roberta-base | 0.580 | 0.467 | 0.557 |
| 31 | spanbert-base-cased | 0.650 | 0.545 | 0.624 |
| 32 | BERT-of-Theseus-MNLI | 0.594 | 0.491 | 0.560 |
| 33 | mirror-bert | 0.619 | 0.527 | 0.568 |
| 34 | bio_roberta-base_pubmed | 0.632 | 0.510 | 0.624 |
| 35 | MathBERT | 0.552 | 0.456 | 0.501 |
| 36 | netbert | 0.631 | 0.532 | 0.589 |
| 37 | legal-bert-base-uncased | 0.664 | 0.563 | 0.639 |
| 38 | graphcodebert-base | 0.615 | 0.501 | 0.598 |
| 39 | hateBERT | 0.650 | 0.549 | 0.623 |
| 40 | twitter-roberta-base-sentiment-latest | 0.634 | 0.527 | 0.608 |
| 41 | RadBERT | 0.582 | 0.479 | 0.536 |
| 42 | covid-radbert | 0.562 | 0.461 | 0.518 |
| 43 | mobilebert-uncased | 0.604 | 0.493 | 0.582 |
| 44 | mirrorwic-bert-base-uncased | 0.647 | 0.542 | 0.622 |
| 45 | muppet-roberta-base | 0.593 | 0.475 | 0.589 |
| 46 | Robust-Biomed-RoBERTa-QuestionAnswering | 0.620 | 0.493 | 0.613 |
| 47 | math_pretrained_roberta | 0.596 | 0.477 | 0.578 |
| 48 | BERTLaw | 0.571 | 0.471 | 0.527 |
| 49 | biomed_roberta_base | 0.631 | 0.510 | 0.618 |
| 50 | roberta-argument | 0.631 | 0.517 | 0.613 |


### Table 3. Pre-trained model Performance on the SciFact Dataset
| Number | Pre-trained Model | MRR | P@1 | R@5 |
| :---         | :---      | :---: | :---: | :---: |
1 | bert-base-uncased | 0.618 | 0.515 | 0.724 |
2 | bert-base-cased | 0.607 | 0.505 | 0.709 |
3 | roberta-base | 0.578 | 0.476 | 0.681 |
4 | biobert-base-cased-v1.1 | 0.672 | 0.565 | 0.797 |
5 | electra-base-discriminator | 0.385 | 0.285 | 0.476 |
6 | unsup-simcse-bert-base-uncased | 0.603 | 0.498 | 0.710 |
7 | sup-simcse-bert-base-uncased | 0.612 | 0.513 | 0.721 |
8 | openai-gpt | 0.577 | 0.480 | 0.679 |
9 | bart-base | 0.621 | 0.521 | 0.722 |
10 | scibert_scivocab_cased | 0.647 | 0.527 | 0.795 |
11 | scibert_scivocab_uncased | 0.686 | 0.581 | 0.803 |
12 | distilbert-base-cased | 0.559 | 0.459 | 0.667 |
13 | distilbert-base-uncased | 0.611 | 0.512 | 0.714 |
14 | ernie-2.0-base-en | 0.647 | 0.547 | 0.757 |
15 | distilroberta-base | 0.552 | 0.449 | 0.659 |
16 | distilgpt2 | 0.096 | 0.047 | 0.118 |
17 | distilbert-base-multilingual-cased | 0.567 | 0.462 | 0.683 |
18 | albert-base-v2 | 0.502 | 0.385 | 0.628 |
19 | BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext | 0.720 | 0.609 | 0.852 |
20 | BioLinkBERT-base | 0.726 | 0.619 | 0.858 |
21 | ClinicalBERT | 0.551 | 0.445 | 0.669 |
22 | bluebert_pubmed_uncased_L-12_H-768_A-12 | 0.600 | 0.489 | 0.709 |
23 | bert-medical-ner-proj | 0.579 | 0.472 | 0.694 |
24 | oubiobert-base-uncased | 0.698 | 0.579 | 0.843 |
25 | bioelectra-base-discriminator-pubmed | 0.380 | 0.279 | 0.477 |
26 | BioRedditBERT-uncased | 0.620 | 0.519 | 0.729 |
27 | mirror-bert-base-uncased-word | 0.606 | 0.497 | 0.726 |
28 | SapBERT-from-PubMedBERT-fulltext | 0.689 | 0.577 | 0.818 |
29 | electra-medal | 0.165 | 0.084 | 0.216 |
30 | xlm-roberta-base | 0.518 | 0.406 | 0.635 |
31 | spanbert-base-cased | 0.681 | 0.605 | 0.763 |
32 | BERT-of-Theseus-MNLI | 0.471 | 0.371 | 0.573 |
33 | mirror-bert | 0.490 | 0.410 | 0.568 |
34 | bio_roberta-base_pubmed | 0.606 | 0.490 | 0.730 |
35 | MathBERT | 0.513 | 0.435 | 0.588 |
36 | netbert | 0.568 | 0.492 | 0.647 |
37 | legal-bert-base-uncased | 0.597 | 0.497 | 0.709 |
38 | graphcodebert-base | 0.572 | 0.467 | 0.678 |
39 | hateBERT | 0.581 | 0.494 | 0.674 |
40 | twitter-roberta-base-sentiment-latest | 0.549 | 0.454 | 0.650 |
41 | RadBERT | 0.489 | 0.387 | 0.593 |
42 | covid-radbert | 0.468 | 0.363 | 0.581 |
43 | mobilebert-uncased | 0.275 | 0.157 | 0.398 |
44 | mirrorwic-bert-base-uncased | 0.583 | 0.485 | 0.685 |
45 | muppet-roberta-base | 0.577 | 0.475 | 0.685 |
46 | Robust-Biomed-RoBERTa-QuestionAnswering | 0.578 | 0.462 | 0.714 |
47 | math_pretrained_roberta | 0.546 | 0.437 | 0.657 |
48 | BERTLaw | 0.531 | 0.441 | 0.619 |
49 | biomed_roberta_base | 0.580 | 0.467 | 0.700 |
50 | roberta-argument | 0.564 | 0.463 | 0.669 |

### Table 4. Pre-trained model Performance on the NQ Dataset
| Number | Pre-trained Model | MRR | P@1 | R@5 |
| :---         | :---      | :---: | :---: | :---: |
| 1 | bert-base-uncased | 0.619 | 0.532 | 0.730 |
| 2 | bert-base-cased | 0.567 | 0.471 | 0.690 |
| 3 | roberta-base | 0.600 | 0.510 | 0.714 |
| 4 | biobert-base-cased-v1.1 | 0.542 | 0.449 | 0.660 |
| 5 | electra-base-discriminator | 0.543 | 0.447 | 0.666 |
| 6 | unsup-simcse-bert-base-uncased | 0.615 | 0.526 | 0.728 |
| 7 | sup-simcse-bert-base-uncased | 0.620 | 0.533 | 0.732 |
| 8 | openai-gpt | 0.536 | 0.437 | 0.661 |
| 9 | bart-base | 0.514 | 0.431 | 0.617 |
| 10 | scibert_scivocab_cased | 0.544 | 0.453 | 0.657 |
| 11 | scibert_scivocab_uncased | 0.574 | 0.485 | 0.687 |
| 12 | distilbert-base-cased | 0.558 | 0.461 | 0.679 |
| 13 | distilbert-base-uncased | 0.608 | 0.520 | 0.719 |
| 14 | ernie-2.0-base-en |  0.634 | 0.550 | 0.741 |
| 15 | distilroberta-base | 0.579 | 0.488 | 0.696 |
| 16 | distilgpt2 | 0.489 | 0.383 | 0.621 |
| 17 | distilbert-base-multilingual-cased | 0.550 | 0.454 | 0.670 |
| 18 | albert-base-v2 | 0.542 | 0.448 | 0.660 |
| 19 | BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext | 0.574 | 0.487 | 0.684 |
| 20 | BioLinkBERT-base | 0.577 | 0.491 | 0.687 |
| 21 | ClinicalBERT | 0.551 | 0.460 | 0.665 |
| 22 | bluebert_pubmed_uncased_L-12_H-768_A-12 | 0.543 | 0.450 | 0.660 |
| 23 | bert-medical-ner-proj | 0.617 | 0.529 | 0.730 |
| 24 | oubiobert-base-uncased | 0.565 | 0.476 | 0.679 |
| 25 | bioelectra-base-discriminator-pubmed | 0.482 | 0.387 | 0.597 |
| 26 | BioRedditBERT-uncased | 0.569 | 0.478 | 0.686 |
| 27 | mirror-bert-base-uncased-word | 0.618 | 0.529 | 0.732 |
| 28 | SapBERT-from-PubMedBERT-fulltext | 0.568 | 0.480 | 0.677 |
| 29 | electra-medal | 0.410 | 0.315 | 0.521 |
| 30 | xlm-roberta-base | 0.568 | 0.476 | 0.684 |
| 31 | spanbert-base-cased | 0.615 | 0.529 | 0.723 |
| 32 | BERT-of-Theseus-MNLI | 0.551 | 0.453 | 0.675 |
| 33 | mirror-bert | 0.516 | 0.424 | 0.631 |
| 34 | bio_roberta-base_pubmed | 0.573 | 0.481 | 0.689 |
| 35 | MathBERT | 0.461 | 0.368 | 0.577 |
| 36 | netbert | 0.486 | 0.387 | 0.605 |
| 37 | legal-bert-base-uncased | 0.574 | 0.487 | 0.684 |
| 38 | graphcodebert-base | 0.538 | 0.445 | 0.653 |
| 39 | hateBERT | 0.572 | 0.480 | 0.689 |
| 40 | twitter-roberta-base-sentiment-latest | 0.584 | 0.494 | 0.699 |
| 41 | RadBERT | 0.497 | 0.401 | 0.614 |
| 42 | covid-radbert | 0.492 | 0.397 | 0.611 |
| 43 | mobilebert-uncased | 0.559 | 0.458 | 0.690 |
| 44 | mirrorwic-bert-base-uncased | 0.619 | 0.530 | 0.732 |
| 45 | muppet-roberta-base | 0.598 | 0.510 | 0.710 |
| 46 | Robust-Biomed-RoBERTa-QuestionAnswering | 0.585 | 0.494 | 0.702 |
| 47 | math_pretrained_roberta | 0.571 | 0.478 | 0.688 |
| 48 | BERTLaw | 0.508 | 0.416 | 0.624 |
| 49 | biomed_roberta_base | 0.585 | 0.496 | 0.700 |
| 50 | roberta-argument | 0.597 | 0.511 | 0.711 |
           

## Candidate Pre-trained Models for Model Ranking
In the model ranking stage, we randomly selected 20 models from the aforementioned pool of 50 candidates, which are showed in Table 5.
### Table 5. Pre-trained Models for Model Ranking
| Number | Pre-trained Model 
| :---   | :--- |
| 1 |bert-base-uncased |
| 2 | bert-base-cased |
| 3 | roberta-base |
| 4 | biobert-base-cased-v1.1 |
| 5 | electra-base-discriminator |
| 6 | unsup-simcse-bert-base-uncased |
| 7 | sup-simcse-bert-base-uncased |
| 8 | openai-gpt |
| 9 | bart-base |
| 10 | scibert_scivocab_cased |
| 11 | scibert_scivocab_uncased |
| 12 | distilbert-base-cased |
| 13 | distilbert-base-uncased |
| 14 | ernie-2.0-base-en |
| 15 | distilroberta-base |
| 16 | distilgpt2 |
| 17 | distilbert-base-multilingual-cased |
| 18 | albert-base-v2 |
| 19 | BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext |
| 20 | BioLinkBERT-base |

## An entire prompt used for model retrieval
#### The following content is an entire prompt used for model retrieval on the NQ dataset, with both ICL(In-context Learning) and CoT(Chain-of-Thought). Here, $n_d=10$, $n_m=5$, $\{\mathcal{D}^{\prime}\}$ are ReQA BioASQ 9b and SciFact. 

I will give you some addtional information.The first part of the additional information is some candidate PLMs:
\[1\]bert-base-uncased \
\[2\]bert-base-cased \
\[3\]roberta-base \
\[4\]dmis-lab/biobert-base-cased-v1.1 \
\[5\]google/electra-base-discriminator \
\[6\]princeton-nlp/unsup-simcse-bert-base-uncased \
\[7\]princeton-nlp/sup-simcse-bert-base-uncased \
\[8\]openai-gpt \
\[9\]facebook/bart-base \
\[10\]allenai/scibert_scivocab_cased \
\[11\]allenai/scibert_scivocab_uncased \
\[12\]distilbert-base-cased \
\[13\]distilbert-base-uncased \
\[14\]nghuyong/ernie-2.0-base-en \
\[15\]distilroberta-base \
\[16\]distilgpt2 \
\[17\]distilbert-base-multilingual-cased \
\[18\]albert-base-v2 \
\[19\]microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
\[20\]michiyasunaga/BioLinkBERT-base \
\The second part of the additional information is exampled ranking results of models mentioned above on a given dataset: \
1.Dataset:ReQA BioASQ-9b.Models:\[1,2,5,6,19\].Ranking:\[19,6,1,2,5\] \
2.Dataset:ReQA BioASQ-9b.Models:\[3,4,16,18,19\].Ranking:\[19,4,3,18,16\] \
3.Dataset:ReQA BioASQ-9b.Models:\[8,9,11,12,20\].Ranking:\[11,20,9,12,8\] \
4.Dataset:ReQA BioASQ-9b.Models:\[6,7,9,10,20\].Ranking:\[10,20,7,6,9\] \
5.Dataset:ReQA BioASQ-9b.Models:\[3,8,11,13,14\].Ranking:\[11,14,13,3,8\] \
6.Dataset:SciFact.Models:\[2,5,11,14,16\].Ranking:\[11,14,2,5,16\] \
7.Dataset:SciFact.Models:\[3,10,13,17,20\].Ranking:\[20,10,13,3,17\] \
8.Dataset:SciFact.Models:\[2,3,5,7,12\].Ranking:\[7,2,3,12,5\] \
9.Dataset:SciFact.Models:\[1,2,5,6,8\].Ranking:\[1,2,6,8,5\] \
10.Dataset:SciFact.Models:\[2,5,9,10,16\].Ranking:\[10,9,2,5,16\] \
With the growing number of Pre-trained Language Models recently,fine-tuning all PLMs for model selection is not practical,I usually don't know which PLM should  be selected to achieve the best dataset performance.Here is the problem I want you to solve:For Natural Questions dataset,and the candidate models:
\[1\]bert-base-uncased \
\[2\]bert-base-cased \
\[3\]roberta-base \
\[4\]dmis-lab/biobert-base-cased-v1.1 \
\[5\]google/electra-base-discriminator \
\[6\]princeton-nlp/unsup-simcse-bert-base-uncased \
\[7\]princeton-nlp/sup-simcse-bert-base-uncased \
\[8\]openai-gpt \
\[9\]facebook/bart-base \
\[10\]allenai/scibert_scivocab_cased \
\[11\]allenai/scibert_scivocab_uncased \
\[12\]distilbert-base-cased \
\[13\]distilbert-base-uncased \
\[14\]nghuyong/ernie-2.0-base-en \
\[15\]distilroberta-base \
\[16\]distilgpt2 \
\[17\]distilbert-base-multilingual-cased \
\[18\]albert-base-v2 \
\[19\]microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
\[20\]michiyasunaga/BioLinkBERT-base \
\[21\]medicalai/ClinicalBERT \
\[22\]bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12 \
\[23\]medical-ner-proj/bert-medical-ner-proj \
\[24\]seiya/oubiobert-base-uncased \
\[25\]kamalkraj/bioelectra-base-discriminator-pubmed \
\[26\]cambridgeltl/BioRedditBERT-uncased \
\[27\]cambridgeltl/mirror-bert-base-uncased-word \
\[28\]cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
\[29\]McGill-NLP/electra-medal \
\[30\]xlm-roberta-base \
\[31\]SpanBERT/spanbert-base-cased \
\[32\]canwenxu/BERT-of-Theseus-MNLI \
\[33\]slider/mirror-bert \
\[34\]minhpqn/bio_roberta-base_pubmed \
\[35\]tbs17/MathBERT \
\[36\]antoinelouis/netbert \
\[37\]nlpaueb/legal-bert-base-uncased \
\[38\]microsoft/graphcodebert-base \
\[39\]GroNLP/hateBERT
\[40\]cardiffnlp/twitter-roberta-base-sentiment-latest \
\[41\]StanfordAIMI/RadBERT \
\[42\]StanfordAIMI/covid-radbert \
\[43\]google/mobilebert-uncased \
\[44\]cambridgeltl/mirrorwic-bert-base-uncased \
\[45\]facebook/muppet-roberta-base \
\[46\]mmoradi/Robust-Biomed-RoBERTa-QuestionAnswering \
\[47\]AnReu/math_pretrained_roberta \
\[48\]nguyenthanhasia/BERTLaw \
\[49\]allenai/biomed_roberta_base \
\[50\]chkla/roberta-argument \
Please rank all the 50 models.Rember that you don't have to give an accurate ranking list,any possible inference is OK.
You need to solve this problem in 4 steps:
1.Search information of the given Dataset.
2.Search characteristics, structure, target tasks, applied domain and config settings for every given model.
3.Consider the addtional information I gave you above.
4.Combine all the knowledge you got through steps 1-3 and output the ranking list.
Now please output a brief ranking list of 50 models instead of explanation.

#### Here is the output of Bing Chat responding to the prompt listed above:

Based on the information provided and the characteristics of the models, here is a possible ranking for the Natural Questions dataset:

1. microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
2. dmis-lab/biobert-base-cased-v1.1
3. allenai/scibert_scivocab_cased
4. allenai/scibert_scivocab_uncased
5. nghuyong/ernie-2.0-base-en
6. distilbert-base-cased
7. distilbert-base-uncased
8. bert-base-uncased
9. bert-base-cased
10. roberta-base
11. distilroberta-base
12. distilgpt2
13. distilbert-base-multilingual-cased
14. albert-base-v2
15. michiyasunaga/BioLinkBERT-base
16. medicalai/ClinicalBERT
17. bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12
18. medical-ner-proj/bert-medical-ner-proj
19. seiya/oubiobert-base-uncased
20. kamalkraj/bioelectra-base-discriminator-pubmed
21. cambridgeltl/BioRedditBERT-uncased
22. cambridgeltl/mirror-bert-base-uncased-word
23. cambridgeltl/SapBERT-from-PubMedBERT-fulltext
24. McGill-NLP/electra-medal
25. xlm-roberta-base
26. SpanBERT/spanbert-base-cased
27. canwenxu/BERT-of-Theseus-MNLI
28. slider/mirror-bert
29. minhpqn/bio_roberta-base_pubmed
30. tbs17/MathBERT
31. antoinelouis/netbert
32. nlpaueb/legal-bert-base-uncased
33. microsoft/graphcodebert-base
34. GroNLP/hateBERT
35. cardiffnlp/twitter-roberta-base-sentiment-latest
36. StanfordAIMI/RadBERT
37. StanfordAIMI/covid-radbert
38. google/mobilebert-uncased
39. cambridgeltl/mirrorwic-bert-base-uncased
40. facebook/muppet-roberta-base
41. mmoradi/Robust-Biomed-RoBERTa-QuestionAnswering
42. AnReu/math_pretrained_roberta
43. nguyenthanhasia/BERTLaw
44. allenai/biomed_roberta_base
45. chkla/roberta-argument
46. princeton-nlp/unsup-simcse-bert-base-uncased
47. princeton-nlp/sup-simcse-bert-base-uncased
48. openai-gpt
49. facebook/bart-base
50. google/electra-base-discriminator

Please note that this is a rough estimation and the actual performance may vary depending on the specific task and fine-tuning process. It's always recommended to experiment with a subset of models to find the best fit for your specific task.



## License and Disclaimer
Please see the LICENSE file for details. Downloading data indicates your acceptance of our disclaimer.


## Contact
For help or issues using RBAR framework, please create an issue.
