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
The proposed TELLMe framework involves 2 stage: model retrieval and model selection. We implement experiments on 3 datasets: ReQA BioASQ 9b, SciFact and NQ. It is noted that we use 10,000 samples extracted from NQ to calculate EaSe scores for model ranking, which is named as 'NQ_sample'. All the datasets can be download from [AllNLI.tsv.gz](https://sbert.net/datasets/AllNLI.tsv.gz). The downloaded data package should be unzip to "./data/".


## Candidate Pre-trained Models for Model Retrieval
In the model retrieval stage, we first fine-tuned 50 pre-trained models as candidate pool. All the pre-trained model used can be found on huggingface according to the models' name. The list of pre-trained models and their performance on the ReQA BioASQ 9b, SciFact and NQ datasets are represented as follows.

### Pre-trained model Performance on ReQA BioASQ 9b Dataset
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
| 22 | bluebert_pubmed_uncased_L-12_H-768_A-12 | 0.623 & 0.512 & 0.601 |
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
| 33 | mirror-bert & 0.619 | 0.527 & 0.568 |
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

### Pre-trained model Performance on SciFact Dataset
| Number | Pre-trained Model | MRR | P@1 | R@5 |
| :---         | :---      | :---: | :---: | :---: |
1 | bert-base-uncased | 0.618 | 0.515 | 0.724 |
2 | bert-base-cased | 0.607 | 0.505 | 0.709 |
3 | roberta-base & 0.578 | 0.476 | 0.681 |
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

### Pre-trained model Performance on NQ Dataset
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
In the model ranking stage, we randomly selected 20 models from the aforementioned pool of 50 candidates:
| Number | Pre-trained Model 
| :---   | :---: |
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
