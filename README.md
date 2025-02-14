# HiStruct+ : Improving Extractive Text Summarization with Hierarchical Structure Information

Note: Using https://huggingface.co/markussagen/xlm-roberta-longformer-base-4096 in place of https://huggingface.co/allenai/longformer-base-4096


# Instructions to (attempt to) run script for multilingual setup:
A small part of our data can be accessed [here](https://drive.google.com/file/d/1jdx3Itowup8Y9WuJ6EfYtv0hINzQGwCl/view?usp=sharing). Please uncompress the folder outstude the histruct-multilingual directory.
- Run these 2 steps additionally in your histruct conda env:
1. pip3 install indic-nlp-library
2. pip3 install --upgrade protobuf==3.20.0

## For preprocessing, run the following commands externally in your terminal for preprocessing the data:<br>
(train test and val sets should be present as .txt files)

1) python3 histruct-multilingual/src/preprocess_wiki.py -mode merge_data_splits -dataset wiki -raw_path data_hiwiki/data_hiwiki_raw -save_path data_hiwiki/data_hiwiki_splitted/hiwiki  -log_file data_hiwiki/hiwiki_prepro_merge_data_splits.log

2) python3 histruct-multilingual/src/preprocess_wiki.py -mode format_to_histruct -dataset wiki -base_LM xlm-roberta-base -raw_path data_hiwiki/data_hiwiki_splitted -save_path data_hiwiki/data_hiwiki_roberta  -log_file data_hiwiki/hiwiki_prepro_fth_roberta.log -summ_size 0 -n_cpus 1 -obtain_tok_se false

3) python3 histruct-multilingual/src/preprocess_wiki.py -mode obtain_section_names -dataset wiki -raw_path data_hiwiki/data_hiwiki_raw  -save_path data_hiwiki/data_hiwiki_raw -log_file data_hiwiki/hiwiki_prepro_osn.log 

4) python3 histruct-multilingual/src/preprocess_wiki.py -mode encode_section_names -base_LM xlm-roberta-longformer-base-4096 -dataset wiki -sn_embed_comb_mode sum -raw_path data_hiwiki/data_hiwiki_raw -save_path data_hiwiki/data_hiwiki_raw -log_file data_hiwiki/hiwiki_prepro_esn.log 


## For training, edit the following python command's arguments to run as per your requirements:


python3 histruct-multilingual/src/train.py -task ext -mode train -base_LM xlm-roberta-longformer-base-4096 -add_tok_struct_emb false -tok_pos_emb_type learned_all -tok_se_comb_mode sum -add_sent_struct_emb true -sent_pos_emb_type learned_all -sent_se_comb_mode sum -ext_dropout 0.1 -model_path models/hiwiki_hs_longformerB_s_la_sum_bs500ac2ws10000ts100000_mp28000mns1300_law1024_F-finetune_T-globatt_sn-longformerB-sum_1gpu -batch_size 2 -accum_count 2 -warmup_steps 100 -train_steps 1000 -lr 0.002 -data_path data_hiwiki/data_hiwiki_roberta/hiwiki -temp_dir temp -visible_gpus 0 -report_every 50 -save_checkpoint_steps 1000 -max_pos 28000 -ext_layers 2 -without_sent_pos false -max_nsent 1300 -max_npara 0 -max_nsent_in_para 0 -para_only false -finetune_bert false -use_global_attention true -local_attention_window 1024 -section_names_embed_path data_hiwiki/data_hiwiki_raw/section_names_embed_xlmR_sum.pt -seed 666

# accum_count set to 1

## Abstract
Transformer-based language models usually treat texts as linear sequences. However, most texts also have an inherent hierarchical structure, i.\,e., parts of a text can be identified using their position in this hierarchy. In addition, section titles usually indicate the common topic of their respective sentences. We propose a novel approach to formulate, extract, encode and inject hierarchical structure information explicitly into an extractive summarization model based on a pre-trained, encoder-only Transformer language model (HiStruct+ model), which improves SOTA
ROUGEs for extractive summarization on PubMed and arXiv substantially. Using various experimental settings on three datasets (i.\,e., CNN/DailyMail, PubMed and arXiv), our HiStruct+ model outperforms a strong baseline collectively, which differs from our model only in that the hierarchical structure information is not injected.  It is also observed
that the more conspicuous hierarchical structure the dataset has, the larger improvements
our method gains. The ablation study demonstrates that the hierarchical position information is the main contributor to our model’s SOTA performance.

## Model architecture

![](https://user-images.githubusercontent.com/28861305/158413092-657c34db-51c2-41d2-89de-7dcd2663d2ea.png)

Figure 1: Architecture of the HiStruct+ model. The model consists of a base TLM for sentence encoding and two stacked inter-sentence Transformer layers for hierarchical contextual learning with a sigmoid classifier for extractive summarization. The two blocks shaded in light-green are the HiStruct injection components

## ROUGE results on PubMed and arXiv

| Dataset | HiStruct+ Model                      | ROUGE1 | ROUGE 2 | ROUGE L |  
|---------|--------------------------------------|--------|---------|---------|
| PubMed  | HiStruct+ Longformer-base (15k tok.) | 46.59  | 20.39   | 42.11   |   
| arXiv   | HiStruct+ Longformer-base (28k tok.) | 45.22  | 17.67   | 40.16   |   





## Env. Setup

Requirements: Python 3.8 and Conda

```bash
# Create environment
conda create -n py38_pt18 python=3.8
conda activate py38_pt18

# Install dependencies
pip3 install -r requirements.txt

# Install pytorch
conda install pytorch==1.8.0 torchvision cudatoolkit=10.1 -c pytorch

# Setup pyrouge
pyrouge_set_rouge_path pyrouge/rouge/tools/ROUGE-1.5.5/
conda install -c bioconda perl-xml-parser 
conda install -c bioconda perl-lwp-protocol-https
conda install -c bioconda perl-db-file
```
## Preprocessing of data
#### NOTE: Data preprocessing would take some time. It is recommended to use the preprocessed data if you experiment with CNN/DailyMail, PubMed or arXiv. (see links in Downloads).

- obtain HiStruct information 
- obatin gold labels for extractive summarization (ORACLE)
- tokenize texts with the corresponding tokenizer

#CNN/DailyMail
```bash
#Make sure that you have the standford-corenlp toolkit downloaded
#export CLASSPATH=stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar
#raw data saved in data_cnndm/data_cnndm_raw

# (1). tokenize the sentences and paragraphs respectively 
# output files: data_cnndm/data_cnndm_raw_tokenized_sent, data_cnndm/data_cnndm_raw_tokenized_para
python histruct/src/preprocess.py -mode tokenize -dataset cnndm  -raw_path data_cnndm/data_cnndm_raw -tok_sent_path data_cnndm/data_cnndm_raw_tokenized_sent -tok_para_path data_cnndm/data_cnndm_raw_tokenized_para -log_file data_cnndm/cnndm_prepro_tokenize.log

# (2). extract HiStruct info
# output path: data_cnndm/data_cnndm_raw_tokenized_histruct
python histruct/src/preprocess.py  -dataset cnndm -mode extract_histruct_items -histruct_path data_cnndm/data_cnndm_raw_tokenized_histruct  -tok_sent_path data_cnndm/data_cnndm_raw_tokenized_sent -tok_para_path data_cnndm/data_cnndm_raw_tokenized_para -lower true -log_file data_cnndm/cnndm_prepro_extract_histruct_items.log

# (3). merge data splits for training, validation and testing
#make sure that the mapping files are in the folder 'urls'
python histruct/src/preprocess.py -dataset cnndm -mode merge_data_splits -raw_path data_cnndm/data_cnndm_raw_tokenized_histruct -save_path data_cnndm/data_cnndm_splitted/cnndm -map_path urls -log_file data_cnndm/cnndm_prepro_merge_data_splits.log

# (4). convcert format for HiStruct+ training, perpare gold labels using ORACLE
#base_LM: the tokenizer used, should be consistent with the base TLM involved in the summarization model, choose from [roberta-base, bert-base]
#summ_size: how many sentences should be included in ORACLE summaries, default:0, no specific limitation
#obtain_tok_se: wehther to obatin token-level struture vectors (see Appendix A.5 in the paper), default: false 
python histruct/src/preprocess.py -mode format_to_histruct -dataset cnndm -base_LM roberta-base -raw_path data_cnndm/data_cnndm_splitted -save_path data_cnndm/data_cnndm_roberta  -log_file data_cnndm/cnndm_prepro_fth_roberta.log -summ_size 0 -n_cpus 1 -obtain_tok_se false
```

#PubMed
```bash
#raw data saved in data_pubmed/data_pubmed_raw 

# (1). merge data splits for training, validation and testing
python histruct/src/preprocess.py -mode merge_data_splits -dataset pubmed -raw_path data_pubmed/data_pubmed_raw -save_path data_pubmed/data_pubmed_splitted/pubmed  -log_file data_pubmed/pubmed_prepro_merge_data_splits.log

# (2). convcert format for HiStruct+ training, perpare gold labels using ORACLE
#-base_LM: the tokenizer used, should be consistent with the base TLM involved in the summarization model, Longformer tokenizer is identical to roberta-base tokenizer
#-summ_size: how many sentences should be included in ORACLE summaries, default:0, no specific limitation
#-obtain_tok_se: wehther to obatin token-level struture vectors (see Appendix A.5 in the paper), default: false 
python histruct/src/preprocess.py -mode format_to_histruct -dataset pubmed -base_LM roberta-base -raw_path data_pubmed/data_pubmed_splitted -save_path data_pubmed/data_pubmed_roberta  -log_file data_pubmed/pubmed_prepro_fth_roberta.log -summ_size 0 -n_cpus 1 -obtain_tok_se false

# (3). obtain unique section titles from the raw data
python histruct/src/preprocess.py -mode obtain_section_names -dataset pubmed -raw_path data_pubmed/data_pubmed_raw  -save_path data_pubmed/data_pubmed_raw -log_file data_pubmed/pubmed_prepro_osn.log 

# (4). generate section title embeddings (STEs)
# -base_LM: the TLM used, should be consistent with the base TLM involved in the summarization model
# -sn_embed_comb_mode: how to convert the last hidden states at every token positions to a single vector, sum them up by default
python histruct/src/preprocess.py -mode encode_section_names -base_LM longformer-base-4096 -dataset pubmed -sn_embed_comb_mode sum -raw_path data_pubmed/data_pubmed_raw -save_path data_pubmed/data_pubmed_raw -log_file data_pubmed/pubmed_prepro_esn.log 

# (5). generate classified section title embeddings (classified STEs)
# -base_LM: the TLM used, should be consistent with the base TLM involved in the summarization model
# -sn_embed_comb_mode: how to convert the last hidden states at every token positions to a single vector, sum them up by default
# -section_names_embed_path: the path to the original STE which is generated in the step (4)
# -section_names_cls_file: the predefined dictionary of typical section title classes and the in-class section titles
python histruct/src/preprocess.py -mode encode_section_names_cls -base_LM longformer-base-4096 -dataset pubmed -sn_embed_comb_mode sum -raw_path data_pubmed/data_pubmed_raw -save_path data_pubmed/data_pubmed_raw -log_file data_pubmed/pubmed_prepro_esnc.log  -section_names_embed_path data_pubmed/data_pubmed_raw/section_names_embed_longformerB_sum.pt -section_names_cls_file pubmed_SN_dic_8_Added.json
```

#arXiv
```bash
#raw data saved in data_arxiv/data_arxiv_raw 

# (1). merge data splits for training, validation and testing
python histruct/src/preprocess.py -mode merge_data_splits -dataset arxiv -raw_path data_arxiv/data_arxiv_raw -save_path data_arxiv/data_arxiv_splitted/arxiv -log_file data_arxiv/arxiv_prepro_merge_data_splits.log

# (2). convcert format for HiStruct+ training, perpare gold labels using ORACLE
#-base_LM: the tokenizer used, should be consistent with the base TLM involved in the summarization model, Longformer tokenizer is identical to roberta-base tokenizer
#-summ_size: how many sentences should be included in ORACLE summaries, default:0, no specific limitation
#-obtain_tok_se: wehther to obatin token-level struture vectors (see Appendix A.5 in the paper), default: false 
python histruct/src/preprocess.py -mode format_to_histruct -dataset arxiv -base_LM roberta-base -raw_path data_arxiv/data_arxiv_splitted -save_path data_arxiv/data_arxiv_roberta  -log_file data_arxiv/arxiv_prepro_fth_roberta.log -summ_size 0 -n_cpus 1 -obtain_tok_se false

# (3). obtain unique section titles from the raw data
python histruct/src/preprocess.py -mode obtain_section_names -dataset arxiv -raw_path data_arxiv/data_arxiv_raw -save_path data_arxiv/data_arxiv_raw -log_file data_arxiv/arxiv_prepro_osn.log 

# (4). generate section title embeddings (STEs)
# -base_LM: the TLM used, should be consistent with the base TLM involved in the summarization model
# -sn_embed_comb_mode: how to convert the last hidden states at every token positions to a single vector, sum them up by default
python histruct/src/preprocess.py -mode encode_section_names -base_LM longformer-base-4096 -dataset arxiv -sn_embed_comb_mode sum -raw_path data_arxiv/data_arxiv_raw -save_path data_arxiv/data_arxiv_raw -log_file data_arxiv/arxiv_prepro_esn.log  

# (5). generate classified section title embeddings (classified STEs)
# -base_LM: the TLM used, should be consistent with the base TLM involved in the summarization model
# -sn_embed_comb_mode: how to convert the last hidden states at every token positions to a single vector, sum them up by default
# -section_names_embed_path: the path to the original STE which is generated in the step (4)
# -section_names_cls_file: the predefined dictionary of typical section title classes and the in-class section titles, saved in raw_path
python histruct/src/preprocess.py -mode encode_section_names_cls -base_LM longformer-base-4096 -dataset arxiv -sn_embed_comb_mode sum -raw_path data_arxiv/data_arxiv_raw -save_path data_arxiv/data_arxiv_raw -log_file data_arxiv/arxiv_prepro_esnc.log  -section_names_embed_path data_arxiv/data_arxiv_raw/section_names_embed_longformerB_sum.pt -section_names_cls_file arxiv_SN_dic_10_Added.json
```

## Root directory
#### Please find links in Downloads 
- ./data_cnndm: save the preprocessed CNN/DailyMail data in this folder
- ./data_pubmed: save the preprocessed PubMed data in this folder, the STE and classified STE in ./data_pubmed/data_pubmed_raw
- ./data_arxiv: save the preprocessed arXiv data in this folder, the STE and classified STE in ./data_arxiv/data_arxiv_raw
- ./histruct: unzip the github repository in this folder
- ./models: the trained models are automaticcaly saved in this folder 


## Training and evaluation

See `run_exp_cnndm.py`, `run_exp_pubmed.py` and `run_exp_arxiv.py`. Arguments can be changed in the scripts.

```bash
#run experiments on CNN/DailyMail
python histruct/run_exp_cnndm.py

#run experiments on PubMed
python histruct/run_exp_pubmed.py

#run experiments on arXiv
python histruct/run_exp_arxiv.py
```


## Downloads
- the [raw CNN/DailyMail](https://cs.nyu.edu/~kcho/DMQA/) dataset
- the [raw PubMed & arXiv](https://github.com/armancohan/long-summarization) datasets
- the [preprocessed CNN/DailyMail data](https://github.com/QianRuan/histruct/releases/tag/data_and_models) containing HiStruct information 
- the [preprocessed PubMed data](https://github.com/QianRuan/histruct/releases/tag/data_and_models). containing HiStruct information 
- the [preprocessed arXiv data](https://drive.google.com/file/d/1iJWNZz6hXtKcmlLZ_8AmOHOVnJK1Bx8A/view?usp=sharing) containing HiStruct information 
- PubMed section title embedding, the pre-defined dictionary of the typical section title classes and the in-class section titles, the encoded STE and classified STE (saved in [data_pubmed_raw](https://github.com/QianRuan/histruct/releases/tag/data_and_models))
- arXiv section title embedding, the pre-defined dictionary of the typical section title classes and the in-class section titles, the encoded STE and classified STE (saved in [data_arxiv_raw](https://drive.google.com/file/d/1VARWpuAuPWULzeEd6zMQfNiOmLA1VfFF/view?usp=sharing))
- our best-performed [HiStruct+RoBERTa model on CNN/DailyMail](https://github.com/QianRuan/histruct/releases/tag/data_and_models)
- our best-performed [HiStruct+Longformer model on PubMed](https://github.com/QianRuan/histruct/releases/tag/data_and_models)
- our best-performed [HiStruct+Longformer model on arXiv](https://github.com/QianRuan/histruct/releases/tag/data_and_models)

## Citation
```bash
@inproceedings{ruan-etal-2022-histruct,
    title = "{H}i{S}truct+: Improving Extractive Text Summarization with Hierarchical Structure Information",
    author = "Ruan, Qian  and
      Ostendorff, Malte  and
      Rehm, Georg",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.102",
    doi = "10.18653/v1/2022.findings-acl.102",
    pages = "1292--1308",
}

```

