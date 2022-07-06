Note: Using https://huggingface.co/markussagen/xlm-roberta-longformer-base-4096 in place of https://huggingface.co/allenai/longformer-base-4096


# Instructions to (attempt to) run script for multilingual setup:

- Run these 2 additionally in your histruct conda env:
1. pip3 install indic-nlp-library
2. pip3 install --upgrade protobuf==3.20.0

## For preprocessing, Run the following commands externally in your terminal for preprocessing the data:<br>
(train test and val sets should be present as .txt files)

1) python3 histruct/src/preprocess_wiki.py -mode merge_data_splits -dataset wiki -raw_path data_hiwiki/data_hiwiki_raw -save_path data_hiwiki/data_hiwiki_splitted/hiwiki  -log_file data_hiwiki/hiwiki_prepro_merge_data_splits.log

2) python3 histruct/src/preprocess_wiki.py -mode format_to_histruct -dataset wiki -base_LM xlm-roberta-base -raw_path data_hiwiki/data_hiwiki_splitted -save_path data_hiwiki/data_hiwiki_roberta  -log_file data_hiwiki/hiwiki_prepro_fth_roberta.log -summ_size 0 -n_cpus 1 -obtain_tok_se false

3) python3 histruct/src/preprocess_wiki.py -mode obtain_section_names -dataset wiki -raw_path data_hiwiki/data_hiwiki_raw  -save_path data_hiwiki/data_hiwiki_raw -log_file data_hiwiki/hiwiki_prepro_osn.log 

4) python3 histruct/src/preprocess_wiki.py -mode encode_section_names -base_LM xlm-roberta-longformer-base-4096 -dataset wiki -sn_embed_comb_mode sum -raw_path data_hiwiki/data_hiwiki_raw -save_path data_hiwiki/data_hiwiki_raw -log_file data_hiwiki/hiwiki_prepro_esn.log 


## For training, edit the following python command's arguments to run as per your requirements:


python3 histruct/src/train.py -task ext -mode train -base_LM xlm-roberta-longformer-base-4096 -add_tok_struct_emb false -tok_pos_emb_type learned_all -tok_se_comb_mode sum -add_sent_struct_emb true -sent_pos_emb_type learned_all -sent_se_comb_mode sum -ext_dropout 0.1 -model_path models/hiwiki_hs_longformerB_s_la_sum_bs500ac2ws10000ts100000_mp28000mns1300_law1024_F-finetune_T-globatt_sn-longformerB-sum_1gpu -batch_size 2 -accum_count 2 -warmup_steps 100 -train_steps 1000 -lr 0.002 -data_path data_hiwiki/data_hiwiki_roberta/hiwiki -temp_dir temp -visible_gpus 0 -report_every 50 -save_checkpoint_steps 1000 -max_pos 28000 -ext_layers 2 -without_sent_pos false -max_nsent 1300 -max_npara 0 -max_nsent_in_para 0 -para_only false -finetune_bert false -use_global_attention true -local_attention_window 1024 -section_names_embed_path data_hiwiki/data_hiwiki_raw/section_names_embed_xlmR_sum.pt -seed 666

# accum_count set to 1
