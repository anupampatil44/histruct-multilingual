# HiStruct+ : Improving Extractive Text Summarization with Hierarchical Structure Information

## Abstract
Transformer-based language models usually treat texts as linear sequences. However, most texts also have an inherent hierarchical structure, i.\,e., parts of a text can be identified using their position in this hierarchy. In addition, section titles usually indicate the common topic of their respective sentences. We propose a novel approach to formulate, extract, encode and inject hierarchical structure information explicitly into an extractive summarization model based on a pre-trained, encoder-only Transformer language model (HiStruct+ model), which improves SOTA
ROUGEs for extractive summarization on PubMed and arXiv substantially. Using various experimental settings on three datasets (i.\,e., CNN/DailyMail, PubMed and arXiv), our HiStruct+ model outperforms a strong baseline collectively, which differs from our model only in that the hierarchical structure information is not injected.  It is also observed
that the more conspicuous hierarchical structure the dataset has, the larger improvements
our method gains. The ablation study demonstrates that the hierarchical position information is the main contributor to our model’s SOTA performance.

## Model architecture

![overview_CR_with_sidenotes](https://user-images.githubusercontent.com/28861305/158410915-7243ba5a-5171-4efe-a566-1a788c7de3ad.png)


## Installation

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

## Training and evaluation

See `src/train.py`.

## Downloads
- the [raw CNN/DailyMail](https://cs.nyu.edu/~kcho/DMQA/) dataset
- the [raw PubMed & arXiv](https://github.com/armancohan/long-summarization) datasets
- the preprocessed CNN/DailyMail data containing hierarchical structure information
- the preprocessed PubMed data containing hierarchical structure information
- the preprocessed arXiv data containing hierarchical structure information
- the [pre-defined dictionaries](https://drive.google.com/file/d/1fSHK6r9QIPXNG58p0kzdFIWeRpFcdlqn/view?usp=sharing) of the typical section classes and the corresponding alternative section titles 
- our best-performed HiStruct+RoBERTa model on CNN/DailyMail
- our best-performed HiStruct+Longformer model on PubMed
- our best-performed HiStruct+Longformer model on arXiv
