# CSE 587 - Term Project Code

The outline of this project:

```bash
-- data
|   -- RECCON                 # folder for storing the RECCON dataset. See the readme file under this folder for details.
|   -- process_alpha_BIO.py   # script for generating the 'alpha_BIO' folder
|   -- process_beta_BIO.py    # script for generating the 'beta_BIO' folder
|   -- process_BIO.py         # script for generating the 'RECCON_BIO' folder where all instances are in BIO format.
|   -- process_data.sh        # data processing script

-- pretrained 
|   -- language_model         # folder for storing the pre-trained model, e.g., SpanBERT-base-cased
|   -- word_embedding         # folder for storing the pre-trained word embedding
|   -- download_pretrained.sh # script for preparing the pre-trained resources

-- utils
|   -- cause_detection.py     # dataset class
|   -- metrics_sl_and_qa.py   # evaluation script for both SL and QA formulations
|   -- qa_metrics.py          # evaluation script for QA
|   -- evaluate_squad.py      # official evaluation script for SQuAD
|   -- tool_box.py            # other utils

-- train_graph.py             # main training script

-- eval_graph.py              # main evaluation script

-- models                     # implementation of D-McIN

-- output                     # folder for storing the eval results and model files
```

The main system requirements:

- python == 3.8.0
- pytorch == 1.8.0
- torch_geometric == 2.0.2
- transformers == 4.11.3 

Our infrastructure:

- OS: Ubuntu 18.04 LTS
- CPU: i9-10900KF 
- GPU: GeForce RTX 3090 
- CUDA: 11.1
## Preparations

### 1. Environment Setup

We recommend the readers to use the same environment as us. To set up the Anaconda environments, run the following command:

```bash
conda create -n D-McIN python==3.8.0
conda activate D-McIN
pip install -r requirements.txt
```
Next, install `torch==1.8.0` and `torch_geometric==2.0.2` by using the script below.

```shell
sh setup_env.sh
```

<!-- To install Pytorch, download the wheels from [here](https://download.pytorch.org/whl/torch_stable.html), move the files to the current directory, install Pytorch 1.8.0 from the wheels, for example:

```bash
pip install torch-1.8.0+cu111-cp38-cp38-linux_x86_64.whl
pip install torchvision-0.9.0+cu111-cp38-cp38-linux_x86_64.whl
``` -->

<!-- Before installing torch-geometric, please install other dependencies (e.g., torch-scatter, torch-sparse). Download the wheels from [here](https://data.pyg.org/whl/torch-1.8.0%2Bcu111.html) (according to your environment) and run the following commands:

```bash
pip install torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.0.6-cp38-cp38-linux_x86_64.whl
pip install torch_sparse-0.6.10-cp38-cp38-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl
```

Then, install the torch-geometric:

```bash
pip install torch_geometric==2.0.2
``` -->

### 2. Data Process 

We use the latest benchmark datasets to evaluate our model, namely the [RECCON](https://arxiv.org/pdf/2012.11820.pdf). The datasets are already under the `./data/RECCON/` (see the [readme](https://github.com/RenzeLou/D-McIN/tree/main/data/RECCON) for details). To process the datasets, run the following commands:

```bash
sh ./data/process_data.sh
```

It creates a new folder named `RECCON_BIO`, where all instances are processed into BIO format.

### 3. Pre-trained Resources
Run the following command to download the pre-trained resources:
```bash
sh ./pretrained/download_pretrained.sh
```
After downloading, the `./pretrained` will present the following outline:
```bash
-- pretrained
|   -- language_model
|   |   -- spanbert-base-cased
|   |   |   -- vocab.txt            # vocabulary
|   |   |   -- config.json          # model config
|   |   |   -- pytorch_model.bin    # model parameters

|   -- word_embedding
|   |   -- glove.6B.300d.txt        # word embedding
```
Make sure to prepare all the files above correctly.
## Quick Start
To quickly reproduce the performance we reported in our paper, obtain the model parameters trained by us from [here](https://drive.google.com/file/d/1EGaTvxwHahS5ENmq7RKEsTXr1_mTzdqI/view?usp=sharing), move it to `./output,`, run the following command:
```bash
python eval_graph.py --eval_batch 16 --test_data_path ./data/RECCON_BIO/dailydialog_test.json --test_context_path ./data/RECCON_BIO/dailydialog_test_shared_dialog.json --test_data_path_2 ./data/RECCON_BIO/iemocap_test.json --test_context_path_2 ./data/RECCON_BIO/iemocap_test_shared_dialog.json --model_name McIN --bi_direction --cuda 0 --quick_start
```

The evaluation results will be printed on the console and saved into `./output/test_result_dailydialog.txt` and `./output/test_result_iemocap.txt` as well.

## Train & Evaluate

To train a model by yourself, we recommend the readers to use CPU instead of GPU due to the [nondeterministic operations](https://pytorch.org/docs/stable/notes/randomness.html) of Pytorch:

```bash
python train_graph.py --train_batch 4 --eval_batch 4 --train_data_path ./data/RECCON_BIO/dailydialog_train.json --train_context_path ./data/RECCON_BIO/dailydialog_train_shared_dialog.json --eval_data_path ./data/RECCON_BIO/dailydialog_valid.json --eval_context_path ./data/RECCON_BIO/dailydialog_valid_shared_dialog.json --model_name McIN --bi_direction
```

Then, you can evaluate this model by using the following command:

```bash
python eval_graph.py --eval_batch 16 --test_data_path ./data/RECCON_BIO/dailydialog_test.json --test_context_path ./data/RECCON_BIO/dailydialog_test_shared_dialog.json --test_data_path_2 ./data/RECCON_BIO/iemocap_test.json --test_context_path_2 ./data/RECCON_BIO/iemocap_test_shared_dialog.json --model_name McIN --bi_direction --cuda 0
```
