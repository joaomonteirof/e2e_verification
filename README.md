# E2E verification

Joint metric learning and distance metric learning for end-to-end verification.

## Requirements

```
Python >= 3.6
pytorch >= 1.2
torchvision >=0.4.1
Scikit-learn >=0.19
tqdm
h5py
nevergrad (for hyperparameters search only)
tensorboard
```

## Cifar-10 and Mini-ImageNet

### Cifar will be downloaded automatticaly by Torchvision. Mini-Imagenet can be downloaded from: https://mtl.yyliu.net/download/

### Example for running the train script:

```
python train.py \
--checkpoint-path /path/to/cp \
--save-every 4 \
--data-path /data/train \
--valid-data-path /data/valid/ \
--n-workers 8 \
--seed 1 \
--model resnet \
--batch-size 128 \
--valid-batch-size 128 \
--lr 1e-2 \
--momentum 0.9 \
--l2 1e-4 \
--epochs 600 \
--hidden-size 350 \
--n-hidden 3 \
--dropout-prob 0.01 \
--softmax am_softmax \
--smoothing 0.05
```

### Example for running the hyperparameter search

The seach will run in a serial manner.

```
python hp_search.py \
--batch-size 128 \
--valid-batch-size 128 \
--model resnet \
--budget 40 \
--checkpoint-path /path/to/cp/ \
--epochs 600 \
--data-path /data/mini-imagenet/train \
--valid-data-path /data/mini-imagenet/val
```

### Evaluation

An evaluation script is provided and it will create all possible trials from available test data. Example:

```
python eval.py
--data-path /data/test/
--model resnet
--cp-path /path/to/cp/trained_model.pt
--out-path /results/scores.out
```

## VoxCeleb

### Prepare data

Voxceleb can be downloaded from http://www.robots.ox.ac.uk/~vgg/data/voxceleb/ .

Ous training scripts expect data in a hdf file in a specific format: one group per speaker and speech features stored as datasets (one dataset per audio file) within corresponding group. Data preparation scripts are provided. However, features in [Kaldi](https://kaldi-asr.org/) format are exepected. We utilized the first step of the recipe in https://github.com/kaldi-asr/kaldi/tree/master/egs/voxceleb to do that.

After extracting features with Kaldi, hdfs can be prepared with data_prep.py. Arguments:

```
--path-to-data        Path to scp files with features
--data-info-path      Path to spk2utt and utt2spk
--spk2utt             Path to spk2utt
--utt2spk             Path to utt2spk
--path-to-more-data   Path to extra scp files with features
--more-data-info-path Path to spk2utt and utt2spk
--more-spk2utt Path   Path to spk2utt
--more-utt2spk Path   Path to utt2spk
--out-path Path       Path to output hdf file
--out-name Path       Output hdf file name
--min-recordings      Minimum number of train recordings for speaker to be included
```

Train and validation hdfs are expected.

### Train a model

Once data is pre-processed and stored into hdf files, train models with train.py. Example:

```
python train.py \
--checkpoint-path /path/to/cp \
--save-every 4 \
--train-hdf-file /data/train.hdf \
--valid-hdf-file /data/test.hdf \
--workers 8 \
--seed 1 \
--model TDNN \
--max-gnorm 20.0 \
--batch-size 24 \
--valid-batch-size 32 \
--lr 1.5 \
--momentum 0.85 \
--l2 1e-5 \
--warmup 2000 \
--epochs 100 \
--ncoef 30 \
--n-frames 800 \
--latent-size 512 \
--hidden-size 256 \
--n-hidden 4 \
--dropout-prob 0.01 \
--softmax am_softmax \
--smoothing 0.05 \
--logdir /path/to/logs/ \
```

### Hyperparameters tuning

Serial search over the hyperparameter grid would be impractical for VoxCeleb. We thus provide scripts to search in parallel over slurm or sge clusters. Example:

```
python hp_search_slurm.py \
--train-hdf-file /data/train.hdf \
--valid-hdf-file /data/test.hdf \
--sub-file run_hp_slurm.sh \
--temp-folder temp_vox \
--workers 6 \
--batch-size 24 \
--valid-batch-size 24 \
--epochs 6 \
--budget 45 \
--model TDNN \
--ncoef 30 \
--hp-workers 15 \
--checkpoint-path /path/to/cp \
--logdir /path/to/logs/
```

The sub-file consists of a sge/slurm submission script which will executed from within the search script. In the example, run_hp_slurm.sh should look like the following:

```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=12000M
#SBATCH --time=6-00:00:00
#SBATCH --job-name=hp_search
#SBATCH --wait

python train_hp.py \
--lr ${1} \
--l2 ${2} \
--momentum ${3} \
--smoothing ${4} \
--warmup ${5} \
--latent-size ${6} \
--n-hidden ${7} \
--hidden-size ${8} \
--n-frames ${9} \
--model ${10} \
--ndiscriminators ${11} \
--rproj-size ${12} \
--ncoef ${13} \
--dropout-prob ${14} \
--epochs ${15} \
--batch-size ${16} \
--valid-batch-size ${17} \
--workers ${18} \
--cuda ${19} \
--train-hdf-file ${20} \
--valid-hdf-file ${21} \
--out-file ${22} \
--checkpoint-path ${23} \
--cp-name ${24} \
--softmax ${25} \
--max-gnorm ${26} \
--logdir ${27}
```

### Evaluation

End-to-end evaluation can be performed with eval.py. In this case we consume directly the kaldi files from pre-processed test data. The list of trials has to be further provided consisting of a text file with a trial per line, and a trial corresponds to three colums: enroll utterance id, test utterance id, target/non-target. Example:

```
python eval.py
--test-data /data/test/feats.scp
--trials-path /data/test/trials
--cp-path /path/to/cp/trained_model.pt
--model TDNN
--out-path /results/scores.out
```

We further provide a script called embed.py to compute and save representations of a set of recordings so that downstream classifiers can be trained such as PLDA.
