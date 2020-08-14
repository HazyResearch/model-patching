# Model Patching: Closing the Subgroup Performance Gap with Data Augmentation
![Model patching pipeline](assets/model_patching.jpg "Model patching pipeline")
> **Model Patching: Closing the Subgroup Performance Gap with Data Augmentation**\
> Karan Goel*, Albert Gu*, Sharon Li, Christopher Ré\
> Stanford University\
> Arxiv Preprint



> **Abstract.** Classifiers in machine learning are often brittle when deployed. 
Particularly concerning are models with inconsistent performance on specific _subgroups_ of a class, 
e.g., exhibiting disparities in skin cancer classification in the presence or absence of a spurious bandage.
To mitigate these performance differences, 
we introduce _model patching_, 
a two-stage framework for improving robustness that encourages the model to be invariant to subgroup differences, and focus on class information shared by subgroups.
Model patching
first models subgroup features within a class and learns semantic transformations between them,
and then trains a classifier with data augmentations that deliberately manipulate subgroup features.
We instantiate model patching with CAMEL, which (1) uses a CycleGAN to learn the intra-class, inter-subgroup augmentations, and (2) balances subgroup performance using a theoretically-motivated subgroup consistency regularizer, accompanied by a new robust ohttps://app.wandb.ai/hazy-research/celeba/runs/xs4l2gi0?workspace=user-alberthttps://app.wandb.ai/hazy-research/celeba/runs/xs4l2gi0?workspace=user-albertbjective.
We demonstrate CAMEL's effectiveness on 3 benchmark datasets, with reductions in robust error of up to 33\% relative to the best baseline. Lastly, CAMEL successfully patches a model that fails due to spurious features on a real-world skin cancer dataset. 


## 4-Minute Explanation Video
Click the figure to watch this short video explaining our work.

[![IMAGE ALT TEXT HERE](assets/model_patching_youtube.png)](https://www.youtube.com/watch?v=IqRh-SVNl-c)

## Setup

Create a Python environment and install the dependencies.
```bash
# Clone the repository
git clone https://github.com/HazyResearch/model-patching.git
cd model-patching/

# Create a Conda environment
conda create -n model_patching python=3.6
conda activate model_patching

# Install dependencies
pip install -r requirements.txt
```

Download datasets from the Google Cloud Bucket,
```bash
wget https://storage.googleapis.com/model-patching/celeba_tfrecord_128.zip
wget https://storage.googleapis.com/model-patching/waterbirds_tfrecord_224.zip
wget https://storage.googleapis.com/model-patching/isic_tfrecord_224_v2.zip
```

We also include a release of the MNIST-Correlation dataset that we created for convenience,
```bash
wget https://storage.googleapis.com/model-patching/mnist_correlation_npy.zip
```


## Stage 1: Learning Augmentations with a Subgroup Transformation Model

For Stage 1 with CycleGAN Augmented Model Patching (CAMEL), we include configs for training CycleGAN models. Typically, we train one model per class, where the model learns transformations between the subgroups of the class. This is not necessary, and you could alternatively train e.g. a single StarGAN model for all classes and subgroups in your setting.  

#### Training from Scratch
It is not necessary to train these models to reproduce our results, and you can just reuse the augmented datasets that we provide to skip this step.
```bash
# Training a single CycleGAN model for MNIST-Correlation
python -m augmentation.methods.cyclegan.train --config augmentation/configs/stage-1/mnist-correlation/config.yaml

# Training CycleGAN models on Waterbirds
python -m augmentation.methods.cyclegan.train --config augmentation/configs/stage-1/waterbirds/config-1.yaml
python -m augmentation.methods.cyclegan.train --config augmentation/configs/stage-1/waterbirds/config-2.yaml

# Training CycleGAN models on CelebA-Undersampled
python -m augmentation.methods.cyclegan.train --config augmentation/configs/stage-1/celeba/config-1.yaml
python -m augmentation.methods.cyclegan.train --config augmentation/configs/stage-1/celeba/config-2.yaml
```

#### Reusing our CycleGANs
We provide .tfrecord datasets that can be used to replicate the outputs of Stage 1,
```bash
wget https://storage.googleapis.com/model-patching/stage-1-tfrecords.zip
```

We also include links to the logs for the CycleGAN models on Weights and Biases,
```bash
# CycleGAN model for MNIST-Correlation
https://app.wandb.ai/hazy-research/mnist-correlation/runs/hfyg9z4t

# CycleGAN model for Waterbirds
https://app.wandb.ai/hazy-research/waterbirds/runs/vla2y0m7
https://app.wandb.ai/hazy-research/waterbirds/runs/5f2gmy7w

# CycleGAN model for CelebA-Undersampled
https://app.wandb.ai/hazy-research/celeba/runs/xbqhzkx3
https://app.wandb.ai/hazy-research/celeba/runs/xs4l2gi0
```


## Stage 2: Training an End-Model

For Stage 2, we include configs for training classifiers with consistency regularization and Group DRO [Sagawa et al., ICLR 2020], as well as standard ERM training. 

```bash
# Training {CAMEL, Group DRO, ERM}
# on {MNIST-Correlation, Waterbirds, CelebA-Undersampled}
python -m augmentation.methods.robust.train \
  --config augmentation/configs/stage-2/{mnist-correlation,waterbirds,celeba}/{camel,gdro,erm}/config.yaml
```

<!-- ## Citation
If you use our codebase, or otherwise found our work valuable, please cite us
```
@inproceedings{goelmodelpatching,
  title={Model Patching: Closing the Subgroup Performance Gap with Data Augmentation},
  author={Karan Goel and Albert Gu and Sharon Li and Christopher Re},
  booktitle={Arxiv},
  year={2020}
}
``` -->