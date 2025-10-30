# CuMPerLay: Learning Cubical Multiparameter Persistence Vectorizations

Official implementation of "CuMPerLay: Learning Cubical Multiparameter Persistence Vectorizations".

We present CuMPerLay, a novel differentiable vectorization layer that enables the integration of Cubical Multiparameter Persistence (CMP) into deep learning pipelines.
Please refer to [our paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Korkmaz_CuMPerLay_Learning_Cubical_Multiparameter_Persistence_Vectorizations_ICCV_2025_paper.pdf) and [project page](https://circle-group.github.io/cumperlay/) for more details on the project and the methodology followed.

## Installation

You can install all requirements using [environment.yml](./environment.yml) and the [cmp](https://github.com/circle-group/cmp) repository containing our C++/CUDA implementations as follows:
```bash
conda env create -f environment.yml
conda activate cumperlay
# Install the cmp package
pip install --no-build-isolation "git+https://github.com/circle-group/cmp.git#egg=cmp"
```
Note that we tested the method with gcc-12/g++-12 and CUDA 12.2 (may work with other compatible PyTorch/CUDA and gcc/g++ versions). Please see [cmp](https://github.com/circle-group/cmp) repository for installation instruction on our cmp package. If you need to install the library without installing any system-wide packages, you can utilize the conda environment to install the required compilers before the `pip install` step:
```bash
conda activate cumperlay
# Install gcc/g++ 12.2
conda install -c conda-forge gcc==12.2 gxx==12.2
export CC="gcc" CXX="g++"
# Install the cmp package
pip install --no-build-isolation "git+https://github.com/circle-group/cmp.git#egg=cmp"
```

Please see [environment.yml](./environment.yml) for all package requirements. `torch >= 2.0` is required and we have tested the method with Pytorch 2.3.0 and CUDA 12.2.

## Usage

To train the model (as an example to run the TopoSwin-MP model on the `ISIC` dataset with the default parameters, updating data.dataroot config parameter in [config/isic-swin-silv2-inpd-multreg-enp.yaml](./config/isic-swin-silv2-inpd-multreg-enp.yaml) to the correct dataroot), you can use the following command:

```bash
OMP_NUM_THREADS=1 python -u train.py --config config/isic-swin-silv2-inpd-multreg-enp.yaml --train --gpu 0
```

Please refer to the [config](./config/) folder for run configurations and [train.py](./train.py) for running experiments. You can refer to the same training file for evaluating the checkpoints of a trained model. For multiclass/multilabel related configuration parameters, you can check the [Pytorch Lightning Trainer](./lmp/trainers/cls_trainer.py).

For dataset configurations, please see [lmp/data](./lmp/data).

Further instructions on usage and demos will be available soon.

## Relevant Code

Please see [lmp/nn/networks/lmp_swin_v2.py](./lmp/nn/networks/lmp_swin_v2.py) for the combined SwinV2 model with our CumPerLay layer, [lmp/nn/persistence/cubical_torch.py](./lmp/nn/persistence/cubical_torch.py) for the persistence implementation (that uses the cubical_ph package from cuda-cumper folder), [lmp/nn/representations/silhouette_v2.py](./lmp/nn/representations/silhouette_v2.py) for our vectorization.

## Data

The dataset used in this project is not included in the repository.
You can download the datasets from [ISIC 2018](https://challenge.isic-archive.com/data/#2018) (task3), [CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) (Cropped Images) and [Glaucoma](https://data.mendeley.com/datasets/s9bfhswzjb/1). For the segmentation datasets, you need to download them using TorchVision beforehand, further instructions on this will be available, but you can refer to [lmp/data/pascal_voc_sbd.py](./lmp/data/pascal_voc_sbd.py).

## Citation

If you use this codebase/library, or otherwise found our work valuable, please cite:
```
@InProceedings{Korkmaz_2025_CuMPerLay,
    author    = {Korkmaz, Caner and Nuwagira, Brighton and Coskunuzer, Baris and Birdal, Tolga},
    title     = {CuMPerLay: Learning Cubical Multiparameter Persistence Vectorizations},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {27084-27094}
}
```

## Acknowledgements

- https://github.com/threestudio-project/threestudio for parts of the codebase structure
