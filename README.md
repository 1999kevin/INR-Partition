# Partition Speeds Up Learning Implicit Neural Representations Based on Exponential-Increase Hypothesis

[ICCV2023] Partition Speeds Up Learning Implicit Neural Representations Based on Exponential-Increase Hypothesis. 

Ke Liu, Feng Liu, Haishuai Wang, Ning Ma, Jiajun Bu, Bo Han 

[[paper](https://arxiv.org/abs/2310.14184)]

<p align="center">
<img src= pic/motivation_new.jpg width="1000" height="300"/>
</p>


## Abstract
Implicit neural representations (INRs) aim to learn a continuous function (i.e., a neural network) to represent an image, where the input and output of the function are pixel coordinates and RGB/Gray values, respectively. However, images tend to consist of many objects whose colors are not perfectly consistent, resulting in the challenge that image is actually a discontinuous piecewise function and cannot be well estimated by a continuous function. In this paper, we empirically investigate that if a neural network is enforced to fit a discontinuous piecewise function to reach a fixed small error, the time costs will increase exponentially with respect to the boundaries in the spatial domain of the target signal. We name this phenomenon the exponential-increase hypothesis. Under the exponential-increase hypothesis, learning INRs for images with many objects will converge very slowly. To address this issue, we first prove that partitioning a complex signal into several sub-regions and utilizing piecewise INRs to fit that signal can significantly speed up the convergence. Based on this fact, we introduce a simple partition mechanism to boost the performance of two INR methods for image reconstruction: one for learning INRs, and the other for learning-to-learn INRs. In both cases, we partition an image into different sub-regions and dedicate smaller networks for each part. In addition, we further propose two partition rules based on regular grids and semantic segmentation maps, respectively. Extensive experiments validate the effectiveness of the proposed partitioning methods in terms of learning INR for a single image (ordinary learning framework) and the learning-to-learn framework.


## Get Start
Please install the requirements via:
```
python install -r requirements.txt
```


Please see the instructions in directory ```motivation_fit``` to reproduce experiments for learning INRs for 1D and 2D synthetic signals.

Please see the instructions in directory ```single_image_fit``` to run experiments for learning INRs.

Please see the instructions in directory ```MAML_fit``` to run experiments for learning-to-learn INRs.


## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{liu2023partition,
  title={Partition Speeds Up Learning Implicit Neural Representations Based on Exponential-Increase Hypothesis},
  author={Liu, Ke and Liu, Feng and Wang, Haishuai and Ma, Ning and Bu, Jiajun and Han, Bo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5474--5483},
  year={2023}
}
```


## Acknowledgement

We appreciate the following github repos a lot for their valuable code base:

https://github.com/vsitzmann/siren

https://colab.research.google.com/github/vsitzmann/metasdf/blob/master/MetaSDF.ipynb

