# 3Doodle
<div style="text-align:center">
<img src="assets/teaser.jpg" alt="teaser image"/>
</div>
<br/>
Official implementation of <b>3Doodle: Compact Abstraction of Objects with 3D Strokes, SIGGRAPH (ACM Transactions on Graphics) 2024</b>.

### [Project Page](https://changwoonchoi.github.io/3Doodle) | [Paper](https://dl.acm.org/doi/10.1145/3658156) | [arXiv](https://arxiv.org/abs/2402.03690)
___
## Installation
Create a conda environment and install required libraries.
```
conda env create -f environment.yml
# set your own CUDA path in the activate.sh file
source activate.sh
```

Install other dependencies and this package.
```
# Install CLIP
pip install git+https://github.com/openai/CLIP.git

# Install Open3D
pip install open3d

# Install robust loss
pip install git+https://github.com/jonbarron/robust_loss_pytorch

# Install diffvg
git clone https://github.com/BachiLi/diffvg
cd diffvg
git submodule update --init --recursive
python setup.py install
cd ..

# Install pytorch3d
pip install fvcore
# Replace the version str (py39...pyt1121) with your corresponding python and torch version
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1121/download.html

# Remaining dependencies
pip install -r requirements.txt

# Install this package
pip install -e .
```
## Dataset
You can download our dataset and SfM point clouds from this [link](https://drive.google.com/drive/folders/1tTVVd78TAJIT6GJcfrfKIWN_EyR92FmZ).


```
# Bézier curve only
python train.py --config configs/nerf_synthetic/mic.yaml -ep sanity_check -eg bezier -en mic
# with superquadrics
python train.py --config configs/3doodle_sq/snowman.yaml --stages all -ep sanity_check -eg all -en snowman
```

## Render test images
```
# Bézier curve only
python test_render.py --config logs/sanity_check/bezier/mic_0/config.yaml -ck logs/sanity_check/bezier/mic_0/best.ckpt
# with superquadrics
python test_render.py --config logs/sanity_check/all/snowman_0/config.yaml --stages all -ck logs/sanity_check/all/snowman_0/best.ckpt
```

## Real-time interactive viewer
```
# Bézier curve only
python viewer.py --config logs/sanity_check/bezier/mic_0/config.yaml -ck logs/sanity_check/bezier/mic_0/best.ckpt
# with superquadrics
python viewer.py --config logs/sanity_check/all/snowman_0/config.yaml --stages all -ck logs/sanity_check/all/snowman_0/best.ckpt
```

## Acknowledgements
We thank our friend [Clément Jambon](https://clementjambon.github.io/) for helping with the implementation of the interactive viewer. We would like to clarify that the source codes related to the interactive viewer are borrowed from his work.

## Citation
Please cite as below if you find this paper and repository are helpful to you:
```
@article{10.1145/3658156,
    author = {Choi, Changwoon and Lee, Jaeah and Park, Jaesik and Kim, Young Min},
    title = {3Doodle: Compact Abstraction of Objects with 3D Strokes},
    year = {2024},
    issue_date = {July 2024},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {43},
    number = {4},
    issn = {0730-0301},
    url = {https://doi.org/10.1145/3658156},
    doi = {10.1145/3658156},
    journal = {ACM Trans. Graph.},
    month = {jul},
    articleno = {107},
    numpages = {13},
    keywords = {3D sketch lines, 3D strokes, differentiable rendering}
}
          
```
