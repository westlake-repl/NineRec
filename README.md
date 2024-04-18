<p align="center" width="100%">
  <img src='https://github.com/westlake-repl/NineRec/blob/main/assets/NineRec_logo_banner_v6-3.png' width="100%">
</p>

# [TPAMI 2024]  [NineRec: A Benchmark Dataset Suite for Evaluating Transferable Recommendation](https://arxiv.org/pdf/2309.07705.pdf)

<a href="https://arxiv.org/pdf/2309.07705.pdf" alt="arXiv"><img src="https://img.shields.io/badge/arXiv-2309.07705-FAA41F.svg?style=flat" /></a>
<a href="https://ieeexplore.ieee.org/document/10461053" alt="TPAMI"><img src="https://img.shields.io/badge/TPAMI-2024.3373868-%23002FA7.svg?style=flat" /></a> 
![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red) 
![Foundation Model](https://img.shields.io/badge/Task-Foundation_Model-red) 
![Transfer Learning](https://img.shields.io/badge/Task-Transfer_Learning-red) 
![Recommendation](https://img.shields.io/badge/Task-Recommendation-red) 

Quick links: 
[📋Blog](#Blog) |
[🗃️Download](#Dataset) |
[📭Citation](#Citation) |
[🛠️Code](#Training) |
[🚀Evaluation](#Baseline_Evaluation) |
[🤗Leaderboard](#Leaderboard) |
[👀Others](#Tenrec) |
[💡News](#News)

<p align="center" width="100%">
  <img src='https://camo.githubusercontent.com/ace7effc2b35cda2c66d5952869af563e851f89e5e1af029cfc9f69c7bebe78d/68747470733a2f2f692e696d6775722e636f6d2f77617856496d762e706e67' width="100%">
</p>

# Note
In this paper, we evaluate the TransRec model based on end-to-end training of the recommender backbone and item modality encoder, which is computationally expensive. The reason we do this is because so far there is no widely accepted paradigm for pre-training recommendation models. End-to-end training shows better performance than pre-extracted multimodal features. However, we hope that NineRec can inspire more effective and efficient methods of pre-training recommendation models, rather than just limiting it to the end-to-end training paradigm. If one can develop a very efficient method that goes beyond end-to-end training but can be effectively transferable, it will be a great contribution to the community!!!

# Blog
- [Pre-training and Transfer Learning in Recommender System](https://medium.com/@lifengyi_6964/pre-training-and-transfer-learning-in-recommender-system-907b1011be6e)
- [Multimodal Multi-domain Recommendation System DataSet](https://medium.com/@lifengyi_6964/multimodal-multi-domain-recommendation-system-dataset-e814afcdc68a)
- [Recommendation-Systems-without-Explicit-ID-Features-A-Literature-Review](https://github.com/westlake-repl/Recommendation-Systems-without-Explicit-ID-Features-A-Literature-Review)

# Dataset
<!-- **Kindly note that collecting data and running these TransRec experiments cost us a lot of money. Our lead suggested us to release a sample of 1000 images per dataset before acceptance. If reviewers want to see the entire datasets or plan to use it now for their research, we are more than happy to provide full datasets. Feel free to inform us in the rebuttal stage.** -->

<!-- **We release a sample of 1000 images per dataset. All datasets and code (already attached here) will be provided once the paper is accepted..** -->

All datasets have been released!!  If you have any questions about our dataset and code, please email us.

<!-- Download link: https://sandbox.zenodo.org/record/1153424#.Y9dALnZByw4 -->
## Download link
<!-- **- Zenodo:https://sandbox.zenodo.org/record/1242127** -->
- Google Drive: [Source Datasets](https://drive.google.com/file/d/1fR5XB-dC0CDsyZ4BvtUJJn7i3OnWJa-9/view?usp=sharing), [Downstream Datasets](https://drive.google.com/file/d/15RlthgPczrFbP4U7l6QflSImK5wSGP5K/view?usp=sharing)

If you are interested in pre-training on a larger dataset (even than our source dataset), please visit our PixelRec: https://github.com/westlake-repl/PixelRec. PixelRec can be used as the source data set of NineRec, and these downstream tasks of NineRec are cross-domain/platform scenarios. 

<p align="center" width="100%">
  <img src='https://github.com/westlake-repl/NineRec/blob/main/assets/NineRec_figure1.png' width="90%">
  <img src='https://github.com/westlake-repl/NineRec/blob/main/assets/NineRec_figure2.png' width="90%">
</p>

<!-- We also provide an auto-downloader to make each image easy to download and available permanently. Run `NineRec_downloader.exe` to start downloading. (still 1000 images per dataset before acceptance) -->

<!-- Additionally, we offer an auto-downloader to simplify the process of downloading each image and make them permanently available. To initiate the download process, run the `NineRec_downloader.exe` file. Currently, the auto-downloader is only compatible with Windows systems, but we will provide a Linux version after acceptance. -->

<!-- <div align=center><img width="150" src="https://github.com/anonymous-ninerec/NineRec/blob/main/Downloader/example_image.png"/></div> -->

## Data Format (take QB as an example)
- `QB_cover` contains the raw images in JPG format, with item ID as the file name:
<p align="left" width="100%">
<img src='https://github.com/westlake-repl/NineRec/blob/main/assets/NineRec_figure4.png' width="20%">
</p>

- `QB_behaviour.tsv` contains the user-item interactions in item sequence format, where the first field is the user ID and the second field is a sequence of item ID (has been provided in QB and TN, see Dataset Preparation below to generate this file for others):

User ID     | Item ID Sequence
------------| ----------------------------------------------------
u14500 | v17551 v165612 v288299 v14633 v350433

- `QB_pair.csv` contains the user-item interactions in user-item pair format, where the first field is the user ID, the second field is the item ID, and the third field is a timestamp:

User ID     | Item ID  |  Timestamp
------------| -------- | -------- 
u14500      | v17551 | (only not provided in QB and TN) 

- `QB_item.csv` contains the raw texts, where the first field is the item ID and the second field is the text in Chinese, and the third field is the text in English:

Item ID     | Text in Chinese  |  Text in English
------------| -------- | -------- 
v17551 | 韩国电影，《女教师》 | "Korean Movie, The Governess"

- `QB_url.csv` contains the URL link of items, where the first field is the item ID and the second field is the URL:

Item ID     | URL
------------| --------
v17551      | (only not provided in QB and TN)

*Note that source datasets, Bili_2M and its smaller version Bili_500K, share the same image folder `Source_Bili_2M_cover` for less storage space.  


# Citation
If you use our dataset, code or find NineRec useful in your work, please cite our paper as:

```bib
@article{zhang2023ninerec,
      title={NineRec: A Benchmark Dataset Suite for Evaluating Transferable Recommendation}, 
      author={Jiaqi Zhang and Yu Cheng and Yongxin Ni and Yunzhu Pan and Zheng Yuan and Junchen Fu and Youhua Li and Jie Wang and Fajie Yuan},
      journal={arXiv preprint arXiv:2309.07705},
      year={2023}
}
```
> :warning: **Caution**: It's prohibited to privately modify the dataset and then offer secondary downloads. If you've made alterations to the dataset in your work, you are encouraged to open-source the data processing code, so others can benefit from your methods.  Or notify us of your new dataset so we can put it on this Github with your paper.

# Code
## Environments
```
Pytorch==1.12.1
cudatoolkit==11.2.1
sklearn==1.2.0
python==3.9.12
```
## Dataset Preparation
Run `get_lmdb.py` to get lmdb database for image loading. Run `get_behaviour.py` to convert the user-item pairs into item sequences format.
## Run Experiments
Run `train.py` for pre-training and transferring. Run `test.py` for testing. See more specific instructions in each baseline.

# Baseline_Evaluation

<p align="center" width="100%">
  <img src='https://github.com/westlake-repl/NineRec/blob/main/assets/NineRec_figure3.png' width="90%">
</p>

# Leaderboard
coming soon.

# Tenrec
Tenrec (https://github.com/yuangh-x/2022-NIPS-Tenrec) is the sibling dataset of NineRec, which includes multiple user feedback and platforms. It is suitable for studying ID-based transfer and lifelong learning.   

# News
实验室招聘科研助理、实习生、博士生和博后，请联系通讯作者。
