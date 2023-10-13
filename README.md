# NineRec: A Benchmark Dataset Suite for Evaluating Transferable Recommendation

# Dataset
<!-- **Kindly note that collecting data and running these TransRec experiments cost us a lot of money. Our lead suggested us to release a sample of 1000 images per dataset before acceptance. If reviewers want to see the entire datasets or plan to use it now for their research, we are more than happy to provide full datasets. Feel free to inform us in the rebuttal stage.** -->

<!-- **We release a sample of 1000 images per dataset. All datasets and code (already attached here) will be provided once the paper is accepted..** -->

We have released all 9 downstream datasets, and we will provide access to the source dataset and code (which are already included in this submission) once the paper is accepted.

<!-- Download link: https://sandbox.zenodo.org/record/1153424#.Y9dALnZByw4 -->
Download link: 
- Zenodo:https://sandbox.zenodo.org/record/1242127
- Google Drive: https://drive.google.com/file/d/1LOaZLKiVo80pBw0hXYyNyYnww0smb6Ub/view?usp=drive_link

If you are interested in conducting pre-training, you can find a relatively large image dataset available at https://github.com/westlake-repl/IDvs.MoRec. Please follow the provided instructions to utilize the dataset properly, as it is not fully published yet. If you want to pre-train on a very large-scale image/video/text dataset for a foundation Recsys model, contact our leading authors by email.

<!-- We also provide an auto-downloader to make each image easy to download and available permanently. Run `NineRec_downloader.exe` to start downloading. (still 1000 images per dataset before acceptance) -->

<!-- Additionally, we offer an auto-downloader to simplify the process of downloading each image and make them permanently available. To initiate the download process, run the `NineRec_downloader.exe` file. Currently, the auto-downloader is only compatible with Windows systems, but we will provide a Linux version after acceptance. -->

<!-- <div align=center><img width="150" src="https://github.com/anonymous-ninerec/NineRec/blob/main/Downloader/example_image.png"/></div> -->

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

# Benchmark
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
Run `train.py` for pre-training and transferring. Run `test.py` for testing.

# Leaderboard
coming soon.

# Kind Reminder
Note that you should not perform secondary processing on the dataset provided in this paper and offer a new download. If you need to do so, we encourage you to open source only the code for data processing and cite our work for data downloading and local processing.

# News
实验室招聘科研助理、实习生、博士生和博后，请联系通讯作者。
