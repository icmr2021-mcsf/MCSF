## Evaluating and Extending Unsupervised VideoSummarization Methods

1. Reproduce unsupervised method CSNet and RL-method SUM-Ind.
2. Evaluating [SUM-GAN-AAE](https://core.ac.uk/download/pdf/286400027.pdf), [SUM-GAN-sl](http://doi.acm.org/10.1145/3347449.3357482), [CSNet](https://ojs.aaai.org//index.php/AAAI/article/view/4872), and [SUM-Ind](http://www.openaccess.hacettepe.edu.tr:8080/xmlui/handle/11655/11953) using F1-score and rank correlation coefficients.
2. Implementing MCSF method.


# Main Dependencies
python 3.6

pytorch 1.5

h5py 2.10

hdf5 1.10

tabulate 0.8

tensorboard 2.0

tensorboardx 2.0

# Project Structure
```
Directory: 
- /data
- /csnet (implementation of csnet method)
- /src/evaluation (evaluation using F1-score and rank correlations coefficients)
- /src/visualization 
- /sum-ind (implementation of SUM-Ind method)
- /mcsf-places365-early-fusion 
- /mcsf-places365-late-fusion 
- /mcsf-places365-intermediate-fusion

```
# Datasets
Structured h5 files with the video features and annotations of the SumMe and TVSum datasets are available within the "data" folder. The GoogleNet features of the video frames were extracted by [Ke Zhang](https://github.com/kezhang-cs) and [Wei-Lun Chao] and the h5 files were obtained from [Kaiyang Zhou](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce). 

These files have the following structure:
```
/key
    /features                 2D-array with shape (n_steps, feature-dimension)
    /gtscore                  1D-array with shape (n_steps), stores ground truth improtance score (used for training, e.g. regression loss)
    /user_summary             2D-array with shape (num_users, n_frames), each row is a binary vector (used for test)
    /change_points            2D-array with shape (num_segments, 2), each row stores indices of a segment
    /n_frame_per_seg          1D-array with shape (num_segments), indicates number of frames in each segment
    /n_frames                 number of frames in original video
    /picks                    positions of subsampled frames in original video
    /n_steps                  number of subsampled frames
    /gtsummary                1D-array with shape (n_steps), ground truth summary provided by user (used for training, e.g. maximum likelihood)
    /video_name (optional)    original video name, only available for SumMe dataset
```
Original videos and annotations for each dataset are also available in the authors' project webpages:

**TVSum dataset**: [https://github.com/yalesong/tvsum](https://github.com/yalesong/tvsum) 


**SumMe dataset**: [https://gyglim.github.io/me/vsum/index.html#benchmark](https://gyglim.github.io/me/vsum/index.html#benchmark)



### CSNet
We used the implementation of [SUM-GAN](https://github.com/j-min/Adversarial_Video_Summary) method as a starting point to implement CSNet.

#### How to train
The implementation of CSNet is located under the directory csnet. Run main.py file with the configurations specified in configs.py to train the model.



### SUM-Ind
Make splits
```bash
python create_split.py -d datasets/eccv16_dataset_summe_google_pool5.h5 --save-dir datasets --save-name summe_splits  --num-splits 5
```
As a result, the dataset is randomly split for 5 times, which are saved as json file.

Train and test codes are written in `main.py`. To see the detailed arguments, please do `python main.py -h`.

#### How to train
```bash
python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --gpu 0 --save-dir log/summe-split0 --split-id 0 --verbose
```

#### How to test
```bash
python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --gpu 0 --save-dir log/summe-split0 --split-id 0 --evaluate --resume path_to_your_model.pth.tar --verbose --save-results
```
