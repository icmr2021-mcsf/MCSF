# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input
# import numpy as np
# import glob
#
# model = ResNet50(weights='imagenet', include_top=False, pooling='max')
# file_path = '../../saved_numpy_arrays/RGB_as_numpy/224_dims/*.npy'
# files = glob.glob(file_path)
# saved_path = '../../saved_numpy_arrays/Resnet_features/'
# files.sort()
#
# for i in range(len(files)):
#     file_n = np.load(files[i])
#     fileName = files[i].split('/')[-1].split('.')[0]
#
#     x = preprocess_input(file_n)
#     preds = model.predict(x)
#     np.save(saved_path + fileName + '.npy', preds)
#     print(fileName)
# # preds = np.squeeze(np.squeeze(np.squeeze(preds, axis=0),axis=0), axis=0)


# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import glob
import cv2
import numpy as np
from utils import *
import torch

import argparse

PROCESSED_SUMME = '../data/SumMe/processed/eccv16_dataset_summe_google_pool5.h5'
SUMME_MAPPED_VIDEO_NAMES = '../data/mapped_video_names.json'
SUMME_FLOW_FEATURES = '../saved_numpy_arrays/SumMe/I3D_features/'
DOWNSAMPLED_SUMME_FLOW_FEATURES = '../saved_numpy_arrays/SumMe/I3D_features/FLOW/downsampled'
DOWNSAMPLED_SUMME_RGB_FEATURES = '../saved_numpy_arrays/SumMe/I3D_features/RGB/downsampled'






def arg_parser():
    # ../data/SumMe/videos  ../data/SumMe/GT
    # ../ data / TVSum / video /  ../data/TVSum/data
    # ../data/VSUMM/new_database  ../data/VSUMM/newUserSummary
    parser = argparse.ArgumentParser(description='Downsample features')
    parser.add_argument('--type', default='summe', type=str, help='summe or tvsum')
    parser.add_argument('--dataset', default=PROCESSED_SUMME, type=str)
    parser.add_argument('--mapped_names', default=SUMME_MAPPED_VIDEO_NAMES, type=str)
    parser.add_argument('--flow_features', default=SUMME_FLOW_FEATURES, type=str)

    return parser


def downsample_video(video_frames,picks):
    frames = [video_frames[i] if i < len(video_frames) else video_frames[-1] for i in picks]
    return np.array(frames)


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    # dataset, videos, GT, Sample rate, model architecture
    type, dataset, mapped_names, flow_features = args.type, args.dataset, args.mapped_names, args.flow_features
    processed_dataset = load_processed_dataset(dataset)
    mapped_video_names = read_json(mapped_names)


    path_feature_rgb = flow_features + "RGB" + '/'
    path_feature_flow = flow_features + "FLOW" + '/'
    files_feature_rgb = glob.glob(path_feature_rgb + 'features/*.npy')
    files_feature_flow = glob.glob(path_feature_flow + 'features/*.npy')
    files_feature_rgb.sort()
    files_feature_flow.sort()
    print(files_feature_flow)
    videos_scores = {}
    # for idx, video_data in enumerate(files_feature_rgb):
    for idx, video_data in enumerate(files_feature_flow):
        frame_features = np.load(video_data)
        filename=drop_file_extension(video_data.split('\\')[-1])
        mapped_name = mapped_video_names[filename]
        sampled_frames = downsample_video(frame_features, processed_dataset[mapped_name]['picks'])
        #np.save(DOWNSAMPLED_SUMME_RGB_FEATURES+'/'+mapped_name+'.npy',sampled_frames)
        np.save(DOWNSAMPLED_SUMME_FLOW_FEATURES+'/'+mapped_name+'.npy',sampled_frames)
        print('processed: ', processed_dataset[mapped_name], np.array(processed_dataset[mapped_name + '/features']).shape)
        print('sampled:', sampled_frames.shape)


