from os import listdir
import json
import numpy as np
import h5py
from generate_summary import generate_summary
from evaluation_metrics import evaluate_summary
from utils import init_scores_dict

split_num=4
path = '../exp1/non-overlapping_tvsum/results/split{}'.format(split_num)  # path to the json files with the computed importance scores for each epoch

results = listdir(path)
results.sort(key=lambda video: int(video[6:-5]))
PATH_TVSum = '../data/TVSum/eccv16_dataset_tvsum_google_pool5.h5'
eval_method = 'avg'  # the proposed evaluation method for TVSum videos
#eval_method = 'max'  # the proposed evaluation method for TVSum videos

# for each epoch, read the results' file and compute the f_score
f_score_epochs = []
videos_scores = {}
for idx, epoch in enumerate(results):
    print(epoch)
    all_scores = []
    with open(path + '/' + epoch) as f:
        data = json.loads(f.read())
        keys = list(data.keys())
        if idx == 0:
            video_numbers=[key[6:]for key in keys]
            videos_scores = init_scores_dict(video_numbers)

        for video_name in keys:
            scores = np.asarray(data[video_name])
            all_scores.append(scores)

    all_user_summary, all_shot_bound, all_nframes, all_positions = [], [], [], []
    with h5py.File(PATH_TVSum, 'r') as hdf:
        for video_name in keys:
            video_index = video_name[6:]

            user_summary = np.array(hdf.get('video_' + video_index + '/user_summary'))
            sb = np.array(hdf.get('video_' + video_index + '/change_points'))
            n_frames = np.array(hdf.get('video_' + video_index + '/n_frames'))
            positions = np.array(hdf.get('video_' + video_index + '/picks'))

            all_user_summary.append(user_summary)
            all_shot_bound.append(sb)
            all_nframes.append(n_frames)
            all_positions.append(positions)

    all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions)

    all_f_scores = []
    # compare the resulting summary with the ground truth one, for each video
    for video_name, video_index in zip(keys, range(len(all_summaries))):
        summary = all_summaries[video_index]
        user_summary = all_user_summary[video_index]
        f_score = evaluate_summary(summary, user_summary, eval_method)
        all_f_scores.append(f_score)
        videos_scores[video_name[6:]].append(f_score)

    f_score_epochs.append(np.mean(all_f_scores))
    print("f_score: ", np.mean(all_f_scores))

for key in videos_scores.keys():
    videos_scores[key] = np.round(np.mean(videos_scores[key]), decimals=2)

with open(path + '/video_scores{}.txt'.format(split_num), 'w') as outfile:
    json.dump(videos_scores, outfile)

with open(path + '/f_scores.txt', 'w') as outfile:
    json.dump(f_score_epochs, outfile)