from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import argparse
import logging
import pandas
import os.path as path
import subprocess
import json
import datetime

logging.basicConfig()
log = logging.getLogger("Preprocess Kinetics videos")
log.setLevel(logging.DEBUG)


def get_video_meta_data(fn):
    cmd = 'ffprobe -v quiet -print_format json -show_format -show_streams ' + \
        fn
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, _ = process.communicate()
    # uncomment if you are using Python2
    output = output.decode("utf-8")
    output.replace('\n', '')
    parsed_json = json.loads(output)
    stream_idx = -1
    # pdb.set_trace()
    for i in range(len(parsed_json['streams'])):
        if parsed_json['streams'][i]['codec_type'] == 'video':
            stream_idx = i
            break
    if stream_idx == -1:
        return 0, 0, 0
    height = int(parsed_json['streams'][stream_idx]['height'])
    width = int(parsed_json['streams'][stream_idx]['width'])
    avg_frame_rate = str(parsed_json['streams'][stream_idx]['avg_frame_rate'])
    tmp = avg_frame_rate.split('/')
    avg_frame_rate = float(tmp[0]) / float(tmp[1])
    return height, width, avg_frame_rate


def preprocess_video(root_dir, fn_input, fn_output, dir_out=''):
    data = pandas.read_csv(fn_input, dtype='str')
    count = 0
    miss = 0
    with open(fn_output, 'w') as out_file:
        for _, row in data.iterrows():
            youtube_id = row['youtube_id']
            fn_mp4 = '{}{}.mp4'.format(root_dir, youtube_id)
            fn_mkv = '{}{}.mkv'.format(root_dir, youtube_id)
            fn_mkv2 = '{}{}.mp4.mkv'.format(root_dir, youtube_id)
            if path.isfile(fn_mp4):
                count = count + 1
                fn = fn_mp4
            elif path.isfile(fn_mkv):
                count = count + 1
                fn = fn_mkv
            elif path.isfile(fn_mkv2):
                count = count + 1
                fn = fn_mkv2
            else:
                miss = miss + 1
                fn = ''
            if fn:
                h, w, f = get_video_meta_data(fn)
                if h:
                    ss = int(row['time_start'])
                    time_start = str(datetime.timedelta(seconds=ss))
                    ee = int(row['time_end'])
                    duration = ee - ss
                    fn_out = '{}/{}.mp4'.format(dir_out, youtube_id)
                    if h < w:
                        cmd = 'ffmpeg -y -ss {} -t 00:00:{} -i {} -r 30 -q:v 1 -vf scale=-2:480 {}\n'.format(
                            time_start, duration, fn, fn_out
                        )
                    else:
                        cmd = 'ffmpeg -y -ss {} -t 00:00:{} -i {} -r 30 -q:v 1 -vf scale=480:-2 {}\n'.format(
                            time_start, duration, fn, fn_out
                        )
                    out_file.write(cmd)
                    if (miss + count) % 1000 == 0:
                        print('processed {}'.format(miss + count))

                else:
                    count = count - 1
                    miss = miss + 1
                    print(fn)

    print('miss {} of {}'.format(miss, count + miss))
    return


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess kinetics videos."
    )

    parser.add_argument("--input_file", type=str, default=None,
                        help="Input csv raw file",
                        required=True)
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Directory of raw videos",
                        required=True)
    parser.add_argument("--output_sh", type=str, default=None,
                        help="Output script to extract clips",
                        required=True)
    parser.add_argument("--dir_out", type=str, default=None,
                        help="Dir to store extracted clips",
                        required=True)
    args = parser.parse_args()

    preprocess_video(
        args.video_dir,
        args.input_file,
        args.output_sh,
        args.dir_out
    )


if __name__ == '__main__':
    main()