from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import pickle


def generate_download_script(list_file, download_dir, fn_output):
    with open(list_file, "rb") as file:
        ids = pickle.load(file)

    cmd_prefix = "youtube-dl https://www.youtube.com/watch?v="
    with open(fn_output, "w") as out_file:
        for id in ids:
            cmd = cmd_prefix + "{} --proxy http://fwdproxy:8080 -o {}/{}.mp4\n".format(
                id, download_dir, id
            )
            out_file.write(cmd)


def main():
    parser = argparse.ArgumentParser(description="Download UVO (Kinetics) videos")

    parser.add_argument(
        "--list_file",
        type=str,
        default=None,
        help="Input pkl file containing Youtube ids to be download",
        required=True,
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=None,
        help="Directory to store downlaoding raw videos",
        required=True,
    )
    parser.add_argument(
        "--output_sh",
        type=str,
        default=None,
        help="Output script to download videos",
        required=True,
    )
    args = parser.parse_args()

    generate_download_script(args.list_file, args.download_dir, args.output_sh)


if __name__ == "__main__":
    main()