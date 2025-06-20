import argparse

import os
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str)
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument(
    "--dataset",
    type=str,
    choices=["sharegpt", "ultrachat", "mixture_of_thoughts"],
    default="sharegpt",
)
args = parser.parse_args()

s = 0
if args.dataset == "sharegpt":
    e = 68000 - 1
elif args.dataset == "ultrachat":
    e = 200000 - 1
elif args.dataset == "mixture_of_thoughts":
    e = 100000 - 1

gpus = [[0], [1], [2], [3], [4], [5], [6], [7]]
num_p = len(gpus)
outdir = f"{args.outdir.rstrip('/')}/qwen2_5b_eagle_1_{args.dataset}_{s}_{e}_mufp16"


def split_range(start, end, n, over=False):
    length = end - start + 1  # Include the end
    base_interval = length // n
    additional = length % n  # Get the remainder of the division
    intervals = []
    previous = start

    for i in range(n):
        current_interval = base_interval + (1 if i < additional else 0)
        if over:
            intervals.append((previous, previous + current_interval))
        else:
            intervals.append(
                (previous, previous + current_interval - 1)
            )  # '-1' because the end is inclusive
        previous += current_interval

    return intervals


def run_command(cmd):
    os.system(cmd)


if not os.path.exists(outdir):
    os.makedirs(outdir)


data_a = split_range(s, e, num_p, over=True)
commands = []
for i in range(num_p):
    index = i
    start = data_a[i][0]
    end = data_a[i][1]

    gpu_index = gpus[i]
    gpu_index_str = " ".join(map(str, gpu_index))
    command = "python3 generate_data.py --start={} --end={} --index={} --gpu_index {} --outdir {} --model_name {} --dataset {}".format(
        start, end, index, gpu_index_str, outdir, args.model_name, args.dataset
    )
    commands.append(command)

with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    for command in commands:
        executor.submit(run_command, command)
        print(command)
