import argparse
from time import time

from data.readers import StackLoader
from stack_sim import stack_sim


# --stacktrace_dir "/home/centos/akhvorov/data/log_2018-09-01_2-data"
# --labels_path "/home/centos/akhvorov/data/log_2018-09-01_2.csv"
# --method tracesim


supported_methods = ['tracesim', 'lerch', 'moroo', 'rebucket', 'cosine', 'levenshtein', 'brodie', 'prefix']


def parse_args():
    parser = argparse.ArgumentParser(description='Stack similarity')
    parser.add_argument('--stacktrace_dir', type=str, help='Directory with stacktraces in json format')
    parser.add_argument('--labels_path', type=str, help='CSV file with similarity labels of two stacktraces')
    parser.add_argument('--method', type=str, help='Method for similarity prediction',
                        choices=supported_methods + ['all'])
    return parser.parse_args()


def main():
    args = parse_args()
    start = time()

    stack_loader = StackLoader(args.stacktrace_dir)

    if args.method == 'all':
        res = []
        for method in supported_methods:
            score = stack_sim(stack_loader, args.labels_path, method=method,
                              rand_split=True, split_ratio=0.8)
            res.append((method, score))
        print(res)
    else:
        score = stack_sim(stack_loader, args.labels_path, args.method,
                          rand_split=True, split_ratio=0.8)
        print(score)

    print("Time:", time() - start)


if __name__ == "__main__":
    main()

# tracesim
# 0.8064: [0.7963, 0.8182]
# 0.8046: [0.7811, 0.8304]
# Time: 11478.67515873909

# tracesim, this code
# 0.8064: [0.7953, 0.8171]
# 0.8131: [0.7883, 0.8389]
# Time: 11704.33979344368

# lerch, this code
# 0.7739: [0.7616, 0.7832]
# 0.7594: [0.7329, 0.7865]
