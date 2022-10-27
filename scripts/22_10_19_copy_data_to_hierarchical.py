import os
import sys
import re
import argparse
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Copy files from a flat to hierarchical structure")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--input_dir", help="input dir")
    parser.add_argument("-o", "--output_dir", help="output dir")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    existing = []
    for root, dirs, files in os.walk(args.output_dir):
        for f in files:
            existing.append(f)

    existing = set(existing)

    files_to_copy = []
    for root, dirs, files in os.walk(args.input_dir):
        for f in files:
            if f in existing:
                continue
            file_path = os.path.join(root, f)
            output_path = os.path.join(args.output_dir, f[0:2], f[2:4], f)
            files_to_copy.append((file_path, output_path, f))

    for i, x in enumerate(files_to_copy):

        os.makedirs(os.path.join(args.output_dir, x[2][0:2], x[2][2:4]), exist_ok=True)
        print(f"{i} {len(files_to_copy)} {x[0]} {x[1]}")
        shutil.copyfile(x[0], x[1])

    return 0


if __name__ == "__main__":
    sys.exit(main())
