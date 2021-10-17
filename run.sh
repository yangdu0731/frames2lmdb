#!/bin/bash
# Usage: bash run.sh


frames_info="/home/young/Data/Documents/Projects/frames2lmdb/data/info/datalist.txt"
lmdb_path="/home/young/Data/Documents/Projects/frames2lmdb/data/lmdb/frames.lmdb"

rm -rf $lmdb_path*

python frames2lmdb.py $frames_info $lmdb_path
