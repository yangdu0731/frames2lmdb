import os
import sys
import lmdb
import cv2
import pickle

import numpy as np

from torch.utils.data import Dataset, DataLoader
from cv2 import IMREAD_COLOR


class FramesFolder(Dataset):
  def __init__(self, frames_info):
    super(FramesFolder, self).__init__()

    self.frames_info = frames_info
    self.frames_dir_list = []

    with open(self.frames_info) as f:
      for line in f:
        self.frames_dir_list.append(line.strip())

  def __len__(self):
    return len(self.frames_dir_list)

 
  def __getitem__(self, idx):
    frame_list = os.listdir(self.frames_dir_list[idx])
    frame_list.sort()
    frames = []
    frames = dict()
    frames['name'] = self.frames_dir_list[idx].split('/')[-1]
    data = []

    for frame in frame_list:
      frame_path = os.path.join(self.frames_dir_list[idx], frame)

      with open(frame_path, 'rb') as f:
        data.append(f.read())

    frames['data'] = data

    return frames

def test_lmdb(lmdb_path, lmdb_key="bar_dir/image_00001.jpg"):
  env = lmdb.open(lmdb_path, subdir=False, 
                  readonly=True, lock=False, readahead=False, meminit=False)

  with env.begin(write=False) as txn:
    byteflow = txn.get(lmdb_key.encode('ascii'))
  
  img_np = np.frombuffer(byteflow, np.uint8)
  img = cv2.imdecode(img_np, IMREAD_COLOR)
  cv2.imshow(lmdb_key, img)
  cv2.waitKey(1000)
  cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)

def folder2lmdb(frames_info, lmdb_path, write_frequency=5000, num_workers=4):
  dataset = FramesFolder(frames_info)
  data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)
  db = lmdb.open(lmdb_path, subdir=False, map_size=1099511627776 * 10, 
                 readonly=False, meminit=False, map_async=True)
  txn = db.begin(write=True)               

  for idx, data in enumerate(data_loader):
    frames = data[0]
    frames_name = frames['name']
    frames_data = frames['data']
    num_frames = len(frames_data)

    for t in range(num_frames):
      lmdb_key = '{0}/image_{1:05d}.jpg'.format(frames_name, t)
      lmdb_value = frames_data[t]

      txn.put(lmdb_key.encode('ascii'), lmdb_value)

    if idx % write_frequency == 0:
      txn.commit()
      txn = db.begin(write=True)
  
  txn.commit()
  db.sync()
  db.close()


if __name__ == "__main__":
  folder2lmdb(sys.argv[1], sys.argv[2])
  test_lmdb(sys.argv[2], "bar_dir/image_00000.jpg")
  test_lmdb(sys.argv[2], "bar_dir/image_00001.jpg")
  test_lmdb(sys.argv[2], "foo_dir/image_00000.jpg")
  test_lmdb(sys.argv[2], "foo_dir/image_00001.jpg")

