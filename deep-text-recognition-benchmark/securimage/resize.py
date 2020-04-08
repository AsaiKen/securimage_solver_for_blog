import os.path
import shutil

from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')

for model in ['train', 'test', 'validate']:
  src_dir = os.path.join(DATA_DIR, 'raw', model)
  dst_dir = os.path.join(DATA_DIR, 'resize', model)
  if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
  os.makedirs(dst_dir)
  for name in os.listdir(src_dir):
    if not name.endswith('.png'):
      continue
    src_path = src_dir + '/%s' % name
    dst_path = dst_dir + '/%s' % name
    try:
      img = Image.open(src_path)
      img_resized = img.resize((100, 32))
      # print(base, ext)
      print(dst_path)
      img_resized.save(dst_path)
    except Exception as e:
      os.unlink(dst_path)
      print(e)
