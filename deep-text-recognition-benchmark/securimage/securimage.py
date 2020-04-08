import os
import subprocess
import time
from multiprocessing import Pool

PHP_PATH = '/opt/lampp/bin/php'
if not os.path.exists(PHP_PATH):
  raise FileNotFoundError('please set php binary path to PHP_PATH')
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
PHP_SCRIPT = os.path.join(os.path.dirname(__file__), '../../securimage/securimage_save.php')
POOL_SIZE = 8

target = {'train': 500000, 'validate': 10000, 'test': 10000}


def f(i, start, mode):
  if i % 1000 == 0:
    now = time.time()
    remain = (target[mode] - i) / (i / (now - start))
    print(i, '%.1f sec (remain %.1f sec)' % ((now - start), remain))

  dst_dir = os.path.join(DATA_DIR, 'raw', mode)
  subprocess.check_call([PHP_PATH, PHP_SCRIPT, dst_dir])


if __name__ == '__main__':
  for mode in target:
    print('[*] mode', mode)

    dst_dir = os.path.join(DATA_DIR, 'raw', mode)
    if not os.path.exists(dst_dir):
      os.makedirs(dst_dir)
    count = len(os.listdir(dst_dir))
    print('before', count)

    with Pool(POOL_SIZE) as p:
      start = time.time()
      for i in range(target[mode] - count):
        p.apply_async(f, [i + 1, start, mode])
      p.close()
      p.join()

    end = time.time()
    print('%.1f sec' % (end - start))
    count = len(os.listdir(dst_dir))
    print('after', count)
