import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
for mode in ['train', 'validate', 'test']:
  gt = os.path.join(DATA_DIR, 'resize', mode, 'gt.txt')
  print(gt)
  with open(gt, 'w') as f:
    names = os.listdir(os.path.join(DATA_DIR, 'resize', mode))
    for name in names:
      if not name.endswith('.png'):
        continue
      answer = name.replace('.png', '')
      line = '%s/%s\t%s' % (mode, name, answer)
      f.write(line + '\n')
      print(line)
