import os
import random
import shutil
import string
import sys

from PIL import Image
from flask import Flask, render_template, request, jsonify
from waitress import serve
from werkzeug.utils import secure_filename

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from resolve import resolve

app = Flask(__name__)

ALLOWED_EXTENSIONS = ['png']
app.config['UPLOAD_FOLDER'] = '/tmp'


def randomname():
  return ''.join(random.choices(string.ascii_letters + string.digits, k=64))


def allowed_file(filename):
  return '.' in filename and \
         filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def index():
  return render_template('index.html')


@app.route('/solve', methods=['POST'])
def send():
  img_file = request.files['img_file']
  if img_file and allowed_file(img_file.filename):
    dirpath = os.path.join(app.config['UPLOAD_FOLDER'], randomname())
    os.makedirs(dirpath)
    try:
      filename = secure_filename(img_file.filename)
      filepath = os.path.join(dirpath, filename)
      img_file.save(filepath)
      img = Image.open(filepath)
      img_resize = img.resize((100, 32))
      img_resize.save(filepath)
      anwser = resolve(dirpath)
      return jsonify({'answer': anwser})
    finally:
      shutil.rmtree(dirpath)
  else:
    return jsonify({'error': 'not png file'})


if __name__ == '__main__':
  # app.debug = True
  # app.run()
  serve(app, host='0.0.0.0', port=5000)
