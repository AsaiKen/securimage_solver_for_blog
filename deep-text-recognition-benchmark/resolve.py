import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import os

from dataset import RawDataset, AlignCollate
from model import Model
from utils import CTCLabelConverter, AttnLabelConverter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True
cudnn.deterministic = True


class Option:
  def __init__(self):
    self.FeatureExtraction = 'VGG'
    self.PAD = False
    self.Prediction = 'CTC'
    self.SequenceModeling = 'BiLSTM'
    self.Transformation = 'None'
    self.batch_max_length = 25
    self.batch_size = 192
    self.character = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    self.hidden_size = 256
    self.imgH = 32
    self.imgW = 100
    self.input_channel = 1
    self.num_fiducial = 20
    self.num_gpu = 1
    self.output_channel = 512
    self.rgb = False
    self.saved_model = os.path.join(os.path.dirname(__file__), 'saved_models',
                                    'None-VGG-BiLSTM-CTC-Seed1111/best_accuracy.pth')
    self.sensitive = True
    self.workers = 4


opt = Option()

""" model configuration """
if 'CTC' in opt.Prediction:
  converter = CTCLabelConverter(opt.character)
else:
  converter = AttnLabelConverter(opt.character)
opt.num_class = len(converter.character)

if opt.rgb:
  opt.input_channel = 3
model = Model(opt)
model = torch.nn.DataParallel(model).to(device)

# load model
model.load_state_dict(torch.load(opt.saved_model, map_location='cpu'))

# predict
model.eval()


def resolve(image_folder):
  # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
  AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
  demo_data = RawDataset(root=image_folder, opt=opt)  # use RawDataset
  demo_loader = torch.utils.data.DataLoader(
    demo_data, batch_size=opt.batch_size,
    shuffle=False,
    num_workers=int(opt.workers),
    collate_fn=AlignCollate_demo, pin_memory=True)

  with torch.no_grad():
    for image_tensors, image_path_list in demo_loader:
      batch_size = image_tensors.size(0)
      image = image_tensors.to(device)
      # For max length prediction
      length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
      text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

      if 'CTC' in opt.Prediction:
        preds = model(image, text_for_pred).log_softmax(2)

        # Select max probabilty (greedy decoding) then decode index to character
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        _, preds_index = preds.permute(1, 0, 2).max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)

      else:
        preds = model(image, text_for_pred, is_train=False)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

      for img_name, pred in zip(image_path_list, preds_str):
        if 'Attn' in opt.Prediction:
          pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])

        return pred
