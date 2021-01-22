import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader_stage1, CreateDataLoader_stage2
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import ntpath
from subprocess import check_call
from models.SMAPNET import SMapNet
from torch.autograd import Variable
import PIL.Image as pil_image
import torch
import torch.backends.cudnn as cudnn
import numpy
from torchvision import transforms
import glob
from tqdm import tqdm

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
cudnn.benchmark = True
device_name = 'cuda:' + str(opt.gpu_ids[0])
device = torch.device(device_name if torch.cuda.is_available() else "cpu")
image_dir = opt.results_dir

## extract origional label
os.makedirs(opt.results_dir, exist_ok=True)
imgFol = opt.dataroot
oriLabelFol = os.path.join(opt.results_dir, 'oriLabel')
os.makedirs(oriLabelFol, exist_ok=True)

print('\n-----------------Extracting origional segment-----------------')
seg_check = check_call(['python3 -u ./semantic-segmentation-pytorch-master/test.py --cfg ./semantic-segmentation-pytorch-master/config/ade20k-resnet50dilated-ppm_deepsup.yaml --imgs '+imgFol+' --gpu ' + str(opt.gpu_ids[0]) +' DIR ./semantic-segmentation-pytorch-master/ade20k-resnet50dilated-ppm_deepsup/ TEST.result '+ oriLabelFol +' TEST.checkpoint epoch_20.pth --fmt ' + opt.fmt ],shell=True)


data_loader = CreateDataLoader_stage1(opt)
dataset = data_loader.load_data()

model = create_model(opt)
visualizer = Visualizer(opt)

##load SMapNet for segmentation enhancement, seperated pretrained network
sMapModel =  SMapNet(1)
sMapState_dict = sMapModel.state_dict()
weights =  torch.load(opt.sMapWeights_path)['model_state_dict'] ## add later
sMapModel.load_state_dict(weights)
sMapModel = sMapModel.to(device)
sMapModel.eval()

# test
## use origional label to generate the comp_image
print('\n-----------------Downsampling with embedding original segment-----------------')
pbar = tqdm(total=len(dataset))
for i, data in enumerate(dataset):
    generated, comp_image, up_image = model.inference_stage1(data['image'],data['label'], data['ds'])
    visuals = OrderedDict([('synthesized_image', util.tensor2im(generated.data[0])),
                            ('real_image', util.tensor2im(data['image'][0])),
			               ('comp_image', util.tensor2im(comp_image.data[0])),
                           ('up_image', util.tensor2im(up_image.data[0]))
                           ])
    img_path = data['path']
    short_path = ntpath.basename(img_path[0])
    name = os.path.splitext(short_path)[0]

    upFol = os.path.join(image_dir, 'up')
    compFol = os.path.join(image_dir, 'comp')
    oriSynFol = os.path.join(image_dir, 'oriSyn')
    oriImg = os.path.join(image_dir, 'oriImg')
    os.makedirs(upFol, exist_ok=True)
    os.makedirs(compFol, exist_ok=True)
    os.makedirs(oriSynFol, exist_ok=True)
    os.makedirs(oriImg, exist_ok=True)

    #save tracking results
    for label, image_numpy in visuals.items():
        image_name = '%s.png' % (name)
        if(label=='up_image'):
          save_path = os.path.join(image_dir, 'up', image_name) #save for seg extraction
          util.save_image(image_numpy, save_path)
        if(label=='comp_image'):
          save_path = os.path.join(image_dir, 'comp', image_name) #save for compression
          util.save_image(image_numpy, save_path)
        if(label=='synthesized_image'): #DSSLIC syn Image
          save_path = os.path.join(image_dir, 'oriSyn', image_name) #save for comparison
          util.save_image(image_numpy, save_path)
        if(label=='real_image'): #DSSLIC syn Image
          save_path = os.path.join(image_dir, 'oriImg', image_name) #save for comparison
          util.save_image(image_numpy, save_path)
    pbar.update(1)

##Extract segment from up
irLabelFol = os.path.join(image_dir, 'irLabel')
enLabelFol = os.path.join(image_dir, 'enLabel')
os.makedirs(irLabelFol, exist_ok=True)
os.makedirs(enLabelFol, exist_ok=True)

print('\n--Extracting segment from upsampled version--')
irSeg_check = check_call(['python3 -u ./semantic-segmentation-pytorch-master/test.py --cfg ./semantic-segmentation-pytorch-master/config/ade20k-resnet50dilated-ppm_deepsup.yaml --imgs '+upFol+' --gpu ' + str(opt.gpu_ids[0]) +' DIR ./semantic-segmentation-pytorch-master/ade20k-resnet50dilated-ppm_deepsup/ TEST.result '+ irLabelFol +' TEST.checkpoint epoch_20.pth'  ],shell=True)

##Enhanced the segment
images_path = glob.glob(os.path.join(irLabelFol, '*.png'))
print('\n-----------------Enhancing segments with SMapNet-----------------')
pbar = tqdm(total=len(images_path))
for image_path in images_path:
    filename = os.path.basename(image_path).split('.')[0]
    input = pil_image.open(image_path)#.convert('RGB')
    size = numpy.array(input).shape

    input = transforms.ToTensor()(input).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = sMapModel(input)
        pred = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy().squeeze()
        output = pil_image.fromarray(pred, mode = 'L')

    out_name = os.path.join(enLabelFol, '{}{}'.format(filename,'.png'))
    util.save_image(pred, out_name)
    pbar.update(1)


##Use new enhanced-segment to reconstruct the image
stage2_data_loader = CreateDataLoader_stage2(opt)
stage2_dataset = stage2_data_loader.load_data()

print('\n-----------------Reconstructing frame with enhanced segment-----------------\n')
pbar = tqdm(total=len(stage2_dataset))
for i, data in enumerate(stage2_dataset):
    generated = model.inference_stage2(data['comp'],data['enLabel'], data['ds'])
    visuals = OrderedDict([('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    image_dir = opt.results_dir
    short_path = ntpath.basename(img_path[0])
    name = os.path.splitext(short_path)[0]

    synFol = os.path.join(image_dir, 'syn')
    os.makedirs(synFol, exist_ok=True)

    for label, image_numpy in visuals.items(): #Ours Self-enhanced
        image_name = '%s.png' % (name)
        save_path = os.path.join(image_dir, 'syn', image_name)
        util.save_image(image_numpy, save_path)
    pbar.update(1)


print('\n----Please run the Matlab code in ./evaluation code/main.m to compress the residual and downsampled version (with BPG and FLIF) to get the final results----\n')
