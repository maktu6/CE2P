import argparse
import numpy as np
import torch
# torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
from networks.CE2P import Res_Deeplab
from dataset.datasets import InferDataSet
import os
import torchvision.transforms as transforms
from copy import deepcopy
from utils.transforms import transform_parsing
from utils.utils import get_lip_palette
from PIL import Image
from time import time
# from tqdm import tqdm

DATA_DIRECTORY = '/ssd1/liuting14/Dataset/LIP/'
DATA_LIST_PATH = './dataset/list/lip/valList.txt'
# IGNORE_LABEL = 255
NUM_CLASSES = 20
SNAPSHOT_DIR = './snapshots/'
INPUT_SIZE = (473,473)
PALETTE = get_lip_palette() 

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CE2P Network")
    parser.add_argument("--save-dir", type=str,
                        help="Path for saving inference results.")
    parser.add_argument("--data-dir", type=str,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument('--image-ext', type=str, default='jpg',
                        help='image file name extension (default: jpg)')
    parser.add_argument("--list-path", type=str,
                        help="Path to a txt file containing image names for inference.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of cpu workers for dataloader.")
    # parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
    #                     help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--mirror", action="store_true", help="combined with mirro results")

    return parser.parse_args()

def transform_flip_pred(pred_flip):
    """" transform prediction from fliped images for combined with normal prediction

    Args:
        pred_flip: BxCxHxW
    """
    pred_flip_copy = pred_flip.copy()
    right_idx = [15, 17, 19]
    left_idx = [14, 16, 18]
    for i in range(len(right_idx)):
        pred_flip[:,right_idx[i],:,:] = pred_flip_copy[:,left_idx[i],:,:]
        pred_flip[:,left_idx[i],:,:] = pred_flip_copy[:,right_idx[i],:,:]

    pred_flip = pred_flip[:,:,:,::-1].copy()
    return pred_flip

def infer(model, valloader, input_size, num_samples, gpus, save_dir, mirror=False):
    """
    Args:
        mirror: combined with mirro results(only support single gpu)
    """
    model.eval()

    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    current_t = time()
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            image, meta = batch
            num_images = image.size(0)
            if index % 10 == 0:
                print('%d  processd in %.1fs' % (index * num_images, time()-current_t))
                current_t = time()

            # extract infomation for recovering predicition
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            h = meta['height'].numpy()
            w = meta['width'].numpy()
            n = meta['name']
            if mirror:
                image_flip = torch.from_numpy(image.numpy()[:,:,:,::-1].copy())
                image_all = torch.cat([image, image_flip])
                outputs = model(image_all.cuda())
            else:
                outputs = model(image.cuda())
            if gpus > 1:
                NotImplementedError("inference of muti-GPU has not implemented") # TODO: muti-GPU
                # for output in outputs:
                #     parsing = output[0][-1]
                #     nums = len(parsing)
                #     parsing = interp(parsing).data.cpu().numpy()
                #     parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                #     parsing_preds[idx:idx + nums, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
            else:
                parsing = outputs[0][-1]
                parsing = interp(parsing).data.cpu().numpy()
                if mirror:
                    pred_ori, pred_flip = np.split(parsing, 2, axis=0)
                    pred_flip = transform_flip_pred(pred_flip)
                    parsing = np.mean([pred_ori, pred_flip], axis=0)
                parsing = parsing.transpose(0, 2, 3, 1)  # NCHW->NHWC
                parsing_pred = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                assert len(parsing_pred)==len(s)==len(c)==len(h)==len(w)==len(n)
                transform_and_save(parsing_pred, s, c, h, w, n, input_size, save_dir)

def transform_and_save(pred_batch, scales, centers, heights, widths, names, input_size, save_dir):
    for i in range(len(pred_batch)):
        pred_out = pred_batch[i]
        h, w, s, c = heights[i], widths[i], scales[i], centers[i]
        pred = transform_parsing(pred_out, c, s, w, h, input_size=input_size)
        output_im = Image.fromarray(pred)
        output_im.putpalette(PALETTE)
        output_im.save(os.path.join(save_dir, names[i]+'.png'))


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    # options
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    gpus = [int(i) for i in args.gpu.split(',')]
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    # load data   
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    lip_dataset = InferDataSet(args.data_dir, args.image_ext, crop_size=input_size, transform=transform)
    num_samples = len(lip_dataset)
    valloader = data.DataLoader(lip_dataset, batch_size=args.batch_size * len(gpus), 
                                num_workers=args.num_workers, shuffle=False, pin_memory=True)
    # load model
    model = Res_Deeplab(num_classes=args.num_classes)
    restore_from = args.restore_from
    state_dict = model.state_dict().copy()
    state_dict_old = torch.load(restore_from)

    for key, nkey in zip(state_dict_old.keys(), state_dict.keys()):
        if key != nkey:
            # remove the 'module.' in the 'key'
            state_dict[key[7:]] = deepcopy(state_dict_old[key])
        else:
            state_dict[key] = deepcopy(state_dict_old[key])

    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    # infer and save result
    os.makedirs(args.save_dir, exist_ok=True)
    infer(model, valloader, input_size, num_samples, len(gpus), args.save_dir, args.mirror)
    print("Done.")

if __name__ == '__main__':
    main()
