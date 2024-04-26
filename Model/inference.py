import torch
from datasets import ThermalDataset
from loss_utils import BackprojectDepth
from model import Model
import argparse
import numpy as np
import os
import PIL
import PIL.Image as pil
from torchvision import transforms
from matplotlib import pyplot as plt
import skimage.transform
import cv2
import tqdm
import time


person_data = {
    'p0m2l0_213304_thermal.npy'
}


def get_point_cloud(disparity_map, none):
    h, w = disparity_map.shape

    image_width = 4096
    image_height = 3000
    K = np.array([[4643.03812, 0, 2071.019842, 0],
                        [0, 4643.03812, 1497.908013, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)

    K[0] /= image_width
    K[1] /= image_height

    bp = BackprojectDepth(1, h, w)
    inv_K = torch.from_numpy(np.linalg.pinv(K))

    depth = 1.0 / disparity_map
    depth[disparity_map < 10.0] = 0
    points3d = bp(depth, inv_K[None, ...])[0].T
    return points3d


import plotly.express as px

if __name__ == "__main__":
    parser = argparse.ArgumentParser("inference")
    parser.add_argument('-i', '--input', type=str, help="input file path or directory")
    parser.add_argument('-w', '--weights', type=str, help="directory containing model weights")
    parser.add_argument('--max_disp', type=int, default=64, help="maximum disparity used during training")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels expected for the input. (1=thermal, 3=rgb).")
    parser.add_argument('--load_gt', action='store_true', default=False, help="whether to load ground truth (assumes dataset has ground truth)")
    parser.add_argument('-o', '--output_dir', type=str, default='./debug', help="directory to store all outputs")

    args = parser.parse_args()

    data = None
    if os.path.isdir(args.input):
        inputs = [f'{os.path.join(args.input, x)}' for x in os.listdir(args.input) if x.endswith('.npy')]
    elif args.input.endswith('.npy'):
        inputs = [args.input]
    else:
        print("Invalid directory given. Please pre-process the dataset using the thermal-mono-generator.py script.")
        exit()


    model = Model(args.max_disp, args.num_channels, [0, -1, 1], 1).cuda()
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    times = []
    pbar = tqdm.tqdm(sorted(inputs))
    frame_num = 0
    for inp in pbar:
        pbar.set_description(f"Processing {inp}. Avg Inference Time {np.average(times).round(5)}s")
        fname = inp.split('/')[-1].replace('.npy', '')
        if args.num_channels == 1:
            data = ThermalDataset.load_thermal_data(inp)
        else:
            rgb = np.load(inp)
            data = transforms.ToTensor()(pil.fromarray(rgb, 'RGB'))

        start_time = time.time()
        data = data.cuda()
        pred_disp = model.deeplab(data[None, ...])['out']
        disp_prob = torch.sigmoid(pred_disp)
        disp = model.get_disparity(pred_disp).cpu().detach()[0, 0]
        entropy = -(disp_prob * torch.log(disp_prob)).sum(dim=1, keepdim=True).cpu().detach()
        end_time = time.time()
        times.append(end_time - start_time)

        disp[:args.max_disp] = 0

        sensor = "rgb" if args.num_channels == 3 else "thermal"

        plt.axis('off')
        plt.imshow(data.cpu().detach()[0], cmap='Greys')
        plt.savefig(f'{args.output_dir}/inputs/input_{frame_num:06d}.png', bbox_inches='tight')
        plt.close()

        plt.axis('off')
        plt.imshow(disp, cmap='inferno', vmin=0, vmax=disp.max())
        plt.savefig(f'{args.output_dir}/preds/pred_{frame_num:06d}.png', bbox_inches='tight')
        plt.close()

        if args.load_gt:
            gt = np.load(args.input[:-(len(sensor) + 4)] + 'gt.npy')
            gt = skimage.transform.resize(gt, (384,512), mode='constant')
            plt.axis('off')
            plt.imshow(gt, cmap='inferno', vmin=0, vmax=gt.max())
            plt.savefig(f'{args.output_dir}/gt/gt_{frame_num:06d}.png', bbox_inches='tight')
            plt.close()

        frame_num += 1

        # Only visualize point cloud when 1 input is given to save time.
        # Recommend to first run with the entire data, then choose specific samples to inspect point cloud.
        if len(inputs) == 1:
            pts3d = get_point_cloud(disp, None)

            fig = plt.figure()
            colors = data.cpu().detach().numpy()[0]

            if len(colors.shape) == 2:
                colors = colors * (255 / colors.max())
            
            colors = colors.reshape((-1,))
            fig = px.scatter_3d(x=pts3d[:, 0], y=pts3d[:, 1], z=pts3d[:, 2], color_continuous_scale="magma", color=colors)

            print("plotted.. now updating markers")
            fig.update_traces(marker={'size': 1})
            fig.write_html(f"{args.output_dir}/point_cloud_{fname}.html")
            print(f"Point cloud saved as {args.output_dir}/point_cloud_{fname}.html")
