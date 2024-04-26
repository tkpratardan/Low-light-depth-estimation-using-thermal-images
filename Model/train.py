import numpy as np
import torch
import argparse
import os

from torch.utils.data.dataloader import DataLoader

import tqdm
from model import Model
import datasets
import wandb

USE_WANDB = True


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def get_nearest_frame(fid, all_ids):
    dist = np.array(all_ids) - fid
    return all_ids[np.argmin(np.abs(dist))]


def create_train_val_split(all_files, ratio):
    camera_frame_map = {}
    for f in all_files:
        module_id = f[3]
        if module_id not in camera_frame_map:
            camera_frame_map[module_id] = []
        camera_frame_map[module_id].append(f)

    all_train = []
    all_valid = []

    for camera in camera_frame_map:
        num_files = len(camera_frame_map[camera])
        num_train = num_files * ratio
        num_val = num_files - num_train

        val_step = int(num_files / num_val) * 3
        indices = np.array([False] * num_files)

        for i in range(1, num_files, val_step):
            indices[i] = True
            indices[i - 1] = True
            indices[i + 1] = True


        frames = np.array(camera_frame_map[camera])
        all_train.extend(frames[np.logical_not(indices)])
        all_valid.extend(frames[indices])

    return all_train, all_valid


def prune_start_end_sequences(fnames):
    ids = [int(f.split(" ")[1]) for i,f in enumerate(fnames)]
    valid = []

    frame_counter = 0
    for i in range(len(fnames)):
        fid = int(fnames[i].split(' ')[1])
        # nearest_fid = get_nearest_frame(fid, ids)
        valid.append(f"{fnames[i].split(' ')[0]} {frame_counter}")
        frame_counter += 1

    return valid


def get_dataloaders(batch_size, num_workers, frame_ids):
    datapaths = {
        'front': 'thermal_front_block7',
        'side': 'thermal_mono_norm',
        'front_plain': 'thermal_mono_front',
        'all': 'thermal_all_block7'
    }
    splits = {
        'front': "train_front_block7_files.txt",
        'side': "train_files.txt",
        'front_plain': "thermal_front_plain.txt",
        'all': 'train_all_block7_files.txt'
    }

    mode = 'all'

    datapath = f"/aidtr-mltrain/{datapaths[mode]}"
    fpath = os.path.join("splits/thermal", splits[mode])

    model_height = 384
    model_width = 512
    dataset = datasets.ThermalDataset

    all_files = np.array(readlines(fpath.format("train")))

    train_filenames, val_filenames = create_train_val_split(all_files, 0.8)

    ids = {f.split(" ")[1] for i,f in enumerate(train_filenames)}
    failed = 0
    for f in train_filenames:
        fid = int(f.split(' ')[1])
        if str(fid - 1) not in ids or str(fid + 1) not in ids:
            failed += 1

    print(f"Failed to find consecutive frames for {failed}/{len(train_filenames)}")
    img_ext = '.png'

    print(f"Total number of samples: {len(train_filenames)} (Train), {len(val_filenames)} (Val)")

    train_dataset = dataset(
        datapath, train_filenames, model_height, model_width,
        frame_ids, 1, is_train=True, img_ext=img_ext)
    train_loader = DataLoader(
        train_dataset, batch_size, True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dataset = dataset(
        datapath, val_filenames, model_height, model_width,
        frame_ids, 1, is_train=False, img_ext=img_ext)
    val_loader = DataLoader(
        val_dataset, batch_size, False,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    return train_loader, val_loader


def compute_depth_metrics(pred, gt):
    with torch.no_grad():
        mask = gt > 0
        gt = gt[mask]
        pred = pred[mask]

        thresh = torch.max((gt / pred), (pred / gt))
        a1 = (thresh < 1.25     ).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()

        rmse = (gt - pred) ** 2
        rmse = torch.sqrt(rmse.mean())

        rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
        rmse_log = torch.sqrt(rmse_log.mean())

        abs_rel = torch.mean(torch.abs(gt - pred) / gt)

        sq_rel = torch.mean((gt - pred) ** 2 / gt)

        metrics = {'abs_rel': abs_rel.cpu(), 'sq_rel': sq_rel.cpu(), 'rmse': rmse.cpu(), 'log_rmse': rmse_log.cpu(), 'a1': a1.cpu(), 'a2': a2.cpu(), 'a3': a3.cpu()}
    return metrics



def train_loop(exp_name, num_epochs, val_frequency, max_disp, batch_size, num_workers, frame_ids, lr, use_thermal):
    num_channels = 1 if use_thermal else 3
    input_sensor = "thermal" if use_thermal else "color"
    model = Model(max_disp, num_channels, frame_ids, batch_size)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DistributedDataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_dataloader, val_dataloader = get_dataloaders(batch_size, num_workers, frame_ids)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.7)
    epoch = 0
    best_acc = 0.0
    while epoch < num_epochs:
        avg_train_loss = 0.0
        metrics = {}
        train_acc_metrics = {}
        for inputs in tqdm.tqdm(train_dataloader):
            xb = torch.cat([inputs[(input_sensor, i, 0)] for i in frame_ids], dim=1).cuda()
            yb = inputs["depth_gt"].cuda()
            outputs = model(xb)
            optimizer.zero_grad()
            pure_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            loss = pure_model.loss_fn(inputs, outputs, yb)
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()

            yhat = outputs[('disp', 0)]
            entropy = outputs[('entropy', 0, 0)]
            batch_metrics = compute_depth_metrics(yhat, yb).items()
            for k, v in batch_metrics:
                if k not in train_acc_metrics:
                    train_acc_metrics[k] = 0.0
                train_acc_metrics[k] += v
            del inputs, outputs

        # Train metrics
        metrics['train/images/input'] = wandb.Image(xb[[0, -1], :num_channels], caption="Input Images")
        metrics['train/images/gt'] = wandb.Image(yb[[0, -1]], caption="Ground Truth")
        metrics['train/images/pred'] = wandb.Image(yhat[[0, -1]], caption="Predicted Disparity")
        metrics['train/images/entropy'] = wandb.Image(entropy[[0, -1]], caption="Entropy")

        val_acc_metrics = {}
        with torch.no_grad():
            avg_validation_loss = 0.0     
            for inputs in tqdm.tqdm(val_dataloader):
                xb = torch.cat([inputs[(input_sensor, i, 0)] for i in frame_ids], dim=1).cuda()
                yb = inputs["depth_gt"].cuda()
                outputs = model(xb)
                pure_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                loss = pure_model.loss_fn(inputs, outputs, yb)
                avg_validation_loss += loss.item()
                yhat = outputs[('disp', 0)]            
                entropy = outputs[('entropy', 0, 0)]

                batch_metrics = compute_depth_metrics(yhat, yb).items()
                for k, v in batch_metrics:
                    if k not in val_acc_metrics:
                        val_acc_metrics[k] = 0.0
                    val_acc_metrics[k] += v
                del inputs, outputs

        print(f"Epoch: {epoch}/{num_epochs}, Training Loss: {avg_train_loss / len(train_dataloader)}," 
              f"Validation Loss: {avg_validation_loss / len(val_dataloader)}")

        # Scalar Metrics
        metrics['train/loss'] = avg_train_loss / len(train_dataloader)
        metrics['val/loss'] = avg_validation_loss / len(val_dataloader)
        for k,v in train_acc_metrics.items():
            metrics[f'train/{k}'] = v / len(train_dataloader)
        for k, v in val_acc_metrics.items():
            metrics[f'val/{k}'] = v / len(val_dataloader)
        metrics['epoch'] = epoch

        y_disp = (yhat).cpu()
        # Images
        metrics['val/images/input'] = wandb.Image(xb[[0, -1], :num_channels], caption="Input Images")
        metrics['val/images/gt'] = wandb.Image(yb[[0, -1]], caption="Ground Truth")
        metrics['val/images/pred'] = wandb.Image(y_disp[[0, -1]], caption="Predicted Disparity")
        metrics['val/images/entropy'] = wandb.Image(entropy[[0, -1]], caption="Entropy")

        # Histogram plots
        metrics[f'val/hist/input_hist'] = wandb.Histogram(xb.cpu())
        y_values = yb.cpu()
        inds = np.where(y_values > 0)
        metrics[f'val/hist/gt_hist'] = wandb.Histogram(y_values[inds])
        metrics[f'val/hist/pred_hist'] = wandb.Histogram(y_disp)

        metrics[f'val/hist_local/input_hist0'] = wandb.Histogram(xb[0].cpu())
        metrics[f'val/hist_local/input_hist1'] = wandb.Histogram(xb[-1].cpu())
        ybcpu = yb.cpu()
        metrics[f'val/hist_local/gt_hist0'] = wandb.Histogram(ybcpu[0][np.where(ybcpu[0] > 0)])
        metrics[f'val/hist_local/gt_hist1'] = wandb.Histogram(ybcpu[-1][np.where(ybcpu[-1] > 0)])
        metrics[f'val/hist_local/pred_hist0'] = wandb.Histogram((yhat).cpu()[0])
        metrics[f'val/hist_local/pred_hist1'] = wandb.Histogram((yhat).cpu()[-1])

        if USE_WANDB:
            wandb.log(metrics)

        model_prefix = "thermal_stereogt_" + exp_name
        torch.save(model.state_dict(), f'./weights/{model_prefix}_model_latest.pt')
        if metrics['val/a1'] > best_acc:
            best_acc = metrics['val/a1']
            torch.save(model.state_dict(), f'./weights/{model_prefix}_model_best.pt')

        epoch += 1
        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--max_disp", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_thermal", action='store_true', default=True)

    args = parser.parse_args()

    num_gpu = torch.cuda.device_count() 
    max_disp = 64
    batch_size = 8 * num_gpu # 192
    print("Batch size is ", batch_size)
    num_workers = 16
    num_epochs = 40
    lr = 1e-4
    frame_ids = [0, -1, 1]
    use_thermal = True

    exp_name = args.name
    if USE_WANDB:
        wandb.init(project="monothermal", group="DDP", name=exp_name)
    train_loop(exp_name, num_epochs, 1, max_disp, batch_size, num_workers, frame_ids, lr, use_thermal)

    
