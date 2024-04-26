import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np

from loss_utils import SSIM, BackprojectDepth, Project3D, disp_to_depth, get_smooth_loss, transformation_from_parameters
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from collections import OrderedDict


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]
        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1, num_channels=3):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1, num_channels=3):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images, num_channels=num_channels)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1, num_input_channels=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.num_channels = num_input_channels
        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images, self.num_channels)
            weight = self.encoder.conv1.weight.data.clone()
            self.encoder.conv1 = nn.Conv2d(self.num_channels * num_input_images, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.encoder = resnets[num_layers](pretrained)
            weight = self.encoder.conv1.weight.data.clone()
            self.encoder.conv1 = nn.Conv2d(self.num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        if self.num_channels == 3:
            x = (input_image - 0.45) / 0.225
        else:
            x = input_image
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

class Model(nn.Module):
    def __init__(self, max_disp, num_channels, frame_ids, batch_size):
        super().__init__()
        self.max_disp = max_disp
        self.deeplab = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=max_disp)
        self.deeplab.backbone.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.alpha = 0.4
        self.smoothness = 1e-4
        self.ssim = SSIM()

        self.num_channels = num_channels

        self.backproject_depth = {}
        self.project_3d = {}
        h = 384
        w = 512

        self.backproject_depth[0] = BackprojectDepth(batch_size, h, w)
        self.backproject_depth[0].cuda()

        self.project_3d[0] = Project3D(batch_size, h, w)
        self.project_3d[0].cuda()

        self.frame_ids = frame_ids

        self.pose_encoder = ResnetEncoder(18, False, num_input_images=len(frame_ids), num_input_channels=num_channels).to('cuda')
        self.pose_network = PoseDecoder(self.pose_encoder.num_ch_enc, 1, num_frames_to_predict_for=2).cuda()


    def forward(self, x):
        output = {}
        for i in range(len(self.frame_ids)):
            si = self.num_channels * i
            ei = self.num_channels * (i + 1)
            output[("depth", self.frame_ids[i], 0)] = self.deeplab(x[:, si:ei])['out']

            disp_prob = torch.sigmoid(output[("depth", self.frame_ids[i], 0)])
            output[("entropy", self.frame_ids[i], 0)] = -(disp_prob * torch.log(disp_prob)).sum(dim=1, keepdim=True)

        pose_enc_out = self.pose_encoder(x)
        axisangle, translation = self.pose_network([pose_enc_out])
        for i, f_i in enumerate(self.frame_ids[1:]):
            if f_i != "s":
                output[("axisangle", 0, f_i)] = axisangle
                output[("translation", 0, f_i)] = translation
                output[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, i], translation[:, i])

        return output

    def get_disparity(self, yhat):
        multiplier = 256 / self.max_disp
        y_pred = (torch.nn.Softmax(dim=1)(yhat) * multiplier * torch.arange(0, self.max_disp, device='cuda')[None, :, None, None]).sum(dim=1, keepdim=True)
        return y_pred

    
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        disp = outputs[("disp", 0)]
        disp = F.interpolate(
                disp, [384, 512], mode="bilinear", align_corners=False)

        _, depth = disp_to_depth(disp, 1.0, 100.0)

        outputs[("depth", 0, 0)] = depth

        for i, frame_id in enumerate(self.frame_ids[1:]):
            T = outputs[("cam_T_cam", 0, frame_id)]
            cam_points = self.backproject_depth[0](
                depth, inputs[("inv_K", 0)].cuda())

            pix_coords = self.project_3d[0](
                cam_points, inputs[("K", 0)].cuda(), T)

            outputs[("sample", frame_id, 0)] = pix_coords

            outputs[("color", frame_id, 0)] = F.grid_sample(
                inputs[("color", frame_id, 0)].cuda(),
                outputs[("sample", frame_id, 0)].cuda(),
                padding_mode="border")


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss.mean(dim=(1,2,3)).mean()

    def loss_fn(self, inputs, outputs, ytrue):
        # (Softmax(yhat) * arange(0, max_disp)).sum() = y_pred
        y_pred = self.get_disparity(outputs[('depth', 0, 0)])
        mask = ytrue > 0
        distance = y_pred - ytrue
        distance[torch.logical_not(mask)] = 0.0
        num_pixels = mask.sum()
        per_pixel_loss = distance.abs().mean(dim=1).sum() / (self.max_disp * num_pixels)

        outputs[('disp', 0)] = y_pred
        self.generate_images_pred(inputs, outputs)

        reprojection_loss = 0.0
        src_img = inputs[('color', 0, 0)].cuda()
        for i in range(1, len(self.frame_ids)):
            reprojection_loss += self.compute_reprojection_loss(outputs[('color', self.frame_ids[i], 0)],
                                                               src_img)

        input_sensor = "thermal" if self.num_channels == 1 else "color"
        smooth_loss = get_smooth_loss(y_pred, inputs[(input_sensor, 0, 0)].cuda())
        return (1 - self.alpha) * torch.mean(reprojection_loss) + self.alpha * per_pixel_loss + self.smoothness * smooth_loss
