import os
import time
import numpy as np
import cv2
import torch
from torchvision import transforms
from .nets import S3FDNet
from .box_utils import nms_
import torch.nn.functional as F

from liveavatar.utils.device_backend import autocast as device_autocast
from liveavatar.utils.device_backend import env_device_backend, resolve_device_backend

PATH_WEIGHT = '/tmp/pretrained/sfd_face.pth'
if not os.path.exists(PATH_WEIGHT):
    raise ValueError(f"SFD_ckpt not found in {PATH_WEIGHT}! download it from https://drive.google.com/file/d/1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt")
img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')


class S3FD():

    def __init__(self, device=None):

        tstamp = time.time()
        if device is None:
            backend = resolve_device_backend(env_device_backend())
            device = backend
        self.device = torch.device(device)

        print('[S3FD] loading with', self.device)
        self.net = S3FDNet(device=str(self.device)).to(self.device)
        state_dict = torch.load(PATH_WEIGHT, map_location=str(self.device), weights_only=False)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        print('[S3FD] finished loading (%.4f sec)' % (time.time() - tstamp))
        self.img_mean = torch.FloatTensor([104., 117., 123.]).to(self.device)

    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        w, h = image.shape[1], image.shape[0]

        bboxes = np.empty(shape=(0, 5))

        with torch.no_grad():
            for s in scales:
                scaled_img = cv2.resize(image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

                scaled_img = np.swapaxes(scaled_img, 1, 2)
                scaled_img = np.swapaxes(scaled_img, 1, 0)
                scaled_img = scaled_img[[2, 1, 0], :, :]
                scaled_img = scaled_img.astype('float32')
                scaled_img -= img_mean
                scaled_img = scaled_img[[2, 1, 0], :, :]
                x = torch.from_numpy(scaled_img).unsqueeze(0).to(self.device)
                y = self.net(x)

                detections = y.data
                scale = torch.Tensor([w, h, w, h]).to(self.device)

                for i in range(detections.size(1)):
                    j = 0
                    while detections[0, i, j, 0] > conf_th:
                        score = detections[0, i, j, 0]
                        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        bbox = (pt[0], pt[1], pt[2], pt[3], score)
                        bboxes = np.vstack((bboxes, bbox))
                        j += 1

            keep = nms_(bboxes, 0.1)
            bboxes = bboxes[keep]

        return bboxes

    def detect_faces_batch(self, images, conf_th=0.8, scale=1.0):
        b, _, h, w = images.shape
        scale = (torch.Tensor([w, h, w, h]) / scale).to(self.device)

        bboxes_ = []
        with torch.no_grad():
            images = torch.flip(images, [1])
            images = images - self.img_mean[None, :, None, None]
            images = torch.flip(images, [1])
            with device_autocast(images.device.type, dtype=torch.float16, enabled=(images.device.type != 'cpu')):
                y = self.net(images)

            for i_img in range(b):
                bboxes = np.empty(shape=(0, 5))
                detections = y.data

                for i in range(detections.size(1)):
                    j = 0
                    while detections[i_img, i, j, 0] > conf_th:
                        score = detections[i_img, i, j, 0]
                        pt = (detections[i_img, i, j, 1:] * scale).cpu().numpy()
                        bbox = (pt[0], pt[1], pt[2], pt[3], score)
                        bboxes = np.vstack((bboxes, bbox))
                        j += 1

                keep = nms_(bboxes, 0.1)
                bboxes = bboxes[keep]
                bboxes_.append(bboxes)
        return bboxes_
