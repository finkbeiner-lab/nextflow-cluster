import os
import sys
from glob import glob

# sys.path.append('/app/code')

import numpy as np

from cellpose import core, io, metrics, models
import argparse

class CellposeModel:
    base_dir = os.path.join('/app', 'data')

    model_types = [
        *[f'CP{"x" * i}' for i in range(2)],
        *[f'cyto{"2" * i}' for i in range(2)],
        *[f'LC{i}' for i in range(1, 4 + 1)],
        'livecell',
        'nuclei',
        'scratch',
        'tissuenet',
        *[f'TN{i}' for i in range(1, 3 + 1)],
    ]

    def __init__(self, root_dir, model_name, model_type=None):
        assert core.use_gpu(), 'GPU not available'

        root_dir = os.path.join(self.__class__.base_dir, root_dir)
        assert os.path.isdir(root_dir), 'Root directory not found'
        self.root_dir = root_dir

        assert model_name is not None, 'Model name not provided'
        self.model_name = model_name
        model_dir = os.path.join(self.root_dir, 'models')
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=False)
        assert os.path.isdir(model_dir), 'Could not create model directory'
        model_path = os.path.join(model_dir, model_name)

        self.model = None
        if model_type is not None and model_type in self.__class__.model_types:
            self.model = models.CellposeModel(gpu=True, model_type=model_type)
        elif os.path.isfile(model_path):
            self.model = models.CellposeModel(gpu=True, pretrained_model=model_path)


    def load_data(self, img_dir, mask_dir=None, channel=0, channel2=0, img_ext='tif'):
        assert set([channel, channel2]).issubset(list(range(4))), 'Valid channel values not provided'
        self.channels = [channel, channel2]

        img_dir = os.path.join(self.root_dir, img_dir)
        assert os.path.isdir(img_dir), 'Image directory not found'
        imgs, names = list(), list()
        for img_path in glob(os.path.join(img_dir, f'*.{img_ext}')):
            imgs.append(io.imread(img_path))
            names.append('.'.join(os.path.basename(img_path).split('.')[:-1]))
        self.imgs, self.names, self.masks = imgs, names, None
        if mask_dir is not None:
            mask_dir = os.path.join(self.root_dir, mask_dir)
            assert os.path.isdir(mask_dir), 'Mask directory not found'
            masks = list()
            for name in self.names:
                mask_path = os.path.join(mask_dir, name + '_seg.npy')
                assert os.path.isfile(mask_path), 'Mask file not found'
                mask_dict = np.load(mask_path, allow_pickle=True).item()
                assert 'masks' in mask_dict.keys(), 'Mask entry not found'
                masks.append(mask_dict.get('masks'))
            self.masks = masks

    def train(self, epochs=100, learning_rate=0.1, weight_decay=0.0001):
        assert self.model is not None, 'Model not provided'
        assert self.imgs is not None, 'Images not provided'
        assert self.masks is not None, 'Masks not provided'

        return self.model.train(
            self.imgs,
            self.masks,
            save_path=self.root_dir,
            model_name=self.model_name,
            channels=self.channels,
            n_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ), self.model.diam_labels.item()

    def eval(self, out_dir, diameter=30.0, flow_threshold=0.4, cellprob_threshold=0.0, ap_threshold=0.5):
        assert self.model is not None, 'Model not provided'
        assert self.imgs is not None, 'Images not provided'

        out_dir = os.path.join(self.root_dir, out_dir)
        assert os.path.isdir(out_dir), 'Output directory not found'

        masks, flows, styles = self.model.eval(self.imgs, channels=self.channels, diameter=diameter, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)
        aps = [None] * len(masks)
        if self.masks is not None:
            aps = np.array(metrics.average_precision(self.masks, masks, threshold=[ap_threshold]))[..., 0].transpose(1, 0)

        out_paths = list()
        for img, name, mask, flow, style, ap in zip(self.imgs, self.names, masks, flows, styles, aps):
            out_dict = dict(
                img=img,
                masks=mask,
                chan_choose=self.channels,
                ismanual=False,
                flows=flow,
                style=style,
                est_diam=diameter,
            )
            if self.masks is not None:
                out_dict.update(ap=ap)
            out_path = os.path.join(out_dir, name + '_seg.npy')
            np.save(out_path, out_dict)
            out_paths.append(out_path)

        return out_paths

if __name__ == '__main__':
    pass