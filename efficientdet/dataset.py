import os
import torch
import numpy as np
import pickle

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2


class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

class TobyCustom(Dataset):
    '''
    Config dataset to load the fourth channel image and concate it to the original image,
        and to easier to load my data.
    '''
    def __init__(self, root_dir, side_dir, annot_path, val = False ,transform=None):
        # root_dir: 'D:/Etri_tracking_data/Etri_full/image_4channels_vol1/
        self.root_dir = root_dir
        self.names = [int(i.split('.')[0]) for i in os.listdir(self.root_dir)]
        self.names.sort()

        # 'D:/Etri_tracking_data/Etri_full/image_vol1_Sejin/'
        self.side_dir = side_dir
        self.transform = transform
        with open(annot_path, 'r') as f:
            self.annot = f.readlines()
        self.classes = {'ROI': 0}
        self.labels = {0: 'ROI'}

    def __len__(self):
        number = len(self.names)
        if number > 1000:
            return number//3
        else:
            return number
        # return len(self.names)

        # return 10

    def __getitem__(self, idx):
        # close
        image_path = self.names[idx]
        image_path = str(image_path) + '.png'
        img = self.load_image(image_path)
        other_path = self.side_dir + image_path
        last_layer = cv2.imread(other_path, 0)
        last_layer = np.expand_dims(last_layer, axis = -1)
        # Forgot last time
        # last_layer/=255.
        img = np.concatenate((img,last_layer), axis = 2)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_path):
        path = self.root_dir + image_path
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
        # return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        annotation = [[float(i) for i in self.annot[image_index].split(',')] + [0]]
        return np.array(annotation)


def collater(data):
    ''' Convert numpy array to Tensor '''
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        # print('Resizer')
        # print(sample)
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 4))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Flip_X(object):
    ''' Flip image by X axis '''
    def __call__(self, sample, p=0.5):
        # print('Flip_X')
        # print(sample)
        if np.random.rand() < p:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample

class Flip_Y(object):
    ''' Flip image by Y axis '''
    def __call__(self, sample, p=0.5):
        # print('Flip_Y')
        # print(sample)
        if np.random.rand() < p:
            image, annots = sample['img'], sample['annot']
            image = image[:, :, ::-1]

            rows, cols, channels = image.shape

            y1 = annots[:, 1].copy()
            y2 = annots[:, 3].copy()

            y_tmp = y1.copy()

            annots[:, 1] = cols - y2
            annots[:, 3] = cols - y_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):
    ''' Normalize image by it means and std, this mean and std from COCO '''
    def __init__(self, mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5]):
        # mean=[0.485, 0.456, 0.406, 0] 
        # std=[0.229, 0.224, 0.225, 1]
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        # print('Normalizer')
        # print(sample)
        image, annots = sample['img'], sample['annot']
        image = image.astype(np.float32)
        image/=255.
        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}

class Equalize(object):
    ''' Equalized by it histogram '''
    def __call__(self, sample, p = 0.5):
        # print('Equalize')
        # print(sample)
        if np.random.rand() < p:
            image, annots = sample['img'], sample['annot']
            image = self.apply(image)
            sample = {'img': image, 'annot': annots}
        return sample

    def apply(self, img, mask=None):
        _,_,c = img.shape
        if c == 1:
            return cv2.equalizeHist(img)
        else:
            for i in range(c):
                img[... , i] = cv2.equalizeHist(img[..., i])
            return img

class Brightness(object):
    ''' Change brightness of image '''
    def __init__(self, brightness_limit = 0.2, constrast_limit = 0.2, brightness_by_max=True):
        assert constrast_limit >= 0
        self.beta = 0 + np.random.uniform(-brightness_limit, brightness_limit)
        self.beta_by_max = brightness_by_max

    def __call__(self, sample, p = 0.5):
        # print('Brightness')
        # print(sample)
        if np.random.rand() < p:
            image, annots = sample['img'], sample['annot']
            image = self.apply(image)
            sample = {'img': image, 'annot': annots}
        return sample 

    def apply(self, img):
        dtype = np.dtype("uint8")
        max_value = 255
        lut = np.arange(0, max_value + 1).astype("float32")
        if self.beta != 0:
            if self.beta_by_max:
                lut += self.beta * max_value
            else:
                lut += self.beta * np.mean(img)
        lut = np.clip(lut, 0, max_value).astype(dtype)
        img = cv2.LUT(img, lut)
        return img

class Constrast(object):
    ''' Change constrast of image '''
    def __init__(self, constrast_limit = 0.2):
        assert constrast_limit >= 0
        self.alpha = 1 + np.random.uniform(-constrast_limit, constrast_limit)

    def __call__(self, sample, p = 0.5):
        # print('Constrast')
        # print(sample)
        if np.random.rand() < p:
            image, annots = sample['img'], sample['annot']
            image = self.apply(image)
            sample = {'img': image, 'annot': annots}
        return sample 

    def apply(self, img):
        dtype = np.dtype("uint8")
        max_value = 255
        lut = np.arange(0, max_value + 1).astype("float32")
        if self.alpha != 1:
            lut *= self.alpha
        lut = np.clip(lut, 0, max_value).astype(dtype)
        img = cv2.LUT(img, lut)
        return img



from torchvision.transforms import Compose

class ComposeAlb(Compose):
    ''' Add the probability to augment an image in normal Compose from Pytorch '''
    def __init__(self, transforms, p = 0.5):
        self.augment = dict()
        self.transforms = transforms
        for t in transforms:
            self.augment[t.__class__.__name__] = t
        if 'Resizer' not in self.augment.keys() or 'Normalizer' not in self.augment.keys():
            raise Exception('Please check if there are Resizer and Normalizer or not? If you want that, please modify the code -> just need to comment those lines :3')
        self.p = p
        
    def __call__(self, img):
        if np.random.rand() < self.p:
            for n,a in self.augment.items():
                if n in ['Resizer', 'Normalizer']:
                    continue
                else:
                    img = a(img)
        img = self.augment['Normalizer'](img)
        img = self.augment['Resizer'](img)
        return img 

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


# class TobyCustom4COCO(Dataset):
#     '''
#     Config dataset to load the fourth channel image and concate it to the original image,
#         and to easier to load my data.
#     '''
#     def __init__(self, root_dir, side_dir, annot_path, val = False ,transform=None):
#         # root_dir: 'D:/Etri_tracking_data/Etri_full/image_4channels_vol1/
#         self.root_dir = root_dir
#         self.names = [int(i.split('.')[0]) for i in os.listdir(self.root_dir)]
#         self.names.sort()

#         # 'D:/Etri_tracking_data/Etri_full/image_vol1_Sejin/'
#         self.side_dir = side_dir
#         self.transform = transform
#         with open(annot_path, 'r') as f:
#             self.annot = f.readlines()
#         self.classes = {'ROI': 0}
#         self.labels = {0: 'ROI'}

#     def __len__(self):
#         return len(self.names)
#         # return 10

#     def __getitem__(self, idx):
#         # close
#         image_path = self.names[idx]
#         image_path = str(image_path) + '.png'
#         img = self.load_image(image_path)
#         other_path = self.side_dir + image_path
#         last_layer = cv2.imread(other_path, 0)
#         last_layer = np.expand_dims(last_layer, axis = -1)
#         # Forgot last time
#         # last_layer/=255.
#         img = np.concatenate((img,last_layer), axis = 2)
#         annot = self.load_annotations(idx)
#         sample = {'img': img, 'annot': annot}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample

#     def load_image(self, image_path):
#         path = self.root_dir + image_path
#         img = cv2.imread(path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         return img
#         # return img.astype(np.float32) / 255.

#     def load_annotations(self, image_index):
#         # annotation = [[float(i) for i in self.annot[image_index].split(',')] + [0]]
#         # return np.array(annotation)
        