from utils.utils import *
from efficientdet.dataset import *

class TobyTestCustom(Dataset):
    def __init__(self, root_dir, side_dir, annot_path = None, transform = transform):
        # root_dir: 'D:/Etri_tracking_data/Etri_full/image_4channels_vol1/
        self.root_dir = root_dir
        # 'D:/Etri_tracking_data/Etri_full/image_vol1_Sejin/'
        self.side_dir = side_dir

        if annot_path:
            with open(annot_path, 'r') as f:
                self.annot = f.readlines()

        self.image_nums = 1020
        self.classes = {'ROI': 0}
        self.labels = {0: 'ROI'}
    
    def __len__(self):
        return self.image_nums
        # return 10

    def __getitem__(self, idx):
        idx += 16980
        img_path = self.root_dir + str(idx) + '.png'        
        other_path = self.side_dir + str(idx) + '.png'

        last_layer = cv2.imread(other_path, 0)
        last_layer = np.expand_dims(last_layer, axis = -1)
        img = np.concatenate((img,last_layer), axis = 2)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        path = self.root_dir + str(image_index) + '.png'
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        annotation = [[float(i) for i in self.annot[image_index].split(',')] + [0]]
        # print(np.array(annotation))
        return np.array(annotation)

        