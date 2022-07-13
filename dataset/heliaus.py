import os
import lmdb
import torch
import compress_pickle as pickle
import bisect
import numpy as np
from PIL import Image, ImageFile
from pathlib import Path
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as T
from torchvision.transforms import functional as F
from tqdm import tqdm
from utils.build_data import transform
from torch.utils.data import DataLoader
from pycocotools.coco import COCO    

import pdb
import sys

# Class remapping - person is merged with people, car with special vehicle
remapping_cat = {0:1, 1:2, 2:1, 3:3, 4:1, 5:4, 6:5}


class fpdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def iterate_dir(path_dir, endswith=None, filter=[]):
    '''
    iterate sub folders, returns an scandir object
    filter: list of folder names to explore instead of exploring all folders
    '''
    for sub_dir in os.scandir(path_dir):
        if filter:
            if sub_dir.name in filter:
                yield sub_dir
        
        elif endswith:
            if sub_dir.name.endswith(endswith):
                yield sub_dir
        
        else:
            yield sub_dir


def extract_timestamp(filename):
    """
    extract timestamp from a filename.
    example: 5112_1599678178664110.png > int(1599678178664110)
    """
    if "_" in filename:
        # lwir filename
        ts = filename.split('_')[1].split('.')[0]
    else:
        # rgb filename
        ts = filename.split('.')[0]
    return int(ts)


def gen_sorted_image_pairs(rgb_dir, lwir_dir, return_labels=True, threshold=0.1e6):
    """
    Parse two folders containing rgb and thermal images and yield close matching pairs.
    RGB folder dictate the number of images

    returns: 
        - fp_rgb_img, fp_rgb_lbl, fp_lwi_img, fp_lwi_lbl (return_labels=True): 
        - fp_rgb_img, fp_lwi_img (return_labels=False)
    
    rgb_dir (str): rgb image dir
    lwir_dir (str): lwir image dir
    return_labels (bool): return annotation filepath if their exists
    threshold (float): maximum difference (in micro second) between an rgb and a lwir image 
    """
    # timestamp is in micro seconds (us)
    dic = {'lwi': {'dir': lwir_dir, 'timestamps': {}}, 
           'rgb': {'dir': rgb_dir, 'timestamps': {}}}
    
    # parse rgb and lwir images to extract their timestamps (in us)
    # each timestamps (integer) maps to the image and label absolute path
    for name in dic:
        for f in iterate_dir(dic[name]['dir'], endswith=".png"):
            ts = extract_timestamp(f.name)
            dic[name]['timestamps'][ts] = [f.path]

            if return_labels:
                fp_lbl = f.path.replace('.png', '.txt')
                fp_lbl = fp_lbl if os.path.isfile(fp_lbl) else ""
                dic[name]['timestamps'][ts].append(fp_lbl)

    # we sort lwir images by timestamps and we loop in rgb images to find the closest match
    l_ts_lwi = sorted(dic['lwi']['timestamps'].keys())
    l_ts_rgb = sorted(dic['rgb']['timestamps'].keys())
    for ts_lwi in l_ts_lwi:
        idx = bisect.bisect_left(l_ts_rgb, ts_lwi)
        # print(idx)
        ts_rgb = l_ts_rgb[idx-1]
        if np.abs(ts_lwi - ts_rgb) < threshold:
            rgb_pair = dic['rgb']['timestamps'][ts_rgb]
            lwi_pair = dic['lwi']['timestamps'][ts_lwi]
            pair = rgb_pair + lwi_pair
            yield pair


def gen_aligned_images(root='/mnt/heliaus/', filter=[]):
    """
    generator that yield aligned pairs of rgb and thermal images.
    timestamps of rgb and lwir images are not stricly identical, we perform bisect search to match the pairs
    
    root: has to be '/mnt/heliaus/' for now
    filter: list of folder names where the images are. Faster than exploring all folders
    """
    
    dic_rgb_lwir = {'706': '772', '707': '773', '708': '770', '709': '771'}
    dic_lwir_rgb = {dic_rgb_lwir[k]:k for k in dic_rgb_lwir}

    folders = ['20200908_Lindau_Daytime', '20200909_Lindau_Nighttime', '20201021_Lindau_Friedrichshafen_Nighttime', '20201021_Lindau_Friedrichshafen_Daytime']


    for dir in iterate_dir(root, filter=['disk1', 'disk2']): # disk1, disk2, etc...
        for sub_dir in iterate_dir(dir.path, filter=folders): # 20201021_Lindau_Friedrichshafen_Daytime, 20200909_Lindau_Nighttime, etc...

            category = sub_dir.name.split('_')[-1] # Daytime or Nighttime
            gen_subf = iterate_dir(sub_dir.path, filter=filter) if filter else iterate_dir(sub_dir.path, endswith="_l")

            for subf in gen_subf: # '20200908_134212_l', '20200908_134513' etc...
                
                gen_subsubf = iterate_dir(subf.path, endswith="_l")
                
                for subsubf in gen_subsubf: # 'xxx_LWIR_l', 'xxx_LWIR_l', 'xxx_LWIR_l', etc...
                    prefix = subsubf.name.split('_')[0]
                    
                    dir_lwi = "{}/{}_enhanced_sharpened_update".format(subsubf.path, prefix)
                    # find RGB folder from LWIR prefix: 770, 771, 772 etc...
                    dir_rgb = "{}/{}_RGB/{}_cropped".format(subf.path, dic_lwir_rgb[prefix], dic_lwir_rgb[prefix])
                    
                    gen_pairs = gen_sorted_image_pairs(rgb_dir, lwir_dir, return_labels=True, threshold=0.5e6)
                    for fp_rgb_img, fp_rgb_lbl, fp_lwi_img, fp_lwi_lbl in gen_pairs:
                        if fp_rgb_lbl and fp_lwi_lbl:
                            yield fp_rgb_img, fp_rgb_lbl, fp_lwi_img, fp_lwi_lbl, category, subf.name


def init_coco(annotations_file='/mnt/heliaus/free_road_annotations/mscoco_anno.json'):
    coco = COCO(annotations_file)
    images = coco.loadImgs(coco.getImgIds())
    image_id_dict = {i['file_name']:i['id'] for i in images}
    print(f"Found {len(image_id_dict)} images annotated with coco annotator")
    return coco, image_id_dict

def parse_polygon_labels(coco, image_id_dict, ty, fps, road_label=6, width=640, height=480):
    """
    given a coco object and a image_id, parse the coco formatted polygonal segmentation and return a numpy array mask label
    currently only lwir annotations for free road class are available
    """
    image_id = get_coco_image_id(image_id_dict, coco, fps)
    if image_id is None:
        return ty, False

    # initialize empty mask
    mask = np.zeros((height, width))

    # search for all annotations with given image_id
    ann_ids = coco.getAnnIds(image_id)
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        # annotation fix: segmentation should be a list of lists
        ann['segmentation'] = [ann['segmentation']]
        # get a numpy binary mask for this annotation
        mask += coco.annToMask(ann)

    mask = mask.astype(bool)*1
    ty[mask>0] = road_label

    return ty, True

def get_coco_image_id(image_id_dict, coco, fps):
    if coco and image_id_dict:
        lwir_split = fps[1].split('/')
        #print(fps[1])
        #print(len(lwir_split))
        seq_num = lwir_split[5].replace('_l','')
        lwir_num = lwir_split[7][:3]
        frame_num = lwir_split[8][:-4]
        image_name = f'images/{seq_num}/LWIR/{lwir_num}/{frame_num}.png'
        coco_image_id = image_id_dict.get(image_name, None) # get lwir image id from name
    return coco_image_id


def parse_bbox_labels(ty, fps, width, height):
    for fp in fps:
        if fp:
            #print(f"found labels for: {fps}")
            for line in open(fp).readlines():
                cat, x, y, w, h = line[:-1].split(" ")
                cat = remapping_cat[int(cat)]
                x = int(float(x) * width)
                y = int(float(y) * height)
                w = int(float(w) * width)
                h = int(float(h) * height)
                # broadcast
                # attention: <x> <y> - are center of rectangle (are not top-left corner) 
                ty[max(y-h//2,0):min(y+h//2,height-1), max(x-w//2,0):min(x+w//2, width-1)] = cat
        else:
            #print(f"no labels for: {fps}")
            pass
    return ty

def parse_labels(fps=[], width=640, height=480, image_id_dict=None, coco=None, annotation_type='bbox_poly'):
    """
    parse annotation files (fps) and return a numpy array mask label
    combines rgb and lwir annotations
    """
        
    ty = np.zeros((height, width))
    if annotation_type == 'bbox':
        # get only bounding boxes annotation
        ty = parse_bbox_labels(ty, fps, width, height)
    elif annotation_type == 'bbox_poly':
        # gets first polygon label, then bounding box
        ty, polygon_exists =  parse_polygon_labels(coco, image_id_dict, ty, fps, width=width, height=height)
        # for consistent labeling: only gets bounding box if there is a polygon annotation
        if polygon_exists:
            ty = parse_bbox_labels(ty, fps, width, height)

    ty = np.uint8(ty)
    if np.sum(ty) == 0:
         return None
    return F.to_pil_image(ty)


def create_csv(root, split):
    """
    explore image folder and write their paths into 'train.csv', 'val.csv' or 'test.csv'
    filter: list of folder explore (explore all sub folders by default)
    """
    map_dir = {
        'train': [], # all other folders
        'val': ['20201022_132638_l', '20201022_133527_l', '20201021_154127_l', '20200909_192704_l', '20201021_173129_l', '20201021_165814_l', '20201021_173329_l', '20201022_140433_l'],
        'test_night' : ['20201021_165614_l', '20201021_171716_l', '20201021_174854_l', '20201021_172528_l'],
        'test_day': ['20200908_143747_l', '20201022_134831_l', '20201022_132838_l']
    }
    map_dir['test'] = map_dir['test_night'] + map_dir['test_day']


    gen_pairs = gen_aligned_images(root, map_dir[split])

    fp = os.path.join(root, split + ".csv")
    f = open(fp, 'w')

    file_log = tqdm(position=0, unit=' images pairs', desc='creating ({}) with'.format(fp))

    for data in gen_pairs:
        parent_folder = data[-1] # 20201022_133527_l, ..., help to write data in the proper set (train, val or test)
        data = data[:-1]

        if (parent_folder in map_dir['test_night']) or (parent_folder in map_dir['test_day']):
            if split == 'test':
                f.write(",".join(data) + "\n")
                file_log.update(1)
        
        elif parent_folder in map_dir['val']:
            if split == 'val':
                f.write(",".join(data) + "\n")
                file_log.update(1)
        
        elif split == 'train':
            f.write(",".join(data) + "\n")
            file_log.update(1)

    return fp


class CsvDataset():

    def __init__(self, csv_path, unlabelled=False, annotation_type='bbox_poly'):
        
        self.data = []
        self.coco, self.image_id_dict = init_coco()
        self.unlabelled = unlabelled
        self.annotation_type = annotation_type
        with open(csv_path) as f:
            self.data = [line[:-1].split(',') for line in f]


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        rgb_fp, rgb_lbl_fp, lwi_fp, lwi_lbl_fp, cat = self.data[index]

        # There is weird bug where some images fails to be read from remote mounted disk
        # Thus we use try/except
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        #if True:
        try:
            rgb_pil = Image.open(rgb_fp).resize((640, 480), Image.BILINEAR)
            lwi_pil = Image.open(lwi_fp)
            lbl_pil = parse_labels([rgb_lbl_fp, lwi_lbl_fp], width=640, height=480, coco=self.coco, image_id_dict=self.image_id_dict, annotation_type=self.annotation_type)
            if lbl_pil is None and self.unlabelled:
                return rgb_pil, lwi_pil, cat
            elif lbl_pil is not None:
                return rgb_pil, lwi_pil, lbl_pil, cat
            else:
                return None, None, None, None
        #else:
        except:
            print(f"Failed to open {rgb_fp} or {lwi_fp}")
            return None, None, None, None


class LmdbDataset():
    '''
    sudo mount -t cifs -o username=git-smb,password=gitsmb123,uid=uid,gid=gid //samba.anotherbrain.lan/heliaus /mnt/heliaus
    '''
    def __init__(self, root='/mnt/heliaus/', lmdb_dir='dataset/heliaus', category=['Daytime', 'Nighttime'], split='train', unlabelled=False, annotation_type='bbox_poly'):
        self.root = root
        self.lmdb_dir = lmdb_dir
        self.split = split
        self.unlabelled = unlabelled
        self.annotation_type = annotation_type
        
        csv_fp = "{}/{}.csv".format(root,split)

        if unlabelled:
            lmdb_fp = "{}/{}_unlabelled.lmdb".format(lmdb_dir, split)
        else:
            lmdb_fp = "{}/{}_{}.lmdb".format(lmdb_dir, split, annotation_type)
            #lmdb_fp = "{}/{}.lmdb".format(lmdb_dir, split)


        if not os.path.exists(csv_fp):
            csv_fp = create_csv(root, split)
        
        if not os.path.exists(lmdb_fp):
            self.create_lmdb(lmdb_fp, csv_fp)

        self.db = lmdb.open(lmdb_fp, readonly=True, lock=False, readahead=False, meminit=False)
        with self.db.begin(write=False) as txn:
            self.indexes = []
            for cat in category:
                index = pickle.loads(txn.get(cat.encode("utf-8")), compression='pickle')
                self.indexes.extend(index)


    def create_lmdb(self, lmdb_fp, csv_fp):
        
        dataset = CsvDataset(csv_fp, unlabelled=self.unlabelled, annotation_type=self.annotation_type)
        loader = DataLoader(dataset, batch_size=15, num_workers=8, collate_fn=lambda x: x)

        db = lmdb.open(lmdb_fp, map_size=(1 << 40))

        dic = {'Daytime': [], 'Nighttime':[]}
        entries_to_write = []
        lmdb_index = 0
        for batch in tqdm(loader, total=len(loader), desc='creating {}'.format(lmdb_fp)):
            for sample in batch:
                # ugly fix to filter images that can't be read by PIL
                if sample[0]:
                    k, v = str(lmdb_index).encode("utf-8"), pickle.dumps(sample, compression='pickle')
                    cat = sample[-1] # 'Daytime' or 'Nighttime'
                    dic[cat].append(lmdb_index)
                    entries_to_write.append((k,v))
                    lmdb_index += 1
            
            # write periotically to prevent memory issue
            with db.begin(write=True) as txn:
                for k, v in entries_to_write:
                    txn.put(k, v)
            entries_to_write = []

        with db.begin(write=True) as txn:
            k, v = "length".encode("utf-8"), pickle.dumps(lmdb_index, compression='pickle')
            txn.put(k, v)
            for name in dic:
                k, v = name.encode("utf-8"), pickle.dumps(dic[name], compression='pickle')
                txn.put(k, v)
        db.close()


    def __len__(self):
        return len(self.indexes)


    def __getitem__(self, index):
        lmdb_idx = self.indexes[index]

        with self.db.begin(write=False) as txn:
            sample = pickle.loads(txn.get(str(lmdb_idx).encode("utf-8")), compression='pickle')

        if self.unlabelled:
            rgb_pil, lwi_pil, cat =  sample[:3]
            return rgb_pil, lwi_pil, None, cat
        else:
            rgb_pil, lwi_pil, lbl_pil, cat =  sample
            return rgb_pil, lwi_pil, lbl_pil, cat


class SequenceDataset():
    '''
    Read folder full of rgb images, search for it's lwir siblings, sort and return them as PIL image
    '''
    def __init__(self, rgb_dir='/mnt/heliaus/disk2/20200909_Lindau_Nighttime/20200909_190408/708_RGB/708_cropped'):

        dic_rgb_lwir = {'706': '772', '707': '773', '708': '770', '709': '771'}
        # dic_lwir_rgb = {dic_rgb_lwir[k]:k for k in dic_rgb_lwir}

        rgb_dir = Path(rgb_dir)
        ### TODO: merge categories here
        self.cat = rgb_dir.parent.parent.parent.name.split('_')[-1]

        idx_rgb = rgb_dir.name.split("_")[0] # '706', '707', '708', etc...
        idx_lwi = dic_rgb_lwir[idx_rgb]
        
        lwi_dir = "{}/{}_LWIR/{}_enhanced_sharpened_update".format(rgb_dir.parent.parent, idx_lwi, idx_lwi)
        # check wether the folder contains labels
        if os.path.exists(lwi_dir.replace('_LWIR', '_LWIR_l')):
            lwi_dir = lwi_dir.replace('_LWIR', '_LWIR_l')
        # print(rgb_dir, lwi_dir)
        gen = gen_sorted_image_pairs(rgb_dir, lwi_dir, return_labels=False)

        self.paths = [list_path for list_path in gen]

    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index):
        fp_rgb, fp_lwi = self.paths[index]

        rgb_pil = Image.open(fp_rgb).resize((640, 480), Image.BILINEAR)
        lwi_pil = Image.open(fp_lwi)
        lbl_pil = F.to_pil_image(np.uint8(-1*torch.ones(480, 640)))
        return rgb_pil, lwi_pil, lbl_pil, self.cat


class Heliaus(VisionDataset):
    '''
    sudo mount -t cifs -o username=git-smb,password=gitsmb123,uid=uid,gid=gid //samba.anotherbrain.lan/heliaus /mnt/heliaus
    '''
    def __init__(self, root='/mnt/heliaus/', lmdb_dir='dataset/heliaus', category=['Daytime', 'Nighttime'], split='train', use_thermal=True, sequence_dir="", augmentation=False, unlabelled=False, annotation_type='bbox_poly', use_lmdb=True):
        """
        Heliaus dataset
        return ONLY supervised pairs: (RGB and LWIR) + (RGB_label and LWIR_label)
        sequence_dir (str): folder path the rgb images, must end with "*_cropped"
        """
        super().__init__(root)
        self.root = root
        self.lmdb_dir = lmdb_dir
        self.category = category
        self.split = split
        self.use_thermal = use_thermal
        self.sequence_dir = sequence_dir
        self.augmentation = augmentation
        self.unlabelled = unlabelled
        self.num_classes = 7
        #self.annotations = annotations

        # original heliaus classes
        # color are custom and gather several classes 

        if sequence_dir:
            self.dataset = SequenceDataset(rgb_dir=sequence_dir)
            self.testing = True
        else:
            if use_lmdb:
                self.dataset = LmdbDataset(root, lmdb_dir, category, split, unlabelled=unlabelled, annotation_type=annotation_type)
            else:
                csv_fp = "{}/{}.csv".format(self.root, self.split)                        
                self.dataset = CsvDataset(csv_fp, unlabelled=unlabelled, annotation_type=annotation_type)
            self.testing = False


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index, w_crop=640, h_crop=480):

        sample = self.dataset[index]        
        
        if self.unlabelled:
            rgb_pil, lwi_pil, cat =  sample[:3]
            lbl_pil = None
        else:
            rgb_pil, lwi_pil, lbl_pil, cat =  sample
        
        if self.use_thermal:
            lwi_pil = np.uint8(255*(np.array(lwi_pil)/((2**16)-1.0))) # resize to [0-255]
            lwi_pil = Image.fromarray(lwi_pil)
            w, h = lwi_pil.size
        else:
            lwi_pil = None

        
        # temporary
        #lbl_pil = lbl_pil[0]
        tx, ty = transform(rgb_pil, lbl_pil, None, crop_size=(h_crop,w_crop), scale_size=(1.0, 1.0), augmentation=self.augmentation, thermal_image=lwi_pil, testing=self.testing)
        if ty is not None:
            return tx, ty.squeeze(0)
        else:
            return tx


if __name__ == "__main__":

    import sys
    sys.path.append('../')

    from utils.build_data import transform
    name = 'sequence'
    cats = ['Daytime', 'Nighttime']
    split='val'
    rgb_dir="/mnt/heliaus/disk1/20201021_Lindau_Friedrichshafen_Daytime/20201021_155232_l/709_RGB/709_cropped"

    if name == 'csv':
        dataset = CsvDataset('/mnt/heliaus/train.csv')
        loader = DataLoader(dataset, batch_size=15, num_workers=8, collate_fn=lambda x: x)
        for batch in tqdm(loader):
            pass

    if name == 'lmdb':
        dataset = LmdbDataset(root="/mnt/heliaus", lmdb_dir="dataset/heliaus", category=cats, split=split)
        loader = DataLoader(dataset, batch_size=15, num_workers=4, collate_fn=lambda x: x)
        for batch in tqdm(loader):
            pass

    if name == 'pairs':
        rgb_dir="/mnt/heliaus/disk2/20200908_Lindau_Daytime/20200908_134513/707_RGB/707_cropped"
        lwir_dir = "/mnt/heliaus/disk2/20200908_Lindau_Daytime/20200908_134513/773_LWIR/773_enhanced_sharpened_update"
        out = gen_sorted_image_pairs(rgb_dir, lwir_dir, return_labels=True, threshold=0.1e6)
        # print(len(list(out)))
    
    if name == 'sequence':
        dataset = SequenceDataset(rgb_dir="/mnt/heliaus/disk1/20201021_Lindau_Friedrichshafen_Daytime/20201021_155232_l/709_RGB/709_cropped")

    if name == "dataset":
        dataset = Heliaus(root="/mnt/heliaus/", category=cats, split=split, sequence_dir=rgb_dir)

    if name == 'loader':
        dataset = Heliaus(root="/mnt/heliaus/", lmdb_dir='dataset/heliaus',  category=cats, split=split, use_thermal=True)
        loader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=8, pin_memory=False)
        for batch in tqdm(loader):
            pass
    
