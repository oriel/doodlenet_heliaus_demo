import os
from shutil import copyfile

path_cityscapes_train = "/home/ofrigo/develop/reco/dataset/cityscapes/images/train"
path_cityscapes_val = "/home/ofrigo/develop/reco/dataset/cityscapes/images/val"
path_cityscapes_train_labels = "/home/ofrigo/develop/reco/dataset/cityscapes/labels/train"
path_cityscapes_val_labels = "/home/ofrigo/develop/reco/dataset/cityscapes/labels/val"

path_cityscapes = "/home/ofrigo/develop/reco/dataset/cityscapes"
path_cityscapes_images = "/home/ofrigo/develop/reco/dataset/cityscapes/images"
path_cityscapes_labels = "/home/ofrigo/develop/reco/dataset/cityscapes/labels"

dst_path_combined = "/home/ofrigo/datasets/cityscapes_rtfnet"
dst_train_idx_file = "/home/ofrigo/datasets/cityscapes_rtfnet/train.txt"
dst_val_idx_file = "/home/ofrigo/datasets/cityscapes_rtfnet/val.txt"

# get training, val and test images names from cityscapes
cityscapes_train_idx = os.listdir(path_cityscapes_train)
cityscapes_val_idx = os.listdir(path_cityscapes_val)

# get training, val label names from cityscapes
cityscapes_train_labels_idx = os.listdir(path_cityscapes_train_labels)
cityscapes_val_labels_idx = os.listdir(path_cityscapes_val_labels)

def copy_combine(src_dataset_idxs, src_path, dest_path, partition_idx_file, suffix):
    for fname in src_dataset_idxs:
        # 1) copy image files
        src = os.path.join(src_path, 'images', suffix, fname)
        idx_to_write = fname.replace(".png", "") + suffix
        dest = os.path.join(dest_path, "separated_images", f"{idx_to_write}_rgb.png")
        copyfile(src, dest)
        print(f"copyfile({src},{dest})")

        # 2) copy label files
        src = os.path.join(src_path, 'labels', suffix, fname)
        idx_to_write = fname.replace(".png", "") + suffix
        dest = os.path.join(dest_path, "labels", f"{idx_to_write}.png")
        copyfile(src, dest)
        print(f"copyfile({src},{dest})")

        # 3) rewrite train/val partition on text file
        with open(partition_idx_file, "a") as a_file:
              a_file.write(idx_to_write)
              a_file.write("\n")
        
copy_combine(cityscapes_train_idx, path_cityscapes, dst_path_combined, dst_train_idx_file, "train")
copy_combine(cityscapes_val_idx, path_cityscapes, dst_path_combined, dst_val_idx_file, "val")
