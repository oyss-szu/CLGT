import cv2
import numpy as np
from torch.utils.data import Dataset
import random
import copy
import torch
from tqdm import tqdm
import time
import scipy.io as sio
import os
from glob import glob
import matplotlib.pyplot as plt
from CLGT.bev_transform.utils import get_BEV_tensor, get_BEV_projection

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

class CVACTDatasetTrainOurs(Dataset):

    def __init__(self,
                 data_folder,
                 transforms_query=None,
                 transforms_reference=None,
                 transforms_reference_bev=None,
                 groundA_transforms=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        self.transforms_query = transforms_query
        self.transforms_reference = transforms_reference  # satellite
        self.transforms_reference_bev = transforms_reference_bev
        self.groundA_transforms = groundA_transforms

        anuData = sio.loadmat(f'{data_folder}/ACT_data.mat')

        ids = anuData['panoIds']

        train_ids = ids[anuData['trainSet'][0][0][1] - 1]

        train_ids_list = []
        train_idsnum_list = []
        self.idx2numidx = dict()
        self.numidx2idx = dict()
        self.idx_ignor = set()
        i = 0

        for idx in train_ids.squeeze():

            idx = str(idx)

            grd_path = f'ANU_data_small/streetview/{idx}_grdView.jpg'
            sat_path = f'ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
            if not os.path.exists(f'{self.data_folder}/{grd_path}') or not os.path.exists(
                    f'{self.data_folder}/{sat_path}'):
                self.idx_ignor.add(idx)
            else:
                self.idx2numidx[idx] = i
                self.numidx2idx[i] = idx
                train_ids_list.append(idx)
                train_idsnum_list.append(i)
                i += 1

        print("IDs not found in train images:", self.idx_ignor)

        self.train_ids = train_ids_list
        self.train_idsnum = train_idsnum_list
        self.samples = copy.deepcopy(self.train_idsnum)

    def __getitem__(self, index):
        # print("what-------------------------1----------------------")
        idnum = self.samples[index]

        idx = self.numidx2idx[idnum]

        # load query -> ground image
        ground_img = cv2.imread(f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg')
        ground_img = cv2.cvtColor(ground_img, cv2.COLOR_BGR2RGB)

        groundA_img = cv2.imread(f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg')
        groundA_img = cv2.cvtColor(groundA_img, cv2.COLOR_BGR2RGB)

        bev_img = cv2.imread(f'{self.data_folder}/ANU_data_small/BEV/{idx}_grdView.jpg')
        bev_img = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)

        # load reference -> satellite image
        sat_img = cv2.imread(f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg')
        sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)


        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            ground_img = cv2.flip(ground_img, 1)
            sat_img = cv2.flip(sat_img, 1)
            bev_img = cv2.flip(bev_img, 1)
            groundA_img = cv2.flip(groundA_img, 1)
            # ground_img_bev = cv2.flip(sat_img_bev, 1)

            # image transforms
        if self.transforms_query is not None:
            ground_img = self.transforms_query(image=ground_img)['image']

        if self.transforms_reference is not None:
            sat_img = self.transforms_reference(image=sat_img)['image']

        if self.transforms_reference_bev is not None:
            ground_img_bev = self.transforms_reference_bev(image=bev_img)['image']


        if self.groundA_transforms is not None:
            groundA_img = self.groundA_transforms(image=groundA_img)['image']
        # print("what-------------------------2----------------------")

        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])

            # rotate sat img 90 or 180 or 270
            sat_img = torch.rot90(sat_img, k=r, dims=(1, 2))
            ground_img_bev = torch.rot90(ground_img_bev, k=r, dims=(1, 2))

            # use roll for ground view if rotate sat view
            c, h, w = ground_img.shape
            shifts = - w // 4 * r
            ground_img = torch.roll(ground_img, shifts=shifts, dims=2)
            groundA_img = torch.roll(groundA_img, shifts=shifts, dims=2)


        label = torch.tensor(idnum, dtype=torch.long)
        return ground_img, sat_img, ground_img_bev,groundA_img,label


    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):

        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        idx_pool = copy.deepcopy(self.train_idsnum)

        neighbour_split = neighbour_select // 2

        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)

        # Shuffle pairs order
        random.shuffle(idx_pool)

        # Lookup if already used in epoch
        idx_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(idx_pool) > 0:
                idx = idx_pool.pop(0)

                if idx not in idx_batch and idx not in idx_epoch and len(current_batch) < self.shuffle_batch_size:

                    idx_batch.add(idx)
                    current_batch.append(idx)
                    idx_epoch.add(idx)
                    break_counter = 0

                    # check if near sat views within margine
                    if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:

                        near_similarity = similarity_pool[idx][:neighbour_range]

                        near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])

                        far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])

                        random.shuffle(far_neighbours)

                        far_neighbours = far_neighbours[:neighbour_split]

                        near_similarity_select = near_neighbours + far_neighbours

                        for idx_near in near_similarity_select:

                            # check for space in batch
                            if len(current_batch) >= self.shuffle_batch_size:
                                break

                            # check if idx not already in batch or epoch and not in ignor list (missing image)
                            if idx_near not in idx_batch and idx_near not in idx_epoch:
                                idx_batch.add(idx_near)
                                current_batch.append(idx_near)
                                idx_epoch.add(idx_near)
                                similarity_pool[idx].remove(idx_near)
                                break_counter = 0

                else:
                    # if idx fits not in batch and is not already used in epoch -> back to pool
                    if idx not in idx_epoch:
                        idx_pool.append(idx)

                    break_counter += 1

                if break_counter >= 1024:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches
        print("idx_pool:", len(idx_pool))
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.train_ids), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.train_ids) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0], self.samples[-1]))


class CVACTDatasetEval(Dataset):

    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms_query=None,
                 transforms_reference=None,
                 transforms_reference_bev=None,
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms_query = transforms_query
        self.transforms_reference = transforms_reference  # satellite
        self.transforms_reference_bev = transforms_reference_bev

        anuData = sio.loadmat(f'{data_folder}/ACT_data.mat')

        ids = anuData['panoIds']
        # vision_list = []

        if split != "train" and split != "val":
            raise ValueError("Invalid 'split' parameter. 'split' must be 'train' or 'val'")

        if img_type != 'query' and img_type != 'reference' and img_type != 'polar_reference':
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")

        ids = ids[anuData[f'{split}Set'][0][0][1] - 1]

        ids_list = []

        self.idx2label = dict()
        self.idx_ignor = set()

        i = 0

        for idx in ids.squeeze():

            idx = str(idx)
            grd_path = f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg'
            sat_path = f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
            if not os.path.exists(grd_path) or not os.path.exists(sat_path):
                self.idx_ignor.add(idx)
            else:
                self.idx2label[idx] = i
                ids_list.append(idx)
                i += 1
        self.samples = ids_list

    def __getitem__(self, index):

        idx = self.samples[index]
        bev_path = None
        if self.img_type == "reference":
            path = f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
        elif self.img_type == "query":
            path = f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg'
            bev_path = f'{self.data_folder}/ANU_data_small/BEV/{idx}_grdView.jpg'
        elif self.img_type == "polar_reference":
            path = f'{self.data_folder}/ANU_data_small/satview_polarmap/{idx}_satView_polish.jpg'

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if bev_path is not None:
            bev_img = cv2.imread(bev_path)
            bev_img = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)

        # image transforms
        if bev_path is None:
            img = self.transforms_reference(image=img)['image']
        else:
            img = self.transforms_query(image=img)['image']
            bev_img = self.transforms_reference_bev(image=bev_img)['image']

        label = torch.tensor(self.idx2label[idx], dtype=torch.long)

        if bev_path is None:
            return img, label
        else:
            return img, bev_img, label

    def __len__(self):
        return len(self.samples)


class CVACTDatasetTest(Dataset):

    def __init__(self,
                 data_folder,
                 img_type,
                 transforms_query=None,
                 transforms_reference=None,
                 transforms_reference_bev=None,
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.img_type = img_type
        self.transforms_query = transforms_query
        self.transforms_reference = transforms_reference  # satellite
        self.transforms_reference_bev = transforms_reference_bev

        files_sat = glob(f'{self.data_folder}/ANU_data_test/satview_polish/*_satView_polish.jpg')
        files_ground = glob(f'{self.data_folder}/ANU_data_test/streetview/*_grdView.jpg')

        sat_ids = []
        for path in files_sat:
            idx = path.split("/")[-1][:-19]
            sat_ids.append(idx)

        ground_ids = []
        for path in files_ground:
            idx = path.split("/")[-1][:-12]
            ground_ids.append(idx)

            # only use intersection of sat and ground ids
        test_ids = set(sat_ids).intersection(set(ground_ids))

        self.test_ids = list(test_ids)
        self.test_ids.sort()

        self.idx2num_idx = dict()

        for i, idx in enumerate(self.test_ids):
            self.idx2num_idx[idx] = i

    def __getitem__(self, index):

        idx = self.test_ids[index]
        bev_path = None

        if self.img_type == "reference":
            path = f'{self.data_folder}/ANU_data_test/satview_polish/{idx}_satView_polish.jpg'
        else:
            path = f'{self.data_folder}/ANU_data_test/streetview/{idx}_grdView.jpg'
            bev_path = f'{self.data_folder}/ANU_data_test/BEV/{idx}_grdView.jpg'

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if bev_path is not None:
            bev_img = cv2.imread(bev_path)
            bev_img = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
        # image transforms
        if bev_path is None:
            img = self.transforms_reference(image=img)['image']
        else:
            img = self.transforms_query(image=img)['image']
            bev_img = self.transforms_reference_bev(image=bev_img)['image']

        label = torch.tensor(self.idx2num_idx[idx], dtype=torch.long)

        if bev_path is None:
            return img, label
        else:
            return img, bev_img, label

    def __len__(self):
        return len(self.test_ids)

class CVACTDatasetEval_C(Dataset):

    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms_query=None,
                 transforms_reference=None,
                 transforms_reference_bev=None,
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms_query = transforms_query
        self.transforms_reference = transforms_reference  # satellite
        self.transforms_reference_bev = transforms_reference_bev

        anuData = sio.loadmat(f'{data_folder}/ACT_data.mat')

        ids = anuData['panoIds']

        if split != "train" and split != "val":
            raise ValueError("Invalid 'split' parameter. 'split' must be 'train' or 'val'")

        if img_type != 'query' and img_type != 'reference' and img_type != 'polar_reference':
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")

        ids = ids[anuData[f'{split}Set'][0][0][1] - 1]

        ids_list = []

        self.idx2label = dict()
        self.idx_ignor = set()

        i = 0

        for idx in ids.squeeze():

            idx = str(idx)

            grd_path = f'../dataset/ECCV-ZhangQingQang/CVACT_val-C-ALL/{idx}_grdView.jpg'
            sat_path = f'{self.data_folder}/ANU_DATA_SMALL/ANU_data_small/satview_polish/{idx}_satView_polish.jpg'

            if not os.path.exists(grd_path) or not os.path.exists(sat_path):
                self.idx_ignor.add(idx)
            else:
                self.idx2label[idx] = i
                ids_list.append(idx)
                i += 1

        # print(f"IDs not found in {split} images:", self.idx_ignor)

        self.samples = ids_list

    def __getitem__(self, index):

        idx = self.samples[index]
        bev_path = None
        if self.img_type == "reference":
            # path = f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
            path = f'{self.data_folder}/ANU_DATA_SMALL/ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
        elif self.img_type == "query":
            path = f'../dataset/ECCV-ZhangQingQang/CVACT_val-C-ALL/{idx}_grdView.jpg'
            bev_path = f'{self.data_folder}/CVACT_val-C-ALL_bev/{idx}_grdView.jpg'
        elif self.img_type == "polar_reference":
            # path = f'{self.data_folder}/ANU_data_small/satview_polarmap/{idx}_satView_polish.jpg'
            path = f'{self.data_folder}/ANU_DATA_SMALL/ANU_data_small/satview_polarmap/{idx}_satView_polish.jpg'

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if bev_path is not None:
            bev_img = cv2.imread(bev_path)
            bev_img = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)

        # image transforms
        if bev_path is None:
            img = self.transforms_reference(image=img)['image']
        else:
            img = self.transforms_query(image=img)['image']
            bev_img = self.transforms_reference_bev(image=bev_img)['image']

        label = torch.tensor(self.idx2label[idx], dtype=torch.long)

        if bev_path is None:
            return img, label
        else:
            return img, bev_img, label,

    def __len__(self):
        return len(self.samples)


class CVACT_test_DatasetEval_C(Dataset):

    def __init__(self,
                 data_folder,
                 img_type,
                 transforms_query=None,
                 transforms_reference=None,
                 transforms_reference_bev=None,
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.img_type = img_type
        self.transforms_query = transforms_query
        self.transforms_reference = transforms_reference  # satellite
        self.transforms_reference_bev = transforms_reference_bev

        files_sat = glob(f'{self.data_folder}/ANU_DATA_TEST/ANU_data_test/satview_polish/*_satView_polish.jpg')
        files_ground = glob(f'../dataset/ECCV-ZhangQingQang/CVACT_test-C-ALL/*_grdView.jpg')

        sat_ids = []
        for path in files_sat:
            idx = path.split("/")[-1][:-19]
            sat_ids.append(idx)

        ground_ids = []
        for path in files_ground:
            idx = path.split("/")[-1][:-12]
            ground_ids.append(idx)

            # only use intersection of sat and ground ids
        test_ids = set(sat_ids).intersection(set(ground_ids))

        self.test_ids = list(test_ids)
        self.test_ids.sort()

        self.idx2num_idx = dict()

        for i, idx in enumerate(self.test_ids):
            self.idx2num_idx[idx] = i

    def __getitem__(self, index):

        idx = self.test_ids[index]
        bev_path = None

        if self.img_type == "reference":
            path = f'{self.data_folder}/ANU_DATA_TEST/ANU_data_test/satview_polish/{idx}_satView_polish.jpg'
        else:
            path = f'../dataset/ECCV-ZhangQingQang/CVACT_test-C-ALL/{idx}_grdView.jpg'
            bev_path = f'{self.data_folder}/CVACT_test-C-ALL_bev/{idx}_grdView.jpg'

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if bev_path is not None:
            bev_img = cv2.imread(bev_path)
            bev_img = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
        # image transforms
        if bev_path is None:
            img = self.transforms_reference(image=img)['image']
        else:
            img = self.transforms_query(image=img)['image']
            bev_img = self.transforms_reference_bev(image=bev_img)['image']

        label = torch.tensor(self.idx2num_idx[idx], dtype=torch.long)

        if bev_path is None:
            return img, label
        else:
            return img, bev_img, label

    def __len__(self):
        return len(self.test_ids)



class CVACTDatasetEval_C_ALL(Dataset):

    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms_query=None,
                 transforms_reference=None,
                 transforms_reference_bev=None,
                 corruption_type=None,
                 severity=None
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms_query = transforms_query
        self.transforms_reference = transforms_reference  # satellite
        self.transforms_reference_bev = transforms_reference_bev

        self.corruption_type = corruption_type
        self.severity = severity

        anuData = sio.loadmat(f'{data_folder}/ACT_data.mat')

        ids = anuData['panoIds']

        if split != "train" and split != "val":
            raise ValueError("Invalid 'split' parameter. 'split' must be 'train' or 'val'")

        if img_type != 'query' and img_type != 'reference' and img_type != 'polar_reference':
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")

        ids = ids[anuData[f'{split}Set'][0][0][1] - 1]

        ids_list = []

        self.idx2label = dict()
        self.idx_ignor = set()

        i = 0

        for idx in ids.squeeze():

            idx = str(idx)

            grd_path = f'../dataset/ECCV-ZhangQingQang/CVACT_val-C-ALL/{idx}_grdView.jpg'
            sat_path = f'{self.data_folder}/ANU_DATA_SMALL/ANU_data_small/satview_polish/{idx}_satView_polish.jpg'

            if not os.path.exists(grd_path) or not os.path.exists(sat_path):
                self.idx_ignor.add(idx)
            else:
                self.idx2label[idx] = i
                ids_list.append(idx)
                i += 1

        # print(f"IDs not found in {split} images:", self.idx_ignor)

        self.samples = ids_list

    def __getitem__(self, index):

        idx = self.samples[index]
        bev_path = None
        if self.img_type == "reference":
            path = f'{self.data_folder}/ANU_DATA_SMALL/ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
        elif self.img_type == "query":
            path = f'../dataset/ECCV-ZhangQingQang/CVACT_val-C/severity-{self.severity}/{self.corruption_type}/{idx}_grdView.jpg'
            bev_path = path
        elif self.img_type == "polar_reference":
            path = f'{self.data_folder}/ANU_DATA_SMALL/ANU_data_small/satview_polarmap/{idx}_satView_polish.jpg'

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if bev_path is not None:

            bev_img = get_BEV_tensor(img, 384, 384, Fov=85 * 2, dty=0,
                                       dx=0, dy=0, out=None, device='cpu')

        # image transforms
        if bev_path is None:

            img = self.transforms_reference(image=img)['image']
        else:
            img = self.transforms_query(image=img)['image']
            bev_img = self.transforms_reference_bev(image=bev_img)['image']

        label = torch.tensor(self.idx2label[idx], dtype=torch.long)

        if bev_path is None:
            return img, label
        else:
            return img, bev_img, label

    def __len__(self):
        return len(self.samples)
