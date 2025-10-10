import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time
from CLGT.bev_transform.utils import get_BEV_tensor

class CVUSADatasetTrain(Dataset):

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
        
        self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)
        
        self.df = self.df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})
        
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))
        

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
        self.pairs = list(zip(self.df.idx, self.df.sat, self.df.ground))
        
        self.idx2pair = dict()
        train_ids_list = list()
        
        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)
            
        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)
            

    def __getitem__(self, index):
        
        idx, sat, ground = self.idx2pair[self.samples[index]]
        
        # load query -> ground image
        query_img = cv2.imread(f'{self.data_folder}/{ground}')
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # bev transform - especially for cvusa
        Hp, Wp = query_img.shape[0], query_img.shape[1]  # pano shape
        dty = -20
        if dty != 0 or Wp != 2 * Hp:
            ty = (Wp / 2 - Hp) / 2 + dty  # completing pano to the correct proportion
            matrix_K = np.array([[1, 0, 0], [0, 1, ty], [0, 0, 1]])
            new_image = cv2.warpPerspective(query_img, matrix_K, (int(Wp), int(Hp + (Wp / 2 - Hp))))

        # -- bev trans
        bev_img = get_BEV_tensor(new_image, 384, 384, Fov=170, dty=0,
                                   dx=0, dy=0, out=None, device='cpu')

        # load reference -> satellite image
        reference_img = cv2.imread(f'{self.data_folder}/{sat}')
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)


        groundA_img = cv2.imread(f'{self.data_folder}/{ground}')
        groundA_img = cv2.cvtColor(groundA_img, cv2.COLOR_BGR2RGB)
            
        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)
            bev_img = cv2.flip(bev_img, 1)
            groundA_img = cv2.flip(groundA_img, 1)
        
        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
            
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']

        if self.transforms_reference_bev is not None:
            ground_img_bev = self.transforms_reference_bev(image=bev_img)['image']

        if self.groundA_transforms is not None:
            groundA_img = self.groundA_transforms(image=groundA_img)['image']

        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:
        
            r = np.random.choice([1,2,3])
            
            # rotate sat img 90 or 180 or 270
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))
            ground_img_bev = torch.rot90(ground_img_bev, k=r, dims=(1, 2))
            
            # use roll for ground view if rotate sat view
            c, h, w = query_img.shape
            shifts = - w//4 * r
            query_img = torch.roll(query_img, shifts=shifts, dims=2)
            groundA_img = torch.roll(groundA_img, shifts=shifts, dims=2)
                   
            
        label = torch.tensor(idx, dtype=torch.long)  
        
        return query_img, reference_img, ground_img_bev, groundA_img, label
    
    def __len__(self):
        return len(self.samples)
        
        
            
    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            idx_pool = copy.deepcopy(self.train_ids)
        
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
                                
                                # check if idx not already in batch or epoch
                                if idx_near not in idx_batch and idx_near not in idx_epoch and idx_near:
                            
                                    idx_batch.add(idx_near)
                                    current_batch.append(idx_near)
                                    idx_epoch.add(idx_near)
                                    similarity_pool[idx].remove(idx_near)
                                    break_counter = 0
                                    
                    else:
                        # if idx fits not in batch and is not already used in epoch -> back to pool
                        if idx not in idx_batch and idx not in idx_epoch:
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

            
       
class CVUSADatasetEval(Dataset):
    
    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms=None,
                 transforms_reference_bev=None,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms
        self.transforms_reference_bev = transforms_reference_bev
        
        if split == 'train':
            self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)
        else:
            self.df = pd.read_csv(f'{data_folder}/splits/val-19zl.csv', header=None)
        
        self.df = self.df.rename(columns={0:"sat", 1:"ground", 2:"ground_anno"})
        
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
    
        if self.img_type == "reference":
            self.images = self.df.sat.values
            self.label = self.df.idx.values
            
        elif self.img_type == "query":
            self.images = self.df.ground.values
            self.label = self.df.idx.values 
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")
                

    def __getitem__(self, index):
        
        img = cv2.imread(f'{self.data_folder}/{self.images[index]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.img_type == "query":
            # bev transform - especially for cvusa
            Hp, Wp = img.shape[0], img.shape[1]  # pano shape
            dty = -20
            if dty != 0 or Wp != 2 * Hp:
                ty = (Wp / 2 - Hp) / 2 + dty  # completing pano to the correct proportion
                matrix_K = np.array([[1, 0, 0], [0, 1, ty], [0, 0, 1]])
                new_image = cv2.warpPerspective(img, matrix_K, (int(Wp), int(Hp + (Wp / 2 - Hp))))

            # -- bev trans
            bev_img = get_BEV_tensor(new_image, 384, 384, Fov=170, dty=0,
                                       dx=0, dy=0, out=None, device='cpu')

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        if self.transforms_reference_bev is not None:
            bev_img = self.transforms_reference_bev(image=bev_img)['image']
            
        label = torch.tensor(self.label[index], dtype=torch.long)

        if self.img_type == "query":
            return img, bev_img, label
        else:
            return img, label

    def __len__(self):
        return len(self.images)


class CVUSADatasetEval_C(Dataset):

    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms=None,
                 transforms_reference_bev=None,
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms
        self.transforms_reference_bev = transforms_reference_bev

        if split == 'train':
            self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)
        else:
            self.df = pd.read_csv(f'{data_folder}/splits/val-19zl.csv', header=None)

        self.df = self.df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})

        self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))

        if self.img_type == "reference":
            self.images = self.df.sat.values
            self.label = self.df.idx.values

        elif self.img_type == "query":
            self.images = self.df.ground.values
            self.label = self.df.idx.values
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")

    def __getitem__(self, index):
        # 加载卫星图像（路径不变）
        img = cv2.imread(f'{self.data_folder}/{self.images[index]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.img_type == "query":
            # 构造街景图像路径
            pano_name = self.images[index].split('/')[-1]  # 只取文件名
            img = cv2.imread(f'/raid/zhuyingying/zqw/dataset/CVUSA-C-ALL/{pano_name}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # bev transform - especially for cvusa
            Hp, Wp = img.shape[0], img.shape[1]  # pano shape
            dty = -20
            if dty != 0 or Wp != 2 * Hp:
                ty = (Wp / 2 - Hp) / 2 + dty  # completing pano to the correct proportion
                matrix_K = np.array([[1, 0, 0], [0, 1, ty], [0, 0, 1]])
                new_image = cv2.warpPerspective(img, matrix_K, (int(Wp), int(Hp + (Wp / 2 - Hp))))

            # -- bev trans
            bev_img = get_BEV_tensor(new_image, 384, 384, Fov=170, dty=0,
                                       dx=0, dy=0, out=None, device='cpu')

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        if self.transforms_reference_bev is not None and self.img_type == "query":
            bev_img = self.transforms_reference_bev(image=bev_img)['image']

        label = torch.tensor(self.label[index], dtype=torch.long)

        if self.img_type == "query":
            return img, bev_img, label
        else:
            return img, label

    def __len__(self):
        return len(self.images)


class CVUSADatasetEval_C_ALL(Dataset):

    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms=None,
                 transforms_reference_bev=None,
                 corruption_type=None,
                 severity=None
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms
        self.transforms_reference_bev = transforms_reference_bev

        self.corruption_type = corruption_type
        self.severity = severity

        if split == 'train':
            self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)
        else:
            self.df = pd.read_csv(f'{data_folder}/splits/val-19zl.csv', header=None)

        self.df = self.df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})

        self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))

        if self.img_type == "reference":
            self.images = self.df.sat.values
            self.label = self.df.idx.values

        elif self.img_type == "query":
            self.images = self.df.ground.values
            self.label = self.df.idx.values
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")

    def __getitem__(self, index):
        # 加载卫星图像（路径不变）
        img = cv2.imread(f'{self.data_folder}/{self.images[index]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.img_type == "query":
            # 构造街景图像路径
            pano_name = self.images[index].split('/')[-1]  # 只取文件名
            img = cv2.imread(f'/raid/zhuyingying/zqw/dataset/CVUSA-C/severity-{self.severity}/{self.corruption_type}/{pano_name}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # bev transform - especially for cvusa
            Hp, Wp = img.shape[0], img.shape[1]  # pano shape
            dty = -20
            if dty != 0 or Wp != 2 * Hp:
                ty = (Wp / 2 - Hp) / 2 + dty  # completing pano to the correct proportion
                matrix_K = np.array([[1, 0, 0], [0, 1, ty], [0, 0, 1]])
                new_image = cv2.warpPerspective(img, matrix_K, (int(Wp), int(Hp + (Wp / 2 - Hp))))

            # -- bev trans
            bev_img = get_BEV_tensor(new_image, 384, 384, Fov=170, dty=0,
                                       dx=0, dy=0, out=None, device='cpu')

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        if self.transforms_reference_bev is not None and self.img_type == "query":
            bev_img = self.transforms_reference_bev(image=bev_img)['image']

        label = torch.tensor(self.label[index], dtype=torch.long)

        if self.img_type == "query":
            return img, bev_img, label
        else:
            return img, label

    def __len__(self):
        return len(self.images)




            





