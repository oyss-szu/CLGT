import os
import sys
import random
import errno
import time
import torch
import numpy as np
from torch.nn import functional as F
from datetime import timedelta

class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

def setup_system(seed, cudnn_benchmark=True, cudnn_deterministic=True) -> None:
    '''
    Set seeds for for reproducible training
    '''
    # python
    random.seed(seed)
    
    # numpy
    np.random.seed(seed)
    
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
      
        
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def sec_to_min(seconds):
    
    seconds = int(seconds)
    minutes = seconds // 60
    seconds_remaining = seconds % 60
    
    if seconds_remaining < 10:
        seconds_remaining = '0{}'.format(seconds_remaining)
    
    return '{}:{}'.format(minutes, seconds_remaining)

def sec_to_time(seconds):
    return "{:0>8}".format(str(timedelta(seconds=int(seconds))))

def print_time_stats(t_train_start, t_epoch_start, epochs_remaining, steps_per_epoch):
    
    elapsed_time = time.time() - t_train_start
    speed_epoch = time.time() - t_epoch_start 
    speed_batch = speed_epoch / steps_per_epoch
    eta = speed_epoch * epochs_remaining
        
    print("Elapsed {}, {} time/epoch, {:.2f} s/batch, remaining {}".format(
                sec_to_time(elapsed_time), sec_to_time(speed_epoch), speed_batch, sec_to_time(eta)))
    

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, path="checkpoint.pth"):
    state = {
        'epoch': epoch,
        'loss': loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
    }
    # 处理 DataParallel 的模型保存
    if isinstance(model, torch.nn.DataParallel):
        state['model'] = model.module.state_dict()
    else:
        state['model'] = model.state_dict()

    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, scheduler, scaler, path="checkpoint.pth",device="cuda"):
    if not os.path.isfile(path):
        print(f"No checkpoint found at {path}, training from scratch.")
        return 0, None

    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)

    # 加载模型
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])

    # ⚡️关键！搬运 optimizer 的 state 到正确 device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)  # or v.to(model.device) if you have a model.device

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded. Resume from epoch {epoch}")
    return epoch + 1, loss

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两个经纬度坐标之间的距离（单位：米）
    使用Haversine公式
    """
    # 地球半径（米）
    R = 6371000.0

    # 将角度转换为弧度
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)

    # Haversine公式
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = R * c

    return distance


def batch_haversine_distance(coords1, coords2):
    """
    批量计算两组GPS坐标之间的距离矩阵
    coords1, coords2: [B, 2] 形状的张量，每行是 [latitude, longitude]
    返回：[B, B] 形状的距离矩阵，单位为米
    """
    B1 = coords1.shape[0]
    B2 = coords2.shape[0]

    # 展开坐标以便批量计算
    lat1 = coords1[:, 0].unsqueeze(1).expand(B1, B2)  # [B1, B2]
    lon1 = coords1[:, 1].unsqueeze(1).expand(B1, B2)  # [B1, B2]
    lat2 = coords2[:, 0].unsqueeze(0).expand(B1, B2)  # [B1, B2]
    lon2 = coords2[:, 1].unsqueeze(0).expand(B1, B2)  # [B1, B2]

    return haversine_distance(lat1, lon1, lat2, lon2)



def compute_gps_similarity_matrix(query_gps, ref_gps, pos_thresh=50.0, semi_pos_thresh=100.0, sigma=50.0):
    
    dist_matrix = batch_haversine_distance(query_gps, ref_gps)

    sim_matrix = torch.zeros_like(dist_matrix)

    pos_mask = dist_matrix <= pos_thresh
    semi_pos_mask = (dist_matrix > pos_thresh) & (dist_matrix <= semi_pos_thresh)

    sim_matrix[pos_mask] = 1.0
    sim_matrix[semi_pos_mask] = torch.exp(
        -dist_matrix[semi_pos_mask] ** 2 / (2 * sigma ** 2)
    )
    # print("强制对角线为1前：",sim_matrix)
    # input()
    # 强制对角线元素为1.0（如果批次中的查询和参考是一一对应的）
    if query_gps.shape[0] == ref_gps.shape[0]:  # 确保批次大小相等
        diagonal_indices = torch.arange(query_gps.shape[0], device=query_gps.device)
        sim_matrix[diagonal_indices, diagonal_indices] = 7.5
    # print("强制对角线为1后：", sim_matrix)
    # input()
    # 避免除以 0
    sim_matrix = sim_matrix + 1e-8

    sim_matrix = sim_matrix / sim_matrix.sum(dim=1, keepdim=True)  # [B, B]


    return sim_matrix  # [B, B]
