import os
import time
import math
import shutil
import sys
import torch
import pickle
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
import calendar
from tensorboardX import SummaryWriter
from CLGT.utils import setup_system, Logger,save_checkpoint,load_checkpoint
import matplotlib.pyplot as plt
import cv2
import numpy as np
from CLGT.dataset.cvact import CVACTDatasetTrainOurs, CVACTDatasetEval, CVACTDatasetTest
# from CLGT.dataset.cvusa import CVUSADatasetEval, CVUSADatasetTrain
from CLGT.transforms import get_transforms_train_ours, get_transforms_val
from CLGT.utils import setup_system, Logger
from CLGT.trainer import train_ours
from CLGT.evaluate.cvusa_and_cvact import evaluate, calc_sim
from CLGT.loss import InfoNCE
from CLGT.model import TimmModel_Ours

# from ptflops import get_model_complexity_info



@dataclass
class Configuration:
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    # Override model image size
    img_size: int = 384
    scale = 1    # Training
    mixed_precision: bool = True
    seed = 1
    epochs: int = 40
    batch_size: int = int(128 / scale)  # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,1,2,3,4,5,6,7)  # GPU ids for training
    pretrained = False
    drop: float = 0.4
    drop_connect: float = None
    drop_path: float = 0.4
    drop_block: float = None
    gp: str = None
    checkpoint_path: str = "final"

    # Similarity Sampling
    custom_sampling: bool = True  # use custom sampling instead of rando
    gps_sample: bool = True  # use gps sampling
    sim_sample: bool = True  # use similarity sampling
    neighbour_select: int = int(64 / scale)  # max selection size from pool
    neighbour_range: int = int(128 / scale)  # pool size for selection
    gps_dict_path: str = "../dataset/CVACT/gps_dict.pkl"  # path to pre-computed distances

    # Eval
    batch_size_eval: int = int(128 / 2)
    eval_every_n_epoch: int = 4  # eval every n Epoch
    normalize_features: bool = True

    # Optimizer
    clip_grad = 100.  # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False  # Gradient Checkpointing

    # Loss
    label_smoothing: float = 0.1

    # Learning Rate
    lr: float = 0.001 / (scale*2)  # 1 * 10^-4 for ViT | 1 * 10^-3 for CNN

    scheduler: str = "cosine"  # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.00001    # only for "polynomial"
    # Dataset
    data_folder = "../dataset/CVACT"

    # Augment Images
    prob_rotate: float = 0.75  # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5  # flipping the sat image and ground images simultaneously

    # Savepath for model checkpoints
    model_path: str = "./cvact_ours"

    # Eval before training
    zero_shot: bool = False

    # Checkpoint to start from
    checkpoint_start = None

    # set num_workers to 0 if on Windows
    # num_workers: int = 0 if os.name == 'nt' else 8
    num_workers = 16
    # train on GPU if available
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = True

    # make cudnn deterministic
    cudnn_deterministic: bool = False
    # train_fov: float = 180  # train fov (with random shift)
    # fov: float = 90.0  # eval fov (with random shift)
    # random_fov: bool = False  # if True, plase set train_fov to 360



# -----------------------------------------------------------------------------#
# Train Config                                                                #
# -----------------------------------------------------------------------------#

config = Configuration()

#%%
# generate time stamp
gmt = time.gmtime()
ts = calendar.timegm(gmt)

log_name = 'final'
# save_name = f"{ts}_{config.model}"
path = os.path.join('log_dir',log_name)
print("save_name : ", path, flush=True)
if not os.path.exists(path):
    os.makedirs(path)
else:
    print("Note! Saving path existed !")
writer = SummaryWriter(path)


if __name__ == '__main__':

    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   log_name)

    checkpoint_path = "{}/{}/{}/{}".format(config.model_path,
                                   config.model,
                                   config.checkpoint_path,
                                   log_name)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    # -----------------------------------------------------------------------------#
    # Model                                                                       #
    # -----------------------------------------------------------------------------#

    print("\nModel: {}".format(config.model))

    dim = 1024
    model = TimmModel_Ours(config.model,
                           dim=dim,
                           pretrained=config.pretrained,
                            )
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size

    # fov = config.fov  # eval FoV

    image_size_sat = (img_size, img_size)

    img_size_ground = (img_size, img_size)

    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    # Load pretrained Checkpoint
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)


        # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids) # 会导致register的freqs_cis的维度发生变化，请注意

    # Model to device
    model = model.to(config.device)


    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    # -----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    # -----------------------------------------------------------------------------#

    # Transforms
    # Transforms
    sat_transforms_train, sat_transforms_train_bev, ground_transforms_train,groundA_transforms = get_transforms_train_ours(
        image_size_sat,
        img_size_ground,
        mean=mean,
        std=std
        )


    # Train
    train_dataset = CVACTDatasetTrainOurs(data_folder=config.data_folder,
                                            transforms_query=ground_transforms_train,
                                            transforms_reference=sat_transforms_train,
                                            transforms_reference_bev=sat_transforms_train_bev,
                                            groundA_transforms = groundA_transforms,
                                            prob_flip=config.prob_flip,
                                            prob_rotate=config.prob_rotate,
                                            shuffle_batch_size=config.batch_size
                                            )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True,
                                  prefetch_factor=2
                                  )
    # input()
    # Eval
    sat_transforms_val, sat_transforms_val_bev,ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   )

    # Reference Satellite Images
    reference_dataset_val = CVACTDatasetEval(data_folder=config.data_folder,
                                             split="val",
                                             img_type="reference",
                                             transforms_query=ground_transforms_val,
                                             transforms_reference=sat_transforms_val,
                                             transforms_reference_bev=sat_transforms_val_bev,
                                             )

    reference_dataloader_val = DataLoader(reference_dataset_val,
                                          batch_size=config.batch_size_eval,
                                          num_workers=config.num_workers,
                                          shuffle=False,
                                          pin_memory=True,
                                          prefetch_factor=2
                                          )

    # Query Ground Images Test
    query_dataset_val = CVACTDatasetEval(data_folder=config.data_folder,
                                         split="val",
                                         img_type="query",
                                         transforms_query=ground_transforms_val,
                                         transforms_reference=sat_transforms_val,
                                         transforms_reference_bev=sat_transforms_val_bev,
                                         )

    query_dataloader_val = DataLoader(query_dataset_val,
                                      batch_size=config.batch_size_eval,
                                      num_workers=config.num_workers,
                                      shuffle=False,
                                      pin_memory=True,
                                      prefetch_factor=2
                                      )

    print("Reference Images Val:", len(reference_dataset_val))
    print("Query Images Val:", len(query_dataset_val))


    # -----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    # -----------------------------------------------------------------------------#
    if config.gps_sample:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    # -----------------------------------------------------------------------------#
    # Sim Sample                                                                  #
    # -----------------------------------------------------------------------------#

    if config.sim_sample:
        # Query Ground Images Train for simsampling
        query_dataset_train = CVACTDatasetEval(data_folder=config.data_folder,
                                               split="train",
                                               img_type="query",
                                               transforms_query=ground_transforms_val,
                                               transforms_reference=sat_transforms_val,
                                               transforms_reference_bev=sat_transforms_val_bev,
                                               )

        query_dataloader_train = DataLoader(query_dataset_train,
                                            batch_size=config.batch_size_eval,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=True,
                                            prefetch_factor=2)

        reference_dataset_train = CVACTDatasetEval(data_folder=config.data_folder,
                                                   split="train",
                                                   img_type="reference",
                                                   transforms_query=ground_transforms_val,
                                                   transforms_reference=sat_transforms_val,
                                                   transforms_reference_bev=sat_transforms_val_bev,
                                                   )

        reference_dataloader_train = DataLoader(reference_dataset_train,
                                                batch_size=config.batch_size_eval,
                                                num_workers=config.num_workers,
                                                shuffle=False,
                                                pin_memory=True,
                                                prefetch_factor=2)

        print("\nReference Images Train:", len(reference_dataset_train))
        print("Query Images Train:", len(query_dataset_train))

        # input("评估数据加载完成")
        # -----------------------------------------------------------------------------#
    # Loss                                                                        #
    # -----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn,
                            device=config.device,
                            )

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2. ** 10)
    else:
        scaler = None

    # -----------------------------------------------------------------------------#
    # optimizer                                                                   #
    # -----------------------------------------------------------------------------#
    # lambda_weight = torch.nn.Parameter(torch.tensor(0.75, requires_grad=True).to("cuda"))
    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    # -----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    # -----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end=config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)

    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)

    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps)

    else:
        scheduler = None

    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    # -----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    # -----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30 * "-", "Zero Shot", 30 * "-"))

        r_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_val,
                           query_dataloader=query_dataloader_val,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True,
                           )

        if config.sim_sample:
            r1_train, sim_dict = calc_sim(config=config,
                                          model=model,
                                          reference_dataloader=reference_dataloader_train,
                                          query_dataloader=query_dataloader_train,
                                          ranks=[1, 5, 10],
                                          step_size=1000,
                                          cleanup=True)

    # -----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    # -----------------------------------------------------------------------------#
    if config.custom_sampling:
        train_dataloader.dataset.shuffle(sim_dict,
                                         neighbour_select=config.neighbour_select,
                                         neighbour_range=config.neighbour_range)
    # -----------------------------------------------------------------------------#
    # Train                                                                        #
    # -----------------------------------------------------------------------------#
    start_epoch = 1
    best_score = 0

    for epoch in range(start_epoch, config.epochs + 1):

        print("\n{}[Epoch: {}]{}".format(30 * "-", epoch, 30 * "-"))

        train_loss = train_ours(config,
                                           model,
                                           dataloader=train_dataloader,
                                           loss_function=loss_function,
                                           optimizer=optimizer,
                                           scheduler=scheduler,
                                           scaler=scaler,
                                           )

        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))
        writer.add_scalar('Train_Loss', train_loss, epoch)

        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:

            print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))

            r_test = evaluate(config=config,
                               model=model,
                               reference_dataloader=reference_dataloader_val,
                               query_dataloader=query_dataloader_val,
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)
            r1_test = r_test[0]

            # save_checkpoint(model, optimizer, scheduler, scaler, epoch, train_loss,
            #                 path='{}/weights_e{}_{:.4f}.pth'.format(checkpoint_path, epoch, r1_test))

            writer.add_scalars('validation recall@k', {
                'top 1': r_test[0],
                'top 5': r_test[1],
                'top 10': r_test[2],
                'top 1%': r_test[-1]
            }, epoch)

            # if r1_test > best_score:
            #
            #     best_score = r1_test
            #
            #     if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
            #         torch.save(model.module.state_dict(),
            #                    '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
            #     else:
            #         torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))

            if config.sim_sample:
                r1_train, sim_dict = calc_sim(config=config,
                                              model=model,
                                              reference_dataloader=reference_dataloader_train,
                                              query_dataloader=query_dataloader_train,
                                              ranks=[1, 5, 10],
                                              step_size=1000,
                                              cleanup=True)



        if config.custom_sampling:
            train_dataloader.dataset.shuffle(sim_dict,
                                             neighbour_select=config.neighbour_select,
                                             neighbour_range=config.neighbour_range)

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))

#%%
    # -----------------------------------------------------------------------------#
    # Test                                                                        #
    # -----------------------------------------------------------------------------#
    # Reference Satellite Images
    reference_dataset_test = CVACTDatasetTest(data_folder=config.data_folder,
                                              img_type="reference",
                                              transforms_query=ground_transforms_val,
                                              transforms_reference=sat_transforms_val,
                                              transforms_reference_bev=sat_transforms_val_bev,
                                              )

    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)

    # Query Ground Images Test
    query_dataset_test = CVACTDatasetTest(data_folder=config.data_folder,
                                          img_type="query",
                                          transforms_query=ground_transforms_val,
                                          transforms_reference=sat_transforms_val,
                                          transforms_reference_bev=sat_transforms_val_bev,
                                          )

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))

    print("\n{}[{}]{}".format(30 * "-", "Test", 30 * "-"))

    r1_test = evaluate(config=config,
                       model=model,
                       reference_dataloader=reference_dataloader_test,
                       query_dataloader=query_dataloader_test,
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)
    os._exit(0)

