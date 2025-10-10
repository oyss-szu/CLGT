import gc
import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
from torch.nn import functional as F

import tracemalloc


#----------------------------------------------------------------
# 使用融合权重后的trainer函数
#----------------------------------------------------------------

def predict(train_config, model, dataloader):

    # tracemalloc.start()  # 开始监控

    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    img_features_list = []

    ids_list = []
    with torch.no_grad():

        for data in bar:
            l = len(data)
            if l == 2:
                img, ids = data  # eval_loader 返回 (img, label)
            elif l == 3:
                img1, img2, ids = data

            else:
                raise ValueError(f"Unexpected number of return values: {len(data)}")
            ids_list.append(ids)

            with autocast():

                if l==2:
                    img = img.to(train_config.device)
                    img_feature = model(imgq=img)
                elif l==3:
                    img1 = img1.to(train_config.device)
                    img2 = img2.to(train_config.device)

                    img_feature = model(imgq=img1, imgr1=img2)
                else:
                    img1 = img1.to(train_config.device)
                    img2 = img2.to(train_config.device)

                    img_feature = model(imgq=img1, imgr1=img2)

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)

            img_features_list.append(img_feature.to(torch.float32))

        # keep Features on CPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)


    if train_config.verbose:
        bar.close()

    return img_features, ids_list


def train_ours(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):
    # set model train mode
    model.train()
    losses = AverageMeter()

    scale1 = 0.5
    scale2 = 0  # Different versions of PyTorch may affect the feature fusion process; therefore, we conservatively set the weight to 0.
    # wait before starting progress bar
    time.sleep(0.1)
    # Zero gradients for first step0
    optimizer.zero_grad(set_to_none=True)
    step = 1

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    # print("what1")
    # for loop over one epoch

    # for ground_img, sat_img, ground_img_bev,groundA_img, ids in bar:
    for ground_img, sat_img, ground_img_bev, groundA_img, ids in bar:
        if scaler:
            with autocast():

                ground_img = ground_img.to(train_config.device)
                sat_img = sat_img.to(train_config.device)
                ground_img_bev = ground_img_bev.to(train_config.device)
                groundA_img = groundA_img.to(train_config.device)

                sat_img_features,fusiong_features,groundA_features,ground_img_features,ground_bev_img_features= model(imgq=ground_img, imgr1=ground_img_bev, imgr2=sat_img,groundA=groundA_img)
                # sat_img_features,fusiong_features,groundA_features = model(imgq=ground_img, imgr1=ground_img_bev, imgr2=sat_img,groundA=groundA_img)


                loss_cross = loss_function(sat_img_features, fusiong_features, model.module.logit_scale.exp())
                loss_causal = loss_function(fusiong_features, groundA_features,model.module.logit_scale3.exp())

                # loss_cross = loss_function(sat_img_features, ground_img_features, model.module.logit_scale.exp())
                # loss_causal = loss_function(ground_img_features, groundA_features, model.module.logit_scale3.exp())

                loss_bev = loss_function(ground_img_features, ground_bev_img_features, model.module.logit_scale2.exp())

                loss = loss_cross + scale1 * loss_causal + scale2 * loss_bev


                # Forward pass
                losses.update(loss.item())

            scaler.scale(loss).backward()

            # Gradient clipping
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()


            # Zero gradients for next step
            optimizer.zero_grad()

            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"⚠️ NaN detected in gradients of {name}, skipping optimizer step.")

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()


        if train_config.verbose:


            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr": "{:.6f}".format(optimizer.param_groups[0]['lr']),
                       "loss_causal": "{:.6f}".format(loss_causal.item()),
                        "loss_bev": "{:.6f}".format(loss_bev.item()),
                        }
            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg