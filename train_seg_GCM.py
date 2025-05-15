import os
import sys
import yaml
import monai
import torch

from typing import Dict
from objprint import objstr
from easydict import EasyDict
from datetime import datetime
from accelerate import Accelerator
from timm.optim import optim_factory
from dataclasses import dataclass, field
from monai.utils import ensure_tuple_rep

from src.model.HWAUNETR import HWAUNETR
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.loader import get_dataloader_GCM as get_dataloader
from src.utils import Logger, same_seeds, resume_train_state


def train_one_epoch(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
          metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
          post_trans: monai.transforms.Compose, accelerator: Accelerator, epoch: int, step: int):
    # 训练
    model.train()
    for i, image_batch in enumerate(train_loader):
        logits = model(image_batch['image'])
        total_loss = 0
        log = ''
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits, image_batch['label'])
            accelerator.log({'Train/' + name: float(loss)}, step=step)
            total_loss += alpth * loss
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch['label'])
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)
        accelerator.log({
            'Train/Total Loss': float(total_loss),
        }, step=step)
        accelerator.print(
            f'Epoch [{epoch+1}/{config.trainer.num_epochs}][{i + 1}/{len(train_loader)}] Training Loss:{total_loss}',
            flush=True
            )
        step += 1
    scheduler.step(epoch)
    metric = {} 
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update({
            f'Train/mean {metric_name}': float(batch_acc.mean()),
            f'Train/ADC {metric_name}': float(batch_acc[0]),
            f'Train/T2_FS {metric_name}': float(batch_acc[1]),
            f'Train/V {metric_name}': float(batch_acc[2])
        })
    accelerator.log(metric, step=epoch)
    return step

@torch.no_grad()
def val_one_epoch(model: torch.nn.Module,
                  inference: monai.inferers.Inferer, val_loader: torch.utils.data.DataLoader,
                  metrics: Dict[str, monai.metrics.CumulativeIterationMetric], step: int,
                  post_trans: monai.transforms.Compose, accelerator: Accelerator, test: bool=False):
    # val or test
    model.eval()
    dice_acc = 0
    dice_class = []
    hd95_acc = 0
    hd95_class = []
    for i, image_batch in enumerate(val_loader):
        logits = inference(image_batch['image'], model)
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch['label'])
        accelerator.print(
            f'[{i + 1}/{len(val_loader)}] Validation Loading...',
            flush=True)
        
        step += 1
    metric = {}
    if test:
        flag = 'Test'
    else:
        flag = 'Val'
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metrics[metric_name].reset()
        if metric_name == 'dice_metric':
            metric.update({
                f'{flag}/mean {metric_name}': float(batch_acc.mean()),
                f'{flag}/ADC {metric_name}': float(batch_acc[0]),
                f'{flag}/T2_FS {metric_name}': float(batch_acc[1]),
                f'{flag}/V {metric_name}': float(batch_acc[2])
            })
            dice_acc = torch.Tensor([metric[f'{flag}/mean dice_metric']]).to(accelerator.device)
            dice_class = batch_acc
        else:
            metric.update({
                f'{flag}/mean {metric_name}': float(batch_acc.mean()),
                f'{flag}/ADC {metric_name}': float(batch_acc[0]),
                f'{flag}/T2_FS {metric_name}': float(batch_acc[1]),
                f'{flag}/V {metric_name}': float(batch_acc[2])
            })
            hd95_acc = torch.Tensor([metric[f'{flag}/mean hd95_metric']]).to(accelerator.device)
            hd95_class = batch_acc
    accelerator.log(metric, step=epoch)
    return dice_acc, dice_class, hd95_acc, hd95_class, step


if __name__ == '__main__':
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    same_seeds(50)
    logging_dir = os.getcwd() + '/logs/' + config.finetune.GCM.checkpoint + str(datetime.now()).replace(' ','_').replace('-','_').replace(':','_').replace('.','_')
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], logging_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))
    
    accelerator.print('load model...')
    model = HWAUNETR(**config.hwa_unetr)
    
    accelerator.print('load dataset...')
    train_loader, val_loader, test_loader = get_dataloader(config)

    
    inference = monai.inferers.SlidingWindowInferer(roi_size=config.GCM_loader.target_size,     overlap=0.5,sw_device=accelerator.device, device=accelerator.device)
    
    loss_functions = {
        'focal_loss': monai.losses.FocalLoss(to_onehot_y=False),
        'dice_loss': monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True),
    }
    
    metrics = {
        'dice_metric': monai.metrics.DiceMetric(include_background=True,
                                                reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=True),
        'hd95_metric': monai.metrics.HausdorffDistanceMetric(percentile=95, include_background=True,reduction=monai.utils.MetricReduction.MEAN_BATCH,
                                                             get_not_nans=True)
    }
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
    ])

    optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr, betas=(0.9, 0.95))
    
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup,
                                              max_epochs=config.trainer.num_epochs)
    
    
    # start training
    accelerator.print("Start Training!")
    train_step = 0
    best_eopch = -1
    val_step = 0
    best_score = torch.tensor(0)
    best_hd95 = torch.tensor(1000)
    best_test_score = torch.tensor(0)
    best_test_hd95 = torch.tensor(1000)
    starting_epoch = 0
    best_metrics = []
    best_hd95_metrics = []
    best_test_metrics = []
    best_test_hd95_metrics = []
    
    if config.trainer.resume:
        model, optimizer, scheduler, starting_epoch, train_step, best_score, best_metrics, best_hd95, best_hd95_metrics = resume_train_state(model, '{}'.format(
            config.finetune.GCM.checkpoint), optimizer, scheduler, train_loader, accelerator)
        val_step = train_step
    
    model, optimizer, scheduler, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader, test_loader)
    
    best_score = torch.Tensor([best_score]).to(accelerator.device)
    best_hd95 = torch.Tensor([best_hd95]).to(accelerator.device)
    for epoch in range(starting_epoch, config.trainer.num_epochs):
        train_step = train_one_epoch(model, loss_functions, train_loader,
                     optimizer, scheduler, metrics,
                     post_trans, accelerator, epoch, train_step)

        dice_acc, dice_class, hd95_acc, hd95_class, val_step = val_one_epoch(model, inference, val_loader,
                                                                   metrics, val_step,
                                                                   post_trans, accelerator, test=False)
        
        # keep model
        if dice_acc > best_score:
            accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.GCM.checkpoint}/best")
            best_score = dice_acc
            best_metrics = dice_class
            best_hd95 = hd95_acc
            
            # write test acc
            best_test_score, best_test_metrics, best_test_hd95, best_test_hd95_metrics, _ = val_one_epoch(model, inference, test_loader,
                                                                   metrics, -1,
                                                                   post_trans, accelerator,test=True)

        accelerator.print(f'Epoch [{epoch+1}/{config.trainer.num_epochs}] dice acc: {dice_acc} hd95_acc: {hd95_acc} best acc: {best_score}, best hd95: {best_hd95}, best test acc: {best_test_score}, best test hd95: {best_test_hd95}')

        accelerator.print('Cheakpoint...')
        accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.GCM.checkpoint}/checkpoint")
        torch.save({'epoch': epoch, 'best_score': best_score, 'best_metrics': best_metrics, 'best_hd95': best_hd95, 'best_hd95_metrics': best_hd95_metrics},
                    f'{os.getcwd()}/model_store/{config.finetune.GCM.checkpoint}/checkpoint/epoch.pth.tar')
        
    accelerator.print(f"dice score: {best_test_score}")
    accelerator.print(f"dice metrics : {best_test_metrics}")
    accelerator.print(f"hd95 score: {best_test_hd95}")
    accelerator.print(f"hd95 metrics : {best_test_hd95_metrics}")
    

