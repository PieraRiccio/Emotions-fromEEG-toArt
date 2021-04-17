import argparse
import math
import random
import os
import itertools

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils, models
from tqdm import tqdm

try:
    import wandb

except ImportError:
    wandb = None

from stylegan2.model import Generator, Discriminator
from stylegan2.dataset import MultiResolutionDataset
from stylegan2.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from stylegan2.non_leaking import augment, AdaptiveAugment

from stylegan2.loss import StyleLoss
from rgnn.RGNN_utils import SymSimGCNNet


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]

def get_random_labels(batch, n_labels, device):
    labels = (torch.rand(batch)*n_labels).long()
    labels = torch.nn.functional.one_hot(labels, num_classes=n_labels).float().to(device)
    return labels

def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

###############
#     args    #
###############

def run_stylegan(args,eeg_dataloaders,wikiArt_dataloaders):
  device = "cuda"
  args.iter = 800000
  args.batch = 8
  args.n_sample = 64
  args.size = 128
  args.r1 = 10
  args.path_regularize = 2
  args.path_batch_shrink = 2
  args.d_reg_every = 16
  args.g_reg_every = 4
  args.mixing = 0.9
  #args.ckpt = None
  #args.pretrainedRGNN = "/content/Emotions-fromEEG-toArt/checkpoints/rgnn.pth"
  #args.pretrainedALEXNET = "/content/Emotions-fromEEG-toArt/checkpoints/alexnet.pth"
  args.lr = 0.002
  args.channel_multiplier = 2
  args.wandb = False
  args.local_rank = 0
  args.augment = True
  args.augment_p = 0
  args.ada_target = 0.6
  args.ada_length = 500 * 1000
  args.ada_every = 256
  #args.path = None
  #args.patient = 15
  #args.output_folder = "./output"

  n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
  args.distributed = n_gpu > 1

  if args.distributed:
      torch.cuda.set_device(args.local_rank)
      torch.distributed.init_process_group(backend="nccl", init_method="env://")
      synchronize()

  args.latent = 512
  args.n_mlp = 8
  args.start_iter = 0

  ###############
  #   networks  #
  ###############

  generator = Generator(
      args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, n_labels=50
  ).to(device)
  discriminator = Discriminator(
      args.size, channel_multiplier=args.channel_multiplier
  ).to(device)
  g_ema = Generator(
      args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, n_labels=50
  ).to(device)
  g_ema.eval()
  accumulate(g_ema, generator, 0)

  vgg = models.vgg19(pretrained=True).features.eval()

  aux_classifier = torch.load(args.pretrainedALEXNET).to(device)
  aux_classifier.eval()

  RGNN = torch.load(args.pretrainedRGNN).to(device)
  RGNN.eval()

  ###############
  #  optimizers #
  ###############

  g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
  d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

  g_optim = optim.Adam(
      generator.parameters(),
      lr=args.lr * g_reg_ratio,
      betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
  )
  d_optim = optim.Adam(
      discriminator.parameters(),
      lr=args.lr * d_reg_ratio,
      betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
  )

  RGNN_optim = optim.Adam(RGNN.parameters(), lr=2e-4, weight_decay=1e-5)

  ###############
  #  checkpoint #
  ###############

  if args.ckpt is not None:
      print("load model:", args.ckpt)

      ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

      try:
          ckpt_name = os.path.basename(args.ckpt)
          args.start_iter = int(os.path.splitext(ckpt_name)[0])

      except ValueError:
          pass

      generator.load_state_dict(ckpt["g"])
      discriminator.load_state_dict(ckpt["d"])
      g_ema.load_state_dict(ckpt["g_ema"])
      RGNN.load_state_dict(ckpt["RGNN"])

      g_optim.load_state_dict(ckpt["g_optim"])
      d_optim.load_state_dict(ckpt["d_optim"])
      RGNN_optim.load_state_dict(ckpt["RGNN_optim"])

  if args.distributed:
      generator = nn.parallel.DistributedDataParallel(
          generator,
          device_ids=[args.local_rank],
          output_device=args.local_rank,
          broadcast_buffers=False,
      )

      discriminator = nn.parallel.DistributedDataParallel(
          discriminator,
          device_ids=[args.local_rank],
          output_device=args.local_rank,
          broadcast_buffers=False,
      )

  ###############
  #  dataloader #
  ###############

  def sample_data(list_dataloaders):
    while True:
      for batches in zip(*list_dataloaders):
        yield batches[0],batches[4]
        yield batches[1],batches[5]
        yield batches[2],batches[6]
        yield batches[3],batches[7]

  patient_eeg_dataloaders=eeg_dataloaders[args.patient]
  max_batches = max(
      max([len(patient_eeg_dataloaders[k]["train"]) for k in patient_eeg_dataloaders.keys()]),
      max([len(v) for v in wikiArt_dataloaders.values()])
  )
  list_dataloaders_eeg = [itertools.cycle(v["train"]) if len(v["train"])!=max_batches else v["train"] for k,v in patient_eeg_dataloaders.items()]
  list_dataloaders_wikiArt = [itertools.cycle(v) if len(v)!=max_batches else v for v in wikiArt_dataloaders.values()]
  loader = list_dataloaders_eeg+list_dataloaders_wikiArt
  loader = sample_data(loader)

  ###############
  #     init    #
  ###############

  if get_rank() == 0 and wandb is not None and args.wandb:
      wandb.init(project="stylegan 2")

  pbar = range(args.iter)

  if get_rank() == 0:
      pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01, position=0, leave=True)

  mean_path_length = 0
  d_loss_val = 0
  r1_loss = torch.tensor(0.0, device=device)
  g_loss_val = 0
  path_loss = torch.tensor(0.0, device=device)
  path_lengths = torch.tensor(0.0, device=device)
  mean_path_length_avg = 0
  loss_dict = {}

  if args.distributed:
      g_module = generator.module
      d_module = discriminator.module

  else:
      g_module = generator
      d_module = discriminator

  accum = 0.5 ** (32 / (10 * 1000))
  ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
  r_t_stat = 0

  if args.augment and args.augment_p == 0:
      ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 256, device)

  sample_z = torch.randn(args.n_sample, args.latent, device=device)
  sample_labels = torch.tensor([0,1,2,3]).repeat(args.n_sample//4)
  sample_labels = torch.nn.functional.one_hot(sample_labels, num_classes=4).float().to(device)
  sample_eeg=[]
  for i,(neutral,sadness,fear,happiness) in enumerate(zip(
      eeg_dataloaders[args.patient]["neutral"]["train"],
      eeg_dataloaders[args.patient]["sadness"]["train"],
      eeg_dataloaders[args.patient]["fear"]["train"],
      eeg_dataloaders[args.patient]["happiness"]["train"]
  )):
    if(i<10): continue
    elif(i>11): break
    sample_eeg.append(neutral.to(device))
    sample_eeg.append(sadness.to(device))
    sample_eeg.append(fear.to(device))
    sample_eeg.append(happiness.to(device))

  sample_neutral=[eeg.to(device) for eeg in eeg_dataloaders[args.patient]["neutral"]["train"]][:8]
  sample_sadness=[eeg.to(device) for eeg in eeg_dataloaders[args.patient]["sadness"]["train"]][:8]
  sample_fear=[eeg.to(device) for eeg in eeg_dataloaders[args.patient]["fear"]["train"]][:8]
  sample_happiness=[eeg.to(device) for eeg in eeg_dataloaders[args.patient]["happiness"]["train"]][:8]

  lambda_aux = 1e-1
  lambda_style, crit_style = 10, StyleLoss(vgg_module = vgg)
  lambda_eeg_reg = 0.001

  if not os.path.isdir(args.output_folder):
    os.mkdir(args.output_folder)
    os.mkdir(os.path.join(args.output_folder,"images"))
    os.mkdir(os.path.join(args.output_folder,"models"))

  ###############
  #   routine   #
  ###############

  for idx in pbar:
      i = idx + args.start_iter

      if i > args.iter:
          print("Done!")

          break

      eeg, style = next(loader)
      eeg = eeg.to(device)
      real_img, labels = style["images"], style["labels"]
      real_img = real_img.to(device)
      labels = torch.nn.functional.one_hot(labels, num_classes=4).float().to(device)

      ###############
      #   train D   #
      ###############

      requires_grad(generator, False)
      requires_grad(discriminator, True)
      requires_grad(RGNN, False)

      eeg_latent, _, _ = RGNN(eeg)

      noise = mixing_noise(args.batch, args.latent, args.mixing, device)
      fake_img, _ = generator(noise, eeg_latent)

      if args.augment:
          real_img_aug, _ = augment(real_img, ada_aug_p)
          fake_img, _ = augment(fake_img, ada_aug_p)

      else:
          real_img_aug = real_img

      fake_pred = discriminator(fake_img, labels)
      real_pred = discriminator(real_img_aug, labels)
      d_loss = d_logistic_loss(real_pred, fake_pred)

      loss_dict["d"] = d_loss
      loss_dict["real_score"] = real_pred.mean()
      loss_dict["fake_score"] = fake_pred.mean()

      discriminator.zero_grad()
      d_loss.backward()
      d_optim.step()

      if args.augment and args.augment_p == 0:
          ada_aug_p = ada_augment.tune(real_pred)
          r_t_stat = ada_augment.r_t_stat

      d_regularize = i % args.d_reg_every == 0

      if d_regularize:
          real_img.requires_grad = True
          real_pred = discriminator(real_img, labels)
          r1_loss = d_r1_loss(real_pred, real_img)

          discriminator.zero_grad()
          (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

          d_optim.step()

      loss_dict["r1"] = r1_loss

      ###############
      #   train G   #
      ###############

      requires_grad(generator, True)
      requires_grad(discriminator, False)
      requires_grad(RGNN, False)

      eeg_latent, _, _ = RGNN(eeg)

      noise = mixing_noise(args.batch, args.latent, args.mixing, device)
      fake_img, _ = generator(noise, eeg_latent)

      if args.augment:
          fake_img, _ = augment(fake_img, ada_aug_p)

      fake_pred = discriminator(fake_img, labels)
      g_loss = g_nonsaturating_loss(fake_pred)

      """pred_emo = aux_classifier(fake_img)
      loss_aux = F.cross_entropy(pred_emo, torch.argmax(labels,dim=1)) * lambda_aux
      g_loss += loss_aux

      loss_style = crit_style(fake_img, real_img) * lambda_style
      g_loss += loss_style"""

      loss_dict["g"] = g_loss

      generator.zero_grad()
      g_loss.backward()
      g_optim.step()

      g_regularize = i % args.g_reg_every == 0

      if g_regularize:
          path_batch_size = max(1, args.batch // args.path_batch_shrink)
          noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
          fake_img, latents = generator(noise, eeg_latent[:path_batch_size], return_latents=True)

          path_loss, mean_path_length, path_lengths = g_path_regularize(
              fake_img, latents, mean_path_length
          )

          generator.zero_grad()
          weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

          if args.path_batch_shrink:
              weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

          weighted_path_loss.backward()

          g_optim.step()

          mean_path_length_avg = (
              reduce_sum(mean_path_length).item() / get_world_size()
          )

      loss_dict["path"] = path_loss
      loss_dict["path_length"] = path_lengths.mean()

      accumulate(g_ema, g_module, accum)

      ###############
      # train RGNN  #
      ###############

      """requires_grad(generator, False)
      requires_grad(discriminator, False)
      requires_grad(RGNN, True)

      eeg_latent, eeg_out, _ = RGNN(eeg)

      loss_KL = F.kl_div(eeg_out, eeg["y"])
      loss_L1 = torch.norm(adjacency_matrix, p=1) * lambda_eeg_reg
      rgnn_loss = loss_KL + loss_L1

      noise = mixing_noise(args.batch, args.latent, args.mixing, device)
      fake_img, _ = generator(noise, eeg_latent)

      if args.augment:
        fake_img, _ = augment(fake_img, ada_aug_p)

      pred_emo = aux_classifier(fake_img)
      loss_aux = F.cross_entropy(pred_emo, torch.argmax(labels,dim=1)) * lambda_aux
      rgnn_loss += loss_aux

      loss_style = crit_style(fake_img, real_img) * lambda_style
      rgnn_loss += loss_style

      loss_dict["rgnn"] = rgnn_loss
      loss_dict["rgnn_accuracy"] = np.sum(np.argmax(eeg_out.cpu().detach().numpy(), axis=1) == np.argmax(labels.cpu().detach().numpy(), axis=1))/len(labels)

      RGNN.zero_grad()
      rgnn_loss.backward()
      RGNN_optim.step()"""


      ###############
      #   report    #
      ###############

      loss_reduced = reduce_loss_dict(loss_dict)

      d_loss_val = loss_reduced["d"].mean().item()
      g_loss_val = loss_reduced["g"].mean().item()
      #rgnn_loss_val = loss_reduced["rgnn"].mean().item()
      #rgnn_accuracy_val = loss_dict["rgnn_accuracy"].mean().item()
      r1_val = loss_reduced["r1"].mean().item()
      path_loss_val = loss_reduced["path"].mean().item()
      real_score_val = loss_reduced["real_score"].mean().item()
      fake_score_val = loss_reduced["fake_score"].mean().item()
      path_length_val = loss_reduced["path_length"].mean().item()


      if get_rank() == 0:
          pbar.set_description(
              (
                  f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                  f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                  f"augment: {ada_aug_p:.4f};"# rgnn: {rgnn_loss_val:.4f}; rgnn_accuracy: {rgnn_accuracy_val:.4f}"
              )
          )

          if wandb and args.wandb:
              wandb.log(
                  {
                      "Generator": g_loss_val,
                      "Discriminator": d_loss_val,
                      "Augment": ada_aug_p,
                      "Rt": r_t_stat,
                      "R1": r1_val,
                      "Path Length Regularization": path_loss_val,
                      "Mean Path Length": mean_path_length,
                      "Real Score": real_score_val,
                      "Fake Score": fake_score_val,
                      "Path Length": path_length_val,
                  }
              )

          if i % 100 == 0:
              with torch.no_grad():
                  g_ema.eval()
                  sample, _ = g_ema([sample_z], torch.cat([RGNN(eeg)[0] for eeg in sample_eeg],dim=0))
                  utils.save_image(
                      sample,
                      os.path.join(args.output_folder,f"images/{str(i).zfill(6)}.png"),
                      nrow=int(args.n_sample ** 0.5),
                      normalize=True,
                      range=(-1, 1),
                  )

          if i % 1000 == 0:
              with torch.no_grad():
                for label,eeg_sample_list in enumerate([sample_neutral,sample_sadness,sample_fear,sample_happiness]):
                    sample, _ = g_ema([sample_z], torch.cat([RGNN(eeg)[0] for eeg in eeg_sample_list],dim=0))
                    utils.save_image(
                        sample,
                        os.path.join(args.output_folder,f"images/{str(i).zfill(6)}_{label}.png"),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
              torch.save(
                  {
                      "g": g_module.state_dict(),
                      "d": d_module.state_dict(),
                      "g_ema": g_ema.state_dict(),
                      "RGNN": RGNN.state_dict(),
                      "g_optim": g_optim.state_dict(),
                      "d_optim": d_optim.state_dict(),
                      "RGNN_optim": RGNN_optim.state_dict(),
                      "args": args,
                      "ada_aug_p": ada_aug_p,
                  },
                  os.path.join(args.output_folder,f"models/{str(i).zfill(6)}.pt"),
              )