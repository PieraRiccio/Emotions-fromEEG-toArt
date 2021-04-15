import os
import matplotlib.pyplot as plt
import copy
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import SGConv, global_add_pool
from torch_scatter import scatter_add

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################
######SAMPLER#######
####################

class EEG_Sampler:
  def __init__(self,eeg_dataloaders):
    self.len_loaders=[
      len(loader) for loader in eeg_dataloaders
    ]
    self.eeg_dataloaders=[
      itertools.cycle(loader) if len(loader)!=max(self.len_loaders) else loader for loader in eeg_dataloaders
    ]
    self.sampler=self.sample_next()
    self.counter=0
  
  def __len__(self):
    return max(self.len_loaders)*len(self.len_loaders)
  
  def __iter__(self):
    self.counter=0
    return self

  def sample_next(self):
    while True:
      for batches in zip(*self.eeg_dataloaders):
        yield batches[0]
        yield batches[1]
        yield batches[2]
        yield batches[3]
  
  def __next__(self):
    if(self.counter==len(self)):
      raise StopIteration
    self.counter+=1
    return next(self.sampler)

####################
#######MODEL########
####################

def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes

def add_remaining_self_loops(edge_index,
                             edge_weight=None,
                             fill_value=1,
                             num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    mask = row != col
    inv_mask = torch.logical_not(mask)
    loop_weight = torch.full( 
        (num_nodes, ),
        fill_value,
        dtype=None if edge_weight is None else edge_weight.dtype,
        device=edge_index.device)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        remaining_edge_weight = edge_weight[inv_mask]
        if remaining_edge_weight.numel() > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight
        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)

    loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    return edge_index, edge_weight


class NewSGConv(SGConv):
    def __init__(self, num_features, num_classes, K=1, cached=False,
                 bias=True):
        super(NewSGConv, self).__init__(num_features, num_classes, K=K, cached=cached, bias=bias)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        deg = scatter_add(torch.abs(edge_weight), row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        if not self.cached or self.cached_result is None:
            edge_index, norm = NewSGConv.norm(
                edge_index, x.size(0), edge_weight, dtype=x.dtype)

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x

        return self.lin(self.cached_result)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class SymSimGCNNet(torch.nn.Module):
    def __init__(self, num_nodes, learn_edge_weight, edge_weight, num_features, num_hiddens, num_classes, K, dropout=0.5, domain_adaptation=""):
        super(SymSimGCNNet, self).__init__()
        self.domain_adaptation = domain_adaptation
        self.num_nodes = num_nodes
        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=0)
        edge_weight = edge_weight.reshape(self.num_nodes, self.num_nodes)[self.xs, self.ys]
        self.edge_weight = nn.Parameter(edge_weight, requires_grad=learn_edge_weight)
        self.dropout = dropout
        self.conv1 = NewSGConv(num_features=num_features, num_classes=num_hiddens[0], K=K)
        self.fc = nn.Linear(num_hiddens[0], num_classes)
        if self.domain_adaptation in ["RevGrad"]:
            self.domain_classifier = nn.Linear(num_hiddens[0], 2)

    def forward(self, data, alpha=0):
        batch_size = len(data.y)
        x, edge_index = data.x, data.edge_index
        edge_weight = torch.zeros((self.num_nodes, self.num_nodes), device=edge_index.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + edge_weight.transpose(1,0) - torch.diag(edge_weight.diagonal())
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        x = F.relu(self.conv1(x, edge_index, edge_weight))

        domain_output = None
        if self.domain_adaptation in ["RevGrad"]:
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_x)
            domain_output = F.softmax(domain_output)
        x = global_add_pool(x, data.batch, size=batch_size)
        latent_v = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(latent_v)
        x = F.log_softmax(x)
        return latent_v, x, domain_output

####################
######TRAINING######
####################

class Loss():
  def __init__(self):
    self.KL = torch.nn.KLDivLoss()

  def L1Reg(self, adjacency_matrix, alpha=0.001):
    return alpha*torch.norm(adjacency_matrix, p=1)

  def get_contributes(self, prediction, label, adjacency_matrix):
    contributes={}
    contributes["KL"] = self.KL(prediction, label).item()
    contributes["L1Reg"] = self.L1Reg(adjacency_matrix).item()
    contributes["Total"] = contributes["KL"] + contributes["L1Reg"]
    return contributes

  def __call__(self, prediction, label, adjacency_matrix):
    loss=self.KL(prediction,label) + self.L1Reg(adjacency_matrix)
    return loss

def training_routine(model, epochs, train_loader, val_loader, ckpt_folder):
  if not os.path.isdir(ckpt_folder):
    os.mkdir(ckpt_folder)
  history = []
  optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
  loss_function=Loss()
  best_acc = None
  for epoch in range(epochs):
    model.train()

    for batch in train_loader:
      batch = batch.to(device)
      optimizer.zero_grad()
      _,prediction,_=model.forward(batch)
      
      loss=loss_function(prediction, batch["y"], model.edge_weight)
      loss.backward()
      optimizer.step()

    model.eval()
    result = evaluation_routine(model, val_loader, loss_function)
    if(best_acc==None or result['Total']<best_acc):
      best_acc=result['Total']
      best_model_dict = copy.deepcopy(model.state_dict())
      torch.save(best_model_dict,os.path.join(ckpt_folder,"rgnn.pth"))
    epoch_end(epoch, result)
    history.append(result)
  return history, best_model_dict

def evaluation_routine(model, val_loader, loss_function):
  epoch_loss={}
  batch_losses={}
  for batch in val_loader:
    batch = batch.to(device)
    _,prediction, _ = model.forward(batch)

    for k,v in loss_function.get_contributes(prediction, batch["y"], model.edge_weight).items():
      if k not in batch_losses.keys():
        batch_losses[k]=[]
      batch_losses[k].append(v)
  for k in batch_losses.keys():
    epoch_loss[k] = sum(batch_losses[k])/len(batch_losses[k])
  return epoch_loss

def epoch_end(epoch, result):
  output="Epoch [{}], ".format(epoch)
  for k,v in result.items():
    output+="{}: {:.4f} ".format(k,v)
  print(output)

def plot_history(history):
  losses = [x['Total'] for x in history]
  plt.plot(losses, '-x', label="loss")
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend()
  plt.title('Losses vs. No. of epochs')
  plt.grid()
  plt.show()