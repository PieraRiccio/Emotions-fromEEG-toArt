from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import numpy as np
import itertools
import torch
import os
import random

from Models.loss import StyleLoss, PerceptualLoss, calc_gradient_penalty, GeneratorLoss, DiscriminatorLoss
from Models.stylegan_model import Generator, Discriminator
from Models.rgnn_model import SymSimGCNNet

def accumulate(model1, model2, decay=0.5 ** (32 / (10 * 1000))):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
        
class SVN(nn.Module):
    # def __init__(self, z_dims=32, out_class=92):
    def __init__(self, z_dim=32, image_size = 128, out_class=92, regression = False, adjacency_matrix = torch.eye(62)):
        """
            The constructor of style visualization network
            There are x loss term:
            write them
        """
        super().__init__()
        self.z_dim = z_dim
        self.regression = regression
        self.adjacency_matrix = adjacency_matrix

        # Define the loss term weight
        
        self.lambda_style = 10.0
        self.lambda_adver = 1.0
        self.lambda_class = 1e-1
        self.lambda_KL = 1.0
        self.alpha = 0.001 # this is the alpha to be tuned in the RGNN model

        # Define loss lists
        
        self.loss_list_KL = []
        self.loss_list_nodeDAT = []
        self.loss_list_L1Reg = []
        self.loss_list_style = []
        self.loss_list_adver_g = [] 
        self.loss_list_adver_d = []
        self.loss_list_reg_g = [] 
        self.loss_list_reg_d = []
        self.loss_list_class = []
        
        self.Loss_list_KL = []
        self.Loss_list_nodeDAT = []
        self.Loss_list_L1Reg = []
        self.Loss_list_style = []
        self.Loss_list_adver_g = []
        self.Loss_list_adver_d = []
        self.Loss_list_reg_g = []
        self.Loss_list_reg_d = []
        self.Loss_list_class = []

        # Define network structure
        # TODO: look at output dimension for latent vector --> or base the latent_dim on the hyperparameter tuning
        
        self.RGNN = SymSimGCNNet(62, True, torch.tensor(adjacency_matrix, dtype=torch.float32), 5, (64, 64), 4, 2, dropout=0.7, domain_adaptation="RevGrad")
        self.G = Generator(image_size, style_dim=512, n_mlp=8, channel_multiplier=2)

        self.G_ema = Generator(image_size, style_dim=512, n_mlp=8, channel_multiplier=2).eval()
        accumulate(self.G_ema, self.G, 0)

        self.D = Discriminator(image_size, channel_multiplier=2)
        self.aux_C = torch.load("/content/drive/MyDrive/Pieroncina/alexnet.pth").eval()

        # Define optimizer
        
        self.optim_RGNN = torch.optim.Adam(self.RGNN.parameters(), lr=2e-4, weight_decay=1e-5) # TODO: but the parameters need to be tuned in a separate experiment
        
        g_reg_every = 4 #TODO: decide if to pass as parameters
        d_reg_every = 16
        g_reg_ratio = g_reg_every / (g_reg_every + 1)
        d_reg_ratio = d_reg_every / (d_reg_every + 1)
        self.optim_G = torch.optim.Adam(self.G.parameters(), lr=0.002 * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),)
        self.optim_D = torch.optim.Adam(self.D.parameters(), lr=0.002 * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),)

        # Define criterion
        
        self.KL = nn.KLDivLoss()
        self.nodeDAT = nn.BCELoss() # TODO: put the proper formula of the nodeDAT, if necessary
        # self.L1 = L1Reg(self.alpha) # TODO: alpha has to be tuned
        self.vgg = models.vgg19(pretrained=True).features.eval()
        self.crit_style = StyleLoss(vgg_module = self.vgg) # Mysterious style loss
        self.d_logistic_loss = DiscriminatorLoss().D_logistic()
        self.d_reg = DiscriminatorLoss().D_reg()
        self.g_nonsaturating_loss = GeneratorLoss().G_nonsaturating_loss()
        self.g_reg = GeneratorLoss().G_reg()

        # TODO: understand if AlexNet needs regression or not, and its shape output
        if self.regression:
            self.crit_class = nn.MSELoss()
        else:
            self.crit_class = nn.CrossEntropyLoss()
        self.mixing_noise(16, 512, prob=0.9)


    def make_noise(self, batch, latent_dim, n_noise):
        if n_noise == 1:
            return torch.randn(batch, latent_dim).cuda() 
        noises = torch.randn(n_noise, batch, latent_dim).cuda().unbind(0)

        return noises

    def mixing_noise(self, batch, latent_dim, prob):
      if prob > 0 and random.random() < prob:
        self.z = self.make_noise(batch, latent_dim, 2)

      else:
        self.z = [self.make_noise(batch, latent_dim, 1)]

    def get_random_labels(self, batch, n_labels):
      # this is just to test, not needed in the real implementation
      labels = (torch.rand(batch)*n_labels).long()
      labels = torch.nn.functional.one_hot(labels, num_classes=n_labels).float().cuda()
      return labels

    #def forward(self, spec_in):
    def forward(self, eeg_in, beta=0): # TODO: capire bene beta
                
        # 1. Generate the latent vector from the eeg signals using RGNN and classify it
        eeg_latent, eeg_out, domain_out = self.RGNN(eeg_in, beta)
        eeg_latent = eeg_latent.reshape(eeg_latent.shape[0],1,8,8) # TODO: am I sure about these dimensions?

        eeg_latent = self.get_random_labels(16, 4) # THIS IS JUST TO TEST
        fake_style = self.G(self.z, eeg_latent) # TODO: to test, the eeg labels will just be the fake labels

        # 2. Emotion classification with auxiliary classifier
        pred_emo = self.aux_C(fake_style) 

        return eeg_out, fake_style, pred_emo, eeg_latent, domain_out

    # ==============================================================================
    #   IO
    #   TODO: write load state dict and save for RGNN and C
        # for now it is not a problem, because we are not doing checkpoints yet
    # ==============================================================================
    def load(self, path):
        if os.path.exists(path):
            print("Load the pre-trained model from {}".format(path))
            state = torch.load(path)
            for (key, obj) in state.items():
                if len(key) > 10:
                    if key[1:9] == 'oss_list':
                        setattr(self, key, obj)
            self.U.load_state_dict(state['U'])
            self.G.load_state_dict(state['G'])
            self.D.load_state_dict(state['D'])
            self.C.load_state_dict(state['C'])
            self.optim_U.load_state_dict(state['optim_U'])
            self.optim_G.load_state_dict(state['optim_G'])
            self.optim_D.load_state_dict(state['optim_D'])
            self.optim_C.load_state_dict(state['optim_C'])
        else:
            print("Pre-trained model {} is not exist...".format(path))

    def save(self, path):
        # Record the parameters TODO
        state = {
            'U': self.U.state_dict(),
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'C': self.C.state_dict(),
        }

        # Record the optimizer and loss
        state['optim_U'] = self.optim_U.state_dict()
        state['optim_G'] = self.optim_G.state_dict()
        state['optim_D'] = self.optim_D.state_dict()
        state['optim_C'] = self.optim_C.state_dict()
        for key in self.__dict__:
            if len(key) > 10:
                if key[1:9] == 'oss_list':
                    state[key] = getattr(self, key)
        torch.save(state, path)

    # ==============================================================================
    #   Set & Get
    # ==============================================================================
    def getLoss(self, normalize = False):
        loss_dict = {}
        for key in self.__dict__:
            if len(key) > 9 and key[0:9] == 'loss_list':
                if not normalize:
                    loss_dict[key] = round(getattr(self, key)[-1], 6)
                else:
                    loss_dict[key] = np.mean(getattr(self, key))
        return loss_dict

    def getLossList(self):
        loss_dict = {}
        for key in self.__dict__:
            if len(key) > 9 and key[0:9] == 'Loss_list':
                loss_dict[key] = getattr(self, key)
        return loss_dict

    # ==============================================================================
    #   Backward function 
    # ==============================================================================
        
        
    def backward_RGNN(self, eeg_out, gt_KL, adjacency_matrix, domain_output, domain_label, pred_emo, gt_emo, true_style, fake_style):
        """
        eeg loss (KL + CE + L1) + emotion classification loss (on the painting) + Style loss
        """
        
        # RGNN classification loss:
        
        loss_KL = self.KL(eeg_out, gt_KL)
        self.loss_list_KL.append(loss_KL.item())
        loss_nodeDAT = self.nodeDAT(domain_output, domain_label)
        self.loss_list_nodeDAT.append(loss_nodeDAT.item())
        loss_L1 = torch.norm(adjacency_matrix, p=1) * self.alpha #TODO: tune alpha
        self.loss_list_L1Reg.append(loss_L1.item())
        
        loss_class = self.crit_class(pred_emo, gt_emo) * self.lambda_class 
        self.loss_list_class.append(loss_class.item())
       
        loss_style = self.crit_style(fake_style, true_style) * self.lambda_style
        self.loss_list_style.append(loss_style.item())
        
        # sum the losses of the RGNN
        loss_rgnn = loss_KL + loss_nodeDAT + loss_L1
        loss_e = loss_rgnn + loss_style + loss_class
        
        loss_e.backward()

    def backward_G(self, true_style, fake_style, pred_emo, gt_emo):
        """
        Adversarial loss + emotion classification loss (from painting) + Style loss
        """
        # ---------------------------------------------------------------
        # Update for adversarial loss
        # ---------------------------------------------------------------
        ##### WGAN-GP
        # loss_adv = -self.D(fake_style).mean() * self.lambda_adver
        ##### Relativistic LSGAN
        r_logit = self.D(true_style)
        f_logit = self.D(fake_style)
        loss_adv = ((torch.mean((r_logit - torch.mean(f_logit) + 1) ** 2) + torch.mean((f_logit - torch.mean(r_logit) - 1) ** 2))/2) * self.lambda_adver
        self.loss_list_adver_g.append(loss_adv.item())

        # Emotion classification loss (auxiliary classifier)
        loss_class = self.crit_class(pred_emo, gt_emo) * self.lambda_class 
        self.loss_list_class.append(loss_class.item())

        # Update for style loss
        loss_style = self.crit_style(fake_style, true_style) * self.lambda_style
        self.loss_list_style.append(loss_style.item())

        # Merge the several loss terms
        loss_g = loss_style + loss_class + loss_adv
        loss_g.backward()
        
    def backward_D(self, true_style, fake_style):
        """
        Adversarial loss
        """
        loss_adv = 0.0
        ##### WGAN-GP
        # for iter_d in range(3):
        #     r_logit = self.D(true_style)
        #     f_logit = self.D(fake_style)
        #     gradient_penalty = calc_gradient_penalty(self.D, true_style, fake_style)
        #     loss_adv = loss_adv + (f_logit.mean() - r_logit.mean() + gradient_penalty) * self.lambda_adver
        
        ##### Relativistic LSGAN
        r_logit = self.D(true_style)
        f_logit = self.D(fake_style)
        loss_adv = loss_adv + ((torch.mean((r_logit - torch.mean(f_logit) - 1) ** 2) + torch.mean((f_logit - torch.mean(r_logit) + 1) ** 2))/2) * self.lambda_adver

        self.loss_list_adver_d.append(loss_adv.item())
        loss_adv.backward()

    #def backward(self, in_spec, true_style, gt_year, pos = None, neg = None):
    def backward(self, eeg_in, true_style, gt_emo, domain_label): # TODO: do I have to put batch size also?
        """
            Update the parameters of whole model
           
            Args:   eeg_in      (torch.Tensor)  - The input EEG signal. The shape is 
                    true_style  (torch.Tensor)  - GT image. The shape is (B, 3, 256, 256)
                    gt_emo      (torch.Tensor)  - Ground truth emotion (distribution), a vector with 4 entries
                    batch_size  (Int)           - Need batch_size to reshape tensor when calculating year classification loss with batch_size=1.
        """
        
        
        # Update discriminator
        eeg_out, fake_style, pred_emo, eeg_latent, domain_out = self.forward(eeg_in)
        self.optim_D.zero_grad()
        self.backward_D(true_style, fake_style)
        self.optim_D.step()

        # Update generator
        eeg_out, fake_style, pred_emo, eeg_latent, domain_out = self.forward(eeg_in)
        self.optim_G.zero_grad()
        self.backward_G(true_style, fake_style, pred_emo, gt_emo)
        self.optim_G.step()
        
        # Update RGNN
        gt_KL = eeg_in["y"]
        eeg_out, fake_style, pred_emo, eeg_latent, domain_out = self.forward(eeg_in)
        self.optim_RGNN.zero_grad()
        self.backward_RGNN(eeg_out, gt_KL, self.adjacency_matrix, domain_out, domain_label, pred_emo, gt_emo, true_style, fake_style)
        self.optim_RGNN.step()

    # ====================================================================================================================================
    #   Finish epoch --> HISTORY
    # ====================================================================================================================================
    def finish(self):
        for key in self.__dict__:
            if len(key) > 9 and key[0:9] == 'loss_list': # small l is the loss in the batches (dynamic)
                sum_lost_list = getattr(self, 'L' + key[1:]) # big L takes the mean over all batches (static)
                sum_lost_list.append(np.mean(getattr(self, key)))       
                setattr(self, 'L' + key[1:], sum_lost_list)
                print('L'+key[1:]+ ": ----> ")
                print(sum_lost_list[-1])
                setattr(self, key, [])

# if __name__ == "__main__":

#    # Visible GPU
#    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(s) for s in [0, 1])

#    # Test one iteration
#    print ("Test one iteration")

#    model = SVN()
#    model = torch.nn.DataParallel(model).cuda()

#    B = 1

#    in_spec = torch.rand([B, 3, 128, 259]).cuda()
#    true_style = torch.rand([B, 3, 256, 256]).cuda()
#    gt_year = torch.empty(B, dtype=torch.long).random_(98).cuda()

#    # Start Testing
#    model.module.backward(in_spec, true_style, gt_year, batch_size=B)
#    print ("Pass")
# model = SVN(z_dim=256, image_size=64, out_class=54).cuda()
# model.load('experiment_result_history/20190729/Result_no_reg_with_triplet_ch256_1_8.91_exp1/Model/Result_no_reg_with_triplet_ch256_1_8.91_exp1.pth')
