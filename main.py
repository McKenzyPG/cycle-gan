import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim 

from network import *
from dataset import * 
import visdom 

batch_size = 32
nc_mnist = 1
nc_svhn = 3
nf = 64
lr = 0.003
iters = 10000
log_interval = 10
use_cuda= torch.cuda.is_available()
# load datasets
svhn_loader, mnist_loader = get_mnist_svhn_loader(batch_size)

# custom weights initialization called on G and D
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# mnist to svhn
G12 = GenM2S(1, 64, 3)
G12.apply(weights_init)
print(G12)

# svhn to mnist 
G21 = GenS2M(3, 64, 1)
G21.apply(weights_init)
print(G21)

# D1 (MNIST)
D1 = Discriminator(1)
D1.apply(weights_init)
print(D1)

# D2 (SVHN)
D2 = Discriminator(3)
D2.apply(weights_init)
print(D2)

# loss and optimizers
criterion = nn.CrossEntropyLoss()

optim_g = optim.Adam(list(G12.parameters()) + list(G21.parameters()), 
                    lr, [0.5, 0.999])

optim_d = optim.Adam(list(D1.parameters()) + list(D2.parameters()),
                     lr=lr)


if use_cuda:
    G12.cuda()
    G21.cuda()
    D1.cuda()
    D2.cuda()
    criterion.cuda()
    optim_d.cuda()
    optim_g.cuda()

def to_var(x):
    if use_cuda:
        x = x.cuda()
    return Variable(x)

def to_data(x):
    if use_cuda:
        x = x.cpu()
    return x.data.numpy()

def reset_grad():
    optim_g.zero_grad()
    optim_d.zero_grad()

def train_d(s_var, m_var, is_real=True):
    # train D
    # train with real image
    reset_grad()

    #forward mnist
    m_out = D1(m_var)

    # discriminator loss  mnist
    if is_real:
        m_d_loss = torch.mean((m_out-1)**2)
    else:
        m_d_loss = torch.mean(m_out**2)

    # forward svhn
    s_out = D2(s_var)

    # D_loss svhn
    if is_real:
        s_d_loss = torch.mean((s_out-1)**2)
    else:
        s_d_loss = torch.mean((s_out)**2)
    
    # total d ral image loss
    d_real_loss = m_d_loss + s_d_loss

    # compute backward pass
    d_real_loss.backward()

    # update weights
    optim_d.step()

    return m_d_loss, s_d_loss


vis = visdom.Visdom()
lot = vis.line(
    X=torch.zeros((1,)).cpu(),
    Y=torch.zeros((1, 2)).cpu(),
    opts=dict(
        xlabel='Iterations',
        ylabel='Loss',
        title='Current Losses',
        legend=['Gen Loss', 'Disc Loss']
    ))

    
def train():
    svhn_iter = iter(svhn_loader)
    mnist_iter = iter(mnist_loader)
    iter_per_epoch = min(len(svhn_iter), len(mnist_iter))

    fixed_svhn_var = to_var(svhn_iter.next()[0])
    fixed_mnist_var = to_var(mnist_iter.next()[0])
    count = 0
    for step in range(iters):
        if (step+1) % iter_per_epoch == 0:
            print("New epoch starts")
            svhn_iter = iter(svhn_loader)
            mnist_iter = iter(mnist_loader)

        s_data, s_labels = svhn_iter.next()
        s_data, s_labels = to_var(s_data), to_var(s_labels)
        m_data, m_labels = mnist_iter.next()
        m_data, m_labels = to_var(m_data), to_var(m_labels)

        # train D with real
        s_real_loss, m_real_loss = train_d(s_data, m_data)

        # train D with fake
        # get fake svhn
        s_fake = G12(m_data)

        # get fake mnist
        m_fake = G21(s_data)

        s_fake_loss, m_fake_loss = train_d(s_fake, m_fake, is_real=False)

        # train G
        # mnist--> svhn--> mnist
        reset_grad()
        s_fake = G12(m_data)
        m_fake = G21(s_fake)
        s_out = D2(s_fake)

        # print(m_fake, m_data)
        msm_recons_loss = torch.mean((m_data-m_fake)**2)
        msm_d_loss = torch.mean((s_out-1)**2)

        # total loss and update
        msm_g_loss = msm_recons_loss + msm_d_loss
        msm_g_loss.backward()
        optim_g.step()

        # svhn --> mnist-->svhn
        reset_grad()
        m_fake = G21(s_data)
        s_fake = G12(m_fake)
        m_out = D1(m_fake)
        sms_recons_loss = torch.mean((s_data-s_fake)**2)
        sms_d_loss = torch.mean((m_out-1)**2)

        # total loss and update
        sms_g_loss = sms_recons_loss + sms_d_loss
        sms_g_loss.backward()
        optim_g.step()

        if(step+1) % log_interval == 0:
            print('Step [%d], d_loss: %.4f, d_fake_loss: %.4f, g_loss: %.4f'
                  % (step+1, (s_real_loss+m_real_loss).data[0], (s_fake_loss+m_fake_loss).data[0], (msm_g_loss + sms_g_loss).data[0]))
#         vis.line(
#             X=torch.ones((1, 2)).cpu()*count,
#             Y=torch.Tensor(
#                 [(msm_g_loss + sms_g_loss).data[0], (s_real_loss+m_real_loss).data[0]]).unsqueeze(0).cpu(),
#             win=lot,
#             update='append'
#         )
        count += 1

#         if step % 100 == 0:
#             m1_fake_fixed=G21(fixed_svhn_var)
#             s1_fake_fixed=G12(m1_fake_fixed)


#             s2_fake_fixed=G12(fixed_mnist_var)
#             m2_fake_fixed=G21(s2_fake_fixed)

#             concat_t=torch.cat(
#                 [fixed_svhn_var.cpu(), s1_fake_fixed.cpu().data, fixed_mnist_var.cpu().data, m2_fake_fixed.cpu().data], dim=2)

#             grid=vutils.make_grid(concat_t)
#             ndarr=grid.mul(255).clamp(0, 255).byte().numpy()

#             vis.image(ndarr, opts=dict(title='Recons',
#                                         caption='Epoch {} iter {}'.format(count, i)))




train()

