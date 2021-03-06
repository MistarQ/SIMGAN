import utils.functions as functions
import models.model as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from utils.imresize import imresize
import itertools
from torchvision.utils import save_image
from torchvision import transforms as T
from PIL import Image
import numpy as np


def d_loss_cls(opt, predict, label):
    # why label should be [label]
    label = torch.Tensor([label]).to(opt.device)
    # CELoss=nn.CrossEntropyLoss()
    loss = nn.BCELoss()
    return loss(predict, label)


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def train(opt, Gs, Zs, reals, NoiseAmp, Gs2, Zs2, reals2, NoiseAmp2, Gs3, Zs3, reals3, NoiseAmp3):
    real_, real_2, real_3 = functions.read_three_domains(opt)

    in_s = 0
    in_s2 = 0
    in_s3 = 0

    scale_num = 0

    real = imresize(real_, opt.scale1, opt)
    real2 = imresize(real_2, opt.scale1, opt)
    real3 = imresize(real_3, opt.scale1, opt)

    reals = functions.creat_reals_pyramid(real, reals, opt)
    reals2 = functions.creat_reals_pyramid(real2, reals2, opt)
    reals3 = functions.creat_reals_pyramid(real3, reals3, opt)
    nfc_prev = 0

    while scale_num < opt.stop_scale + 1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        D_curr, G_curr, D_curr2, G_curr2 = init_models(opt)

        if nfc_prev == opt.nfc:
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_, scale_num - 1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1)))
            G_curr2.load_state_dict(torch.load('%s/%d/netG2.pth' % (opt.out_, scale_num - 1)))
            D_curr2.load_state_dict(torch.load('%s/%d/netD2.pth' % (opt.out_, scale_num - 1)))

        z_curr, in_s, G_curr, z_curr2, in_s2, G_curr2, z_curr3, in_s3 = train_single_scale(D_curr, G_curr, reals, Gs,
                                                                                           Zs, in_s,
                                                                                           NoiseAmp, D_curr2, G_curr2,
                                                                                           reals2, Gs2, Zs2,
                                                                                           in_s2, NoiseAmp2, reals3,
                                                                                           Gs3, Zs3,
                                                                                           in_s3, NoiseAmp3, opt,
                                                                                           scale_num)

        G_curr = functions.reset_grads(G_curr, False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr, False)
        D_curr.eval()

        G_curr2 = functions.reset_grads(G_curr2, False)
        G_curr2.eval()
        D_curr2 = functions.reset_grads(D_curr2, False)
        D_curr2.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        Gs2.append(G_curr2)
        Zs2.append(z_curr2)
        NoiseAmp2.append(opt.noise_amp2)

        #
        Gs3.append(G_curr2)  # G3 = G2
        Zs3.append(z_curr3)
        NoiseAmp3.append(opt.noise_amp3)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        torch.save(Zs2, '%s/Zs2.pth' % (opt.out_))
        torch.save(Gs2, '%s/Gs2.pth' % (opt.out_))
        torch.save(reals2, '%s/reals2.pth' % (opt.out_))
        torch.save(NoiseAmp2, '%s/NoiseAmp2.pth' % (opt.out_))

        torch.save(Zs3, '%s/Zs3.pth' % (opt.out_))
        torch.save(Gs3, '%s/Gs3.pth' % (opt.out_))
        torch.save(reals3, '%s/reals3.pth' % (opt.out_))
        torch.save(NoiseAmp3, '%s/NoiseAmp3.pth' % (opt.out_))

        scale_num += 1
        nfc_prev = opt.nfc
        del D_curr, G_curr, D_curr2, G_curr2
    return


def train_single_scale(netD, netG, reals, Gs, Zs, in_s, NoiseAmp, netD2, netG2, reals2, Gs2, Zs2, in_s2, NoiseAmp2,
                       reals3, Gs3, Zs3,
                       in_s3, NoiseAmp3, opt,
                       scale_num, centers=None):
    real = reals[len(Gs)]
    real2 = reals2[len(Gs2)]
    real3 = reals3[len(Gs3)]
    save_image(denorm(real.data.cpu()), '%s/real_scale.png' % opt.outf)
    save_image(denorm(real2.data.cpu()), '%s/real_scale2.png' % opt.outf)
    save_image(denorm(real3.data.cpu()), '%s/real_scale3.png' % opt.outf)

    opt.bsz = real.shape[0]
    opt.nzx = real.shape[2]
    opt.nzy = real.shape[3]

    pad_noise = 0
    pad_image = 0
    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))

    lambda_idt = opt.lambda_idt
    lambda_cyc = opt.lambda_cyc
    lambda_tv = opt.lambda_tv

    z_opt = torch.full([opt.bsz, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
    z_opt = m_noise(z_opt)
    z_opt2 = torch.full([opt.bsz, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
    z_opt2 = m_noise(z_opt2)
    z_opt3 = torch.full([opt.bsz, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
    z_opt3 = m_noise(z_opt3)

    # setup optimizer
    optimizerD = optim.Adam(itertools.chain(netD.parameters(), netD2.parameters()), lr=opt.lr_d,
                            betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(itertools.chain(netG.parameters(), netG2.parameters()), lr=opt.lr_g,
                            betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600], gamma=opt.gamma)

    loss_print = {}

    for epoch in range(opt.niter):
        noise_ = functions.generate_noise([3, opt.nzx, opt.nzy], device=opt.device)
        noise_ = m_noise(noise_.expand(opt.bsz, 3, opt.nzx, opt.nzy))

        noise_2 = functions.generate_noise([3, opt.nzx, opt.nzy], device=opt.device)
        noise_2 = m_noise(noise_2.expand(opt.bsz, 3, opt.nzx, opt.nzy))

        noise_3 = functions.generate_noise([3, opt.nzx, opt.nzy], device=opt.device)
        noise_3 = m_noise(noise_3.expand(opt.bsz, 3, opt.nzx, opt.nzy))

        ############################
        # (1) Update D network
        ###########################
        for j in range(opt.Dsteps):
            optimizerD.zero_grad()

            output_B, cls_B = netD(real2)  # image and label for B
            output_B = output_B.to(opt.device)
            cls_B = cls_B.to(opt.device)
            errD_real_B = -output_B.mean()  # errD_real_B
            errD_real_B.backward(retain_graph=True)
            errD_cls_real_B = d_loss_cls(opt, cls_B, [0, 1])  # errD_cls_real_B
            errD_cls_real_B.backward(retain_graph=True)
            loss_print['errD_real_B'] = errD_real_B.item()
            loss_print['errD_cls_real_B'] = errD_cls_real_B.item()

            output_C, cls_C = netD(real3)  # image and label for C
            output_C = output_C.to(opt.device)
            cls_C = cls_C.to(opt.device)  # errD_real_C
            errD_real_C = -output_C.mean()
            errD_real_C.backward(retain_graph=True)
            errD_cls_real_C = d_loss_cls(opt, cls_C, [1, 0])  # errD_cls_real_C
            errD_cls_real_C.backward(retain_graph=True)
            loss_print['errD_real_C'] = errD_real_C.item()
            loss_print['errD_cls_real_C'] = errD_cls_real_C.item()

            output_A = netD2(real).to(opt.device)
            errD_real_A = -output_A.mean()
            errD_real_A.backward(retain_graph=True)
            loss_print['errD_real_A'] = errD_real_A.item()

            if (j == 0) & (epoch == 0):
                if Gs == []:
                    prev_from2 = torch.full([opt.bsz, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    prev_from3 = torch.full([opt.bsz, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    in_s = torch.full([opt.bsz, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    prev_from2 = m_image(prev_from2)
                    prev_from3 = m_image(prev_from3)
                    c_prev_from2 = torch.full([opt.bsz, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    c_prev_from3 = torch.full([opt.bsz, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    z_prev = torch.full([opt.bsz, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    z_prev = m_noise(z_prev)

                    prev2 = torch.full([opt.bsz, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    in_s2 = prev2
                    prev2 = m_image(prev2)
                    c_prev2 = torch.full([opt.bsz, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    z_prev2 = torch.full([opt.bsz, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    z_prev2 = m_noise(z_prev2)

                    prev3 = torch.full([opt.bsz, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    in_s3 = prev3
                    prev3 = m_image(prev3)
                    c_prev3 = torch.full([opt.bsz, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    z_prev3 = torch.full([opt.bsz, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    z_prev3 = m_noise(z_prev3)
                else:
                    # prev for C, prev3 = C->A, c_prev3 = C->A->C
                    prev3, c_prev3 = cycle_rec_reverse(Gs2, Gs, Zs3, reals3, NoiseAmp3, in_s3, m_noise, m_image, opt,
                                                       epoch, [1, 0])
                    # prev for B, prev2 = B->A, c_prev2 = B->A->B
                    prev2, c_prev2 = cycle_rec_reverse(Gs2, Gs, Zs2, reals2, NoiseAmp2, in_s2, m_noise, m_image, opt,
                                                       epoch, [0, 1])
                    # prev for A, prev_from3 = A->C, c_prev_from3 = A->C->A
                    prev_from3, c_prev_from3 = cycle_rec(Gs, Gs2, Zs, reals, NoiseAmp, in_s, m_noise, m_image, opt,
                                                         epoch, [1, 0])
                    # prev for A, prev_from3 = A->B, c_prev_from2 = A->B->A
                    prev_from2, c_prev_from2 = cycle_rec(Gs, Gs2, Zs, reals, NoiseAmp, in_s, m_noise, m_image, opt,
                                                         epoch, [0, 1])
                    # rec for C
                    z_prev3 = draw_concat(Gs, Zs3, reals3, NoiseAmp3, in_s3, 'rec', m_noise, m_image, opt, [1, 0])
                    z_prev3 = m_image(z_prev3)
                    # rec for B
                    z_prev2 = draw_concat(Gs, Zs2, reals2, NoiseAmp2, in_s2, 'rec', m_noise, m_image, opt, [0, 1])
                    z_prev2 = m_image(z_prev2)
                    # rec for A
                    z_prev = draw_concat_reverse(Gs2, Zs, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, opt)
                    z_prev = m_image(z_prev)
            else:
                # prev for C, prev3 = C->A, c_prev3 = C->A->C
                prev3, c_prev3 = cycle_rec_reverse(Gs2, Gs, Zs3, reals3, NoiseAmp3, in_s3, m_noise, m_image, opt, epoch, [1, 0])
                # prev for B, prev2 = B->A, c_prev2 = B->A->B
                prev2, c_prev2 = cycle_rec_reverse(Gs2, Gs, Zs2, reals2, NoiseAmp2, in_s2, m_noise, m_image, opt, epoch, [0, 1])
                # prev for A, prev_from3 = A->C, c_prev_from3 = A->C->A
                prev_from3, c_prev_from3 = cycle_rec(Gs, Gs2, Zs, reals, NoiseAmp, in_s, m_noise, m_image, opt, epoch, [1, 0])
                # prev for A, prev_from2 = A->B, c_prev_from2 = A->B->A
                prev_from2, c_prev_from2 = cycle_rec(Gs, Gs2, Zs, reals, NoiseAmp, in_s, m_noise, m_image, opt, epoch, [0, 1])

            noise = opt.noise_amp * noise_ + m_image(real)
            noise2 = opt.noise_amp2 * noise_2 + m_image(real2)
            noise3 = opt.noise_amp3 * noise_3 + m_image(real3)

            fake_B = netG(noise.detach(), prev_from2, [0, 1])  # prev_from2 = A->B
            output_B, cls_B = netD(fake_B.detach())
            errD_fake_B = output_B.mean()
            errD_fake_B.backward(retain_graph=True)
            loss_print['errD_fake_B'] = errD_fake_B.item()

            fake_C = netG(noise.detach(), prev_from3, [1, 0])  # prev_from3 = A->C
            output, cls_C = netD(fake_C.detach())
            errD_fake_C = output.mean()
            errD_fake_C.backward(retain_graph=True)
            loss_print['errD_fake_C'] = errD_fake_C.item()

            # add cls_loss for c_b1_fake
            # not added in starGAN
            # errD_cls_b1_fake =  d_loss_cls(opt, c_b1, 1)
            # errD_cls_b1_fake.backward(retain_graph=True)
            # loss_print['errD_cls'] = errD_cls_b1_fake.item()

            gradient_penalty_B = functions.calc_conditioned_gradient_penalty(netD, real2, fake_B, opt.lambda_grad,
                                                                             opt.device)
            gradient_penalty_B.backward()
            loss_print['gradient_penalty_B'] = gradient_penalty_B.item()

            gradient_penalty_C = functions.calc_conditioned_gradient_penalty(netD, real3, fake_C, opt.lambda_grad,
                                                                             opt.device)
            gradient_penalty_C.backward()
            loss_print['gradient_penalty_C'] = gradient_penalty_C.item()

            fake2_B = netG2(noise2.detach(), prev2)  # prev2 = B->A
            output2_B = netD2(fake2_B.detach())
            errD_fake2_B = output2_B.mean()
            errD_fake2_B.backward(retain_graph=True)
            loss_print['errD_fake2_B'] = errD_fake2_B.item()

            fake2_C = netG2(noise3.detach(), prev3)  # prev3 = C->A
            output2_C = netD2(fake2_C.detach())
            errD_fake2_C = output2_C.mean()
            errD_fake2_C.backward(retain_graph=True)
            loss_print['errD_fake2_C'] = errD_fake2_C.item()

            gradient_penalty2_B = functions.calc_gradient_penalty(netD2, real, fake2_B, opt.lambda_grad, opt.device)
            gradient_penalty2_B.backward()
            loss_print['gradient_penalty2_B'] = gradient_penalty2_B.item()

            gradient_penalty2_C = functions.calc_gradient_penalty(netD2, real, fake2_C, opt.lambda_grad, opt.device)
            gradient_penalty2_C.backward()
            loss_print['gradient_penalty2_C'] = gradient_penalty2_C.item()

            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for j in range(opt.Gsteps):
            loss_tv = TVLoss()
            optimizerG.zero_grad()

            output_B, cls_B = netD(fake_B)
            errG_B = -output_B.mean() + lambda_tv * loss_tv(fake_B)
            errG_B.backward(retain_graph=True)
            errG_cls_fake_B = d_loss_cls(opt, cls_B, [0, 1])
            errG_cls_fake_B.backward(retain_graph=True)
            loss_print['errG_fake_B'] = errG_B.item()
            loss_print['errG_cls_fake_B'] = errG_cls_fake_B.item()

            output_C, cls_C = netD(fake_C)
            errG_C = -output_C.mean() + lambda_tv * loss_tv(fake_C)
            errG_C.backward(retain_graph=True)
            errG_cls_fake_C = d_loss_cls(opt, cls_C, [1, 0])
            errG_cls_fake_C.backward(retain_graph=True)
            loss_print['errG_fake_C'] = errG_C.item()
            loss_print['errG_cls_fake_C'] = errG_cls_fake_C.item()

            output2_B = netD2(fake2_B)
            errG2_B = -output2_B.mean() + lambda_tv * loss_tv(fake2_B)
            errG2_B.backward(retain_graph=True)
            loss_print['errG2_fake_B'] = errG2_B.item()

            output2_C = netD2(fake2_C)
            errG2_C = -output2_C.mean() + lambda_tv * loss_tv(fake2_C)
            errG2_C.backward(retain_graph=True)
            loss_print['errG2_fake_C'] = errG2_C.item()

            loss = nn.L1Loss()
            Z_opt2 = m_image(real2)
            rec_loss_B = lambda_idt * loss(netG(Z_opt2.detach(), z_prev2, [0, 1]), real2)
            rec_loss_B.backward(retain_graph=True)
            loss_print['rec_loss_B'] = rec_loss_B.item()
            rec_loss_B = rec_loss_B.detach()

            Z_opt3 = m_image(real3)
            rec_loss_C = lambda_idt * loss(netG(Z_opt3.detach(), z_prev3, [1, 0]), real3)
            rec_loss_C.backward(retain_graph=True)
            loss_print['rec_loss_C'] = rec_loss_C.item()
            rec_loss_C = rec_loss_C.detach()

            cyc_loss_B = lambda_cyc * loss(netG(m_image(fake2_B), c_prev2, [0, 1]), real2)  # c_prev2 = B->A->B
            cyc_loss_B.backward(retain_graph=True)
            loss_print['cyc_loss_B'] = cyc_loss_B.item()
            cyc_loss_B = cyc_loss_B.detach()

            cyc_loss_C = lambda_cyc * loss(netG(m_image(fake2_C), c_prev3, [1, 0]), real3)  # c_prev3 = C->A->C
            cyc_loss_C.backward(retain_graph=True)
            loss_print['cyc_loss_C'] = cyc_loss_C.item()
            cyc_loss_C = cyc_loss_C.detach()

            Z_opt = m_image(real)
            rec_loss_A = lambda_idt * loss(netG2(Z_opt.detach(), z_prev), real)
            rec_loss_A.backward(retain_graph=True)
            loss_print['rec_loss_A'] = rec_loss_A.item()
            rec_loss_A = rec_loss_A.detach()

            cyc_loss2_B = lambda_cyc * loss(netG2(m_image(fake_B), c_prev_from2), real)  # c_prev_from2 = A->B->A
            cyc_loss2_B.backward(retain_graph=True)
            loss_print['cyc_loss2_B'] = cyc_loss2_B.item()
            cyc_loss2_B = cyc_loss2_B.detach()

            cyc_loss2_C = lambda_cyc * loss(netG2(m_image(fake_C), c_prev_from3), real)  # c_prev_from3 = A->C->A
            cyc_loss2_C.backward(retain_graph=True)
            loss_print['cyc_loss2_C'] = cyc_loss2_C.item()
            cyc_loss2_C = cyc_loss2_C.detach()

            optimizerG.step()

        if epoch % 25 == 0 or epoch == (opt.niter - 1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter - 1):
            save_image(denorm(fake_B.data.cpu()), '%s/fake_sample_B.png' % (opt.outf))
            save_image(denorm(fake_C.data.cpu()), '%s/fake_sample_C.png' % (opt.outf))
            save_image(denorm(netG2(m_image(fake_B), z_prev).data.cpu()), '%s/cyc_sample_from_B.png' % (opt.outf))
            save_image(denorm(netG2(m_image(fake_C), z_prev).data.cpu()), '%s/cyc_sample_from_C.png' % (opt.outf))
            save_image(denorm(netG2(Z_opt.detach(), z_prev).data.cpu()), '%s/rec_sample.png' % (opt.outf))
            save_image(denorm(fake2_B.data.cpu()), '%s/fake_sample2_B.png' % (opt.outf))
            save_image(denorm(fake2_C.data.cpu()), '%s/fake_sample2_C.png' % (opt.outf))
            save_image(denorm(netG(m_image(fake2_B), z_prev2, [0, 1]).data.cpu()), '%s/cyc_sample2_B.png' % (opt.outf))
            save_image(denorm(netG(m_image(fake2_C), z_prev3, [1, 0]).data.cpu()), '%s/cyc_sample2_C.png' % (opt.outf))
            save_image(denorm(netG(Z_opt2.detach(), z_prev2, [0, 1]).data.cpu()), '%s/rec_sample2_B.png' % (opt.outf))
            save_image(denorm(netG(Z_opt3.detach(), z_prev3, [1, 0]).data.cpu()), '%s/rec_sample2_C.png' % (opt.outf))
            save_image(denorm(z_opt.data.cpu()), '%s/z_opt.png' % (opt.outf))
            save_image(denorm(noise.data.cpu()), '%s/noise.png' % (opt.outf))
            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

            log = " Iteration [{}/{}]".format(epoch, opt.niter)
            for tag, value in loss_print.items():
                log += ", {}: {:.4f}".format(tag, value)
            print(log)

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG, netD, z_opt, netG2, netD2, z_opt2, z_opt3, opt)
    return z_opt, in_s, netG, z_opt2, in_s2, netG2, z_opt3, in_s3


def draw_concat(Gs, Zs, reals, NoiseAmp, in_s, mode, m_noise, m_image, opt, label):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rec':
            count = 0
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = m_image(real_curr)
                G_z = G(z_in.detach(), G_z, label)
                G_z = imresize(G_z.detach(), 1 / opt.scale_factor, opt)
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                count += 1
    return G_z


def draw_concat_reverse(Gs, Zs, reals, NoiseAmp, in_s, mode, m_noise, m_image, opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rec':
            count = 0
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = m_image(real_curr)
                G_z = G(z_in.detach(), G_z)
                G_z = imresize(G_z.detach(), 1 / opt.scale_factor, opt)
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                count += 1
    return G_z


def cycle_rec(Gs, Gs2, Zs, reals, NoiseAmp, in_s, m_noise, m_image, opt, epoch, label):
    x_ab = in_s
    x_aba = in_s
    if len(Gs) > 0:
        count = 0
        for G, G2, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Gs2, Zs, reals, reals[1:], NoiseAmp):
            z = functions.generate_noise([3, Z_opt.shape[2], Z_opt.shape[3]], device=opt.device)
            z = z.expand(opt.bsz, 3, z.shape[2], z.shape[3])
            z = m_noise(z)
            x_ab = x_ab[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
            x_ab = m_image(x_ab)
            z_in = noise_amp * z + m_image(real_curr)
            x_ab = G(z_in.detach(), x_ab, label)

            x_aba = G2(x_ab, x_aba)
            x_ab = imresize(x_ab.detach(), 1 / opt.scale_factor, opt)
            x_ab = x_ab[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
            x_aba = imresize(x_aba.detach(), 1 / opt.scale_factor, opt)
            x_aba = x_aba[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
            count += 1
    return x_ab, x_aba


def cycle_rec_reverse(Gs, Gs2, Zs, reals, NoiseAmp, in_s, m_noise, m_image, opt, epoch, label):
    x_ab = in_s
    x_aba = in_s
    if len(Gs) > 0:
        count = 0
        for G, G2, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Gs2, Zs, reals, reals[1:], NoiseAmp):
            z = functions.generate_noise([3, Z_opt.shape[2], Z_opt.shape[3]], device=opt.device)
            z = z.expand(opt.bsz, 3, z.shape[2], z.shape[3])
            z = m_noise(z)
            x_ab = x_ab[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
            x_ab = m_image(x_ab)
            z_in = noise_amp * z + m_image(real_curr)
            x_ab = G(z_in.detach(), x_ab)

            x_aba = G2(x_ab, x_aba, label)
            x_ab = imresize(x_ab.detach(), 1 / opt.scale_factor, opt)
            x_ab = x_ab[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
            x_aba = imresize(x_aba.detach(), 1 / opt.scale_factor, opt)
            x_aba = x_aba[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
            count += 1
    return x_ab, x_aba


def init_models(opt):
    # generator initialization
    netG = models.ConditionalGeneratorConcatSkip2CleanAddAlpha(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    # discriminator initialization
    netD = models.ConditionalWDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    # generator2 initialization
    netG2 = models.GeneratorConcatSkip2CleanAddAlpha(opt).to(opt.device)
    netG2.apply(models.weights_init)
    if opt.netG2 != '':
        netG2.load_state_dict(torch.load(opt.netG2))
    print(netG2)

    # discriminator2 initialization
    netD2 = models.WDiscriminator(opt).to(opt.device)
    netD2.apply(models.weights_init)
    if opt.netD2 != '':
        netD2.load_state_dict(torch.load(opt.netD2))
    print(netD2)
    return netD, netG, netD2, netG2
