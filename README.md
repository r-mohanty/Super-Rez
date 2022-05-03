# note: set allow_tf32 to False!!!!

# TODO List
gradient accumulation
print D loss, print R1
plot D loss, plot R1
save EMA_Model, save D
resume EMA_Model, resume D
use EMA_Model to visualize and test

# hyperparams to mess with
batch_size
patch_size
ema_beta: magic, don't touch!
lr: learing rate for G
d_lr: learning rate for D
betas: beta1, beta2 for the ADAM optimizer
loss: find the right mix of fidelity and regularization
r1_gamma: multiplier for R1 gradient penalty


# Super-Rez

# Credit Attribution

We used the training procedure provided by the EDSR paper [1] to train our super-resolution network (cs143model.py), which was also inspired by the EDSR network.

# Citations

[1] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, **"Enhanced Deep Residual Networks for Single Image Super-Resolution,"** <i>2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with **CVPR 2017**. </i> [[PDF](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)] [[arXiv](https://arxiv.org/abs/1707.02921)] [[Slide](https://cv.snu.ac.kr/research/EDSR/Presentation_v3(release).pptx)]