import random
import os
import torch.nn as nn
from .encdec import Encoder, Decoder
from .residual_vq import ResidualVQ
import torch.nn.functional as F
import torch
import torch
import os
from collections import defaultdict

import torch
import os
from collections import defaultdict


import torch
import os

class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        #assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)

        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)

        return code_idx, all_codes



    def forward(self, x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        x_out = self.decoder(x_quantized)

        return x_out, commit_loss, perplexity


    def fine_tune(self, idx0,data_2d):
        idx = idx0.unsqueeze(-1)
        idx = idx.long().to(next(self.parameters()).device)
        x_d_all = self.quantizer.get_codes_from_indices(idx)
        x0 = x_d_all[0]
        x = x0.permute(0, 2, 1)
        x_out = self.decoder(x)
        x_reshaped = x_out.reshape(x_out.shape[0], x_out.shape[1], 22, 3)
        x_reshaped[..., 2] = 0
        x_back = x_reshaped.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2])
        #print("x_out",x_out.shape)
        commit_loss = F.mse_loss(x_back, data_2d)
        perplexity = commit_loss
        return x_back,commit_loss,perplexity



    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        x = x_d.sum(dim=0).permute(0, 2, 1)
        x_out = self.decoder(x)


        return x_out
    def forward_decoder_firstlayer(self, idx0):
        """
        只使用 residual VQ 的第0层作为普通 VQ-VAE 来解码。
        idx0: (B, N) 或 (B, N, 1) 或 (N,) (会自动补形状)
        返回: x_out 形状与 decoder 一致 (B, input_width, T)
        """
        import torch
        # 1) 规范索引形状到 (B, N, 1)
        if idx0.dim() == 1:  # (N,)
            idx = idx0.unsqueeze(0).unsqueeze(-1)  # -> (1, N, 1)
        elif idx0.dim() == 2:  # (B, N)
            idx = idx0.unsqueeze(-1)  # -> (B, N, 1)
        elif idx0.dim() == 3:  # (B, N, 1) or (B, N, Q)
            idx = idx0[..., :1]  # 只取第0层
        else:
            raise ValueError(f"idx0 dim={idx0.dim()} 不合法")

        idx = idx.long().to(next(self.parameters()).device)

        # 2) 取出所有层对应的向量，但我们只保留第0层
        # get_codes_from_indices 产出形状通常是 (Q, B, N, D)
        x_d_all = self.quantizer.get_codes_from_indices(idx)
        x0 = x_d_all[0]  # (B, N, D) —— 仅第0层，不做残差相加

        # 3) decoder 期望 (B, D, N)
        x = x0.permute(0, 2, 1)  # (B, D, N)
        x_out = self.decoder(x)  # (B, input_width, T)
        return x_out
