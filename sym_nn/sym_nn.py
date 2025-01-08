import torch
import torch.nn as nn
import numpy as np

from equivariant_diffusion.utils import remove_mean_with_mask, assert_mean_zero_with_mask, assert_correctly_masked
from sym_nn.utils import qr, orthogonal_haar, compute_gradient_norm, GaussianLayer
from sym_nn.perceiver import SymDiffPerceiverConfig, SymDiffPerceiver, SymDiffPerceiverDecoder, TensorPreprocessor, \
                      t_emb_dim, concat_t_emb, positional_encoding, IdentityPreprocessor
from sym_nn.dit import DiT

from qm9.models import EGNN_dynamics_QM9
from sym_nn.gnn import GNNEnc
from timm.models.vision_transformer import Mlp


class DiT_GNN_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        enc_out_channels: int,
        enc_x_scale: float,
        enc_hidden_size: int,
        enc_depth: int,
        enc_num_heads: int,
        enc_mlp_ratio: float,
        use_fused_attn: bool,
        enc_x_emb: str,

        dec_hidden_features: int,

        n_dims: int = 3,

        enc_concat_h: bool = False,

        noise_dims: int = 0,
        noise_std: float = 1.0,
 
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.args = args

        self.in_node_nf = in_node_nf  # dynamics_in_node_nf (includes time)
        self.context_node_nf = context_node_nf
        self.n_dims = n_dims

        self.noise_dims = noise_dims
        self.noise_std = noise_std

        self.enc_concat_h = enc_concat_h

        if enc_concat_h:
            self.gamma_enc_input_dim = n_dims + in_node_nf + noise_dims
        else:
            self.gamma_enc_input_dim = n_dims + noise_dims

        self.gamma_enc = DiT(
            out_channels=enc_out_channels, x_scale=enc_x_scale, 
            hidden_size=enc_hidden_size, depth=enc_depth, 
            num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio, 
            use_fused_attn=use_fused_attn, x_emb=enc_x_emb, 
            input_dim=self.gamma_enc_input_dim
        ).to(device)

        self.gamma_dec = Mlp(
            in_features=enc_hidden_size, hidden_features=dec_hidden_features,
            out_features=n_dims**2
        ).to(device)

        # use GNN for k
        self.gnn_dynamics = EGNN_dynamics_QM9(
            in_node_nf=in_node_nf, context_node_nf=args.context_node_nf,
            n_dims=n_dims, device=device, hidden_nf=args.nf,
            act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
            attention=args.attention, tanh=args.tanh, mode="gnn_dynamics", norm_constant=args.norm_constant,
            inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
            normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method
        )

        self.device = device

    def forward(self):
        raise NotImplementedError

    def _forward(self, t, xh, node_mask, edge_mask, context):
        # t: [bs, 1]
        # xh: [bs, n_nodes, dims]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]

        assert context is None

        bs, n_nodes, _ = xh.shape

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]
        x = remove_mean_with_mask(x, node_mask)

        g = orthogonal_haar(dim=self.n_dims, target_tensor=x)  # [bs, 3, 3]
        g_inv_x = torch.bmm(x, g)  # as x is represented row-wise

        if self.enc_concat_h:
            g_inv_x = torch.cat([g_inv_x, h], dim=-1)
        if self.noise_dims > 0:
            g_inv_x = torch.cat([
                g_inv_x, 
                node_mask * self.noise_std * torch.randn(
                    bs, n_nodes, self.noise_dims, device=self.device
                    )
                ], dim=-1)

        # [bs, n_nodes, hidden_size]
        gamma = node_mask * self.gamma_enc(
            g_inv_x, t.squeeze(-1), node_mask.squeeze(-1), 
            use_final_layer=False
            )

        # decoded summed representation into gamma - this is S_n-invariant
        N = node_mask.sum(1)  # [bs, 1]
        gamma = torch.sum(gamma, dim=1) / N  # [bs, hidden_size] 
        # [bs, 3, 3]
        gamma = qr(
            self.gamma_dec(gamma).reshape(-1, self.n_dims, self.n_dims)
            )[0]
        gamma = torch.bmm(gamma, g.transpose(2, 1))

        # pass through k
        gamma_inv_x = torch.bmm(x, gamma)
        xh = torch.cat([gamma_inv_x, h], dim=-1)
        xh = self.gnn_dynamics._forward(t, xh, node_mask, edge_mask, context)  # [bs, n_nodes, dims] - com_free

        x = xh[:, :, :self.n_dims] 
        h = xh[:, :, self.n_dims:]

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        x = torch.bmm(x, gamma.transpose(2, 1))
        xh = torch.cat([x, h], dim=-1)

        assert_correctly_masked(xh, node_mask)

        return xh


class DiTGaussian_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        xh_hidden_size: int,
        K: int,

        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_dropout: float,

        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.args = args
        self.n_dims = n_dims

        self.gaussian_embedder = GaussianLayer(K=K)

        self.model = DiT(
            out_channels=n_dims+in_node_nf+context_node_nf, x_scale=0.0, 
            hidden_size=hidden_size, depth=depth, 
            num_heads=num_heads, mlp_ratio=mlp_ratio, 
            use_fused_attn=True, x_emb="identity", mlp_dropout=mlp_dropout
            )

        self.xh_embedder = nn.Linear(n_dims+in_node_nf+context_node_nf, xh_hidden_size)
        self.pos_embedder = nn.Linear(K, hidden_size-xh_hidden_size)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.xh_embedder.apply(_basic_init)
        self.pos_embedder.apply(_basic_init)

        self.device = device

    def forward(self):
        raise NotImplementedError

    def _forward(self, t, xh, node_mask, edge_mask, context):
        # t: [bs, 1]
        # xh: [bs, n_nodes, dims]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]

        assert context is None

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]
        x = remove_mean_with_mask(x, node_mask)
        xh = torch.cat([x.clone(), h], dim=-1)
        xh = self.xh_embedder(xh)

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb = self.gaussian_embedder(x, node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb = torch.sum(self.pos_embedder(pos_emb), dim=-2) / N  # [bs, n_nodes, K]

        xh = torch.cat([xh, pos_emb], dim=-1) * node_mask
        xh = self.model(xh, t.squeeze(-1), node_mask.squeeze(-1)) * node_mask

        x = xh[:, :, :self.n_dims] 
        h = xh[:, :, self.n_dims:]

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        xh = torch.cat([x, h], dim=-1)
        assert_correctly_masked(xh, node_mask)

        return xh


class DiT_DitGaussian_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        xh_hidden_size: int,
        K: int,

        enc_hidden_size: int,
        enc_depth: int,
        enc_num_heads: int,
        enc_mlp_ratio: float,

        dec_hidden_features: int,

        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_dropout: float,

        enc_concat_h: bool = False,

        noise_dims: int = 0,
        noise_std: float = 1.0,

        mlp_type: str = "mlp",

        n_dims: int = 3,
        device: str = "cpu",
        use_separate_gauss_embs = False,
        use_separate_dropout = False,
        dropout_gamma_enc = 0,
        dropout_gamma_dec = 0,
        dropout_k = 0,
        use_separate_K = False,
        gamma_K = 0,
        k_K = 0,
        pos_emb_gamma_projection_dim = 0,
        use_gamma_for_sampling = True,
        fix_qr = False
    ) -> None:
        super().__init__()

        self.args = args
        self.n_dims = n_dims
        self.use_separate_gauss_embs = use_separate_gauss_embs
        self.use_gamma_for_sampling = use_gamma_for_sampling
        self.fix_qr = fix_qr

        if not use_separate_gauss_embs:
            self.gaussian_embedder = GaussianLayer(K=K)
        else:
            if use_separate_K:
                self.gaussian_embedder_gamma = GaussianLayer(K=gamma_K)
                self.gaussian_embedder_k = GaussianLayer(K=k_K)
            else:
                self.gaussian_embedder_gamma = GaussianLayer(K=K)
                self.gaussian_embedder_k = GaussianLayer(K=K)

        self.xh_embedder = nn.Linear(n_dims+in_node_nf+context_node_nf, xh_hidden_size)

        if not self.use_separate_gauss_embs:
            self.pos_embedder = nn.Linear(K, hidden_size-xh_hidden_size)
        else:
            if pos_emb_gamma_projection_dim != 0:
                self.pos_embedder_gamma = nn.Linear(K, pos_emb_gamma_projection_dim)
            else:
                self.pos_embedder_gamma = nn.Linear(K, hidden_size-xh_hidden_size)
            self.pos_embedder_k = nn.Linear(K, hidden_size-xh_hidden_size)


        self.noise_dims = noise_dims
        self.noise_std = noise_std

        self.enc_concat_h = enc_concat_h

        if enc_concat_h:
            self.gamma_enc_input_dim = n_dims + in_node_nf + hidden_size-xh_hidden_size + noise_dims
        else:
            self.gamma_enc_input_dim = n_dims + hidden_size-xh_hidden_size + noise_dims

        # Using separate dropout for gamma, and K
        if not use_separate_dropout:
            # enc_out_channels not used 
            self.gamma_enc = DiT(
                out_channels=0, x_scale=0.0, 
                hidden_size=enc_hidden_size, depth=enc_depth, 
                num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio, 
                use_fused_attn=True, x_emb="linear", 
                input_dim=self.gamma_enc_input_dim,
                mlp_type=mlp_type
            ).to(device)

            # add t emb here
            self.gamma_dec = Mlp(
                in_features=enc_hidden_size, hidden_features=dec_hidden_features,
                out_features=n_dims**2
            ).to(device)

            self.k = DiT(
                out_channels=n_dims+in_node_nf+context_node_nf, x_scale=0.0, 
                hidden_size=hidden_size, depth=depth, 
                num_heads=num_heads, mlp_ratio=mlp_ratio,
                mlp_dropout=mlp_dropout, 
                use_fused_attn=True, x_emb="identity",
                mlp_type=mlp_type
                )

        else:
            # enc_out_channels not used 
            self.gamma_enc = DiT(
                out_channels=0, x_scale=0.0, 
                hidden_size=enc_hidden_size, depth=enc_depth, 
                num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio, 
                use_fused_attn=True, x_emb="linear", 
                input_dim=self.gamma_enc_input_dim,
                mlp_type=mlp_type, mlp_dropout=dropout_gamma_enc
            ).to(device)

            # add t emb here
            self.gamma_dec = Mlp(
                in_features=enc_hidden_size, hidden_features=dec_hidden_features,
                out_features=n_dims**2, drop=dropout_gamma_dec
            ).to(device)

            self.k = DiT(
                out_channels=n_dims+in_node_nf+context_node_nf, x_scale=0.0, 
                hidden_size=hidden_size, depth=depth, 
                num_heads=num_heads, mlp_ratio=mlp_ratio,
                mlp_dropout=dropout_k,
                use_fused_attn=True, x_emb="identity",
                mlp_type=mlp_type
                )            

        self.device = device


        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.xh_embedder.apply(_basic_init)
        self.pos_embedder.apply(_basic_init)
        self.gamma_dec.apply(_basic_init)

    def forward(self):
        raise NotImplementedError

    def _forward(self, t, xh, node_mask, edge_mask, context, gamma=None, return_gamma=False):
        # t: [bs, 1]
        # xh: [bs, n_nodes, dims]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]

        assert context is None

        bs, n_nodes, _ = xh.shape

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]

        x = remove_mean_with_mask(x, node_mask)
        if not self.args.com_free:
            assert gamma is not None
            # Karras et al. (2022) scaling
            x = x / torch.sqrt(
                self.args.sigma_data**2 + 
                torch.exp(gamma[1]).unsqueeze(-1)
                )

        g = orthogonal_haar(dim=self.n_dims, target_tensor=x)  # [bs, 3, 3]

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        if not self.use_separate_gauss_embs:
            pos_emb = self.gaussian_embedder(x, node_mask)  # [bs, n_nodes, n_nodes, K]
            pos_emb = torch.sum(self.pos_embedder(pos_emb), dim=-2) / N  # [bs, n_nodes, hidden_size-xh_hidden_size]
        else:
            pos_emb_gamma = self.gaussian_embedder_gamma(x, node_mask)  # [bs, n_nodes, n_nodes, K]
            pos_emb_gamma = torch.sum(self.pos_embedder_gamma(pos_emb_gamma), dim=-2) / N  # [bs, n_nodes, hidden_size-xh_hidden_size]

            pos_emb_k = self.gaussian_embedder_k(x, node_mask)  # [bs, n_nodes, n_nodes, K]
            pos_emb_k = torch.sum(self.pos_embedder_k(pos_emb_k), dim=-2) / N  # [bs, n_nodes, hidden_size-xh_hidden_size]

        g_inv_x = torch.bmm(x.clone(), g.clone())  # as x is represented row-wise

        if self.enc_concat_h:
            g_inv_x = torch.cat([g_inv_x, h.clone()], dim=-1)
        if self.noise_dims > 0:
            g_inv_x = torch.cat([
                g_inv_x, 
                node_mask * self.noise_std * torch.randn(
                    bs, n_nodes, self.noise_dims, device=self.device
                    )
                ], dim=-1)

        if not self.use_separate_gauss_embs:
            g_inv_x = torch.cat([g_inv_x, pos_emb.clone()], dim=-1)
        else:
            g_inv_x = torch.cat([g_inv_x, pos_emb_gamma.clone()], dim=-1)

        # [bs, n_nodes, hidden_size]
        gamma = node_mask * self.gamma_enc(
            g_inv_x, t.squeeze(-1), node_mask.squeeze(-1), 
            use_final_layer=False
            )

        # decoded summed representation into gamma - this is S_n-invariant
        gamma = torch.sum(gamma, dim=1) / N.squeeze(-1)  # [bs, hidden_size] 
        # [bs, 3, 3]
        if self.fix_qr:
            gamma = qr(
                (self.gamma_dec(gamma).reshape(-1, self.n_dims, self.n_dims)).transpose(1, 2)
                )[0].transpose(1, 2)            
        else:
            gamma = qr(
                self.gamma_dec(gamma).reshape(-1, self.n_dims, self.n_dims) 
                )[0]
        gamma = torch.bmm(gamma, g.transpose(2, 1))

        # Using gamma sampling 
        if self.use_gamma_for_sampling:
            gamma_inv_x = torch.bmm(x, gamma.clone())
        else:
            gamma_inv_x = x

        xh = self.xh_embedder(torch.cat([gamma_inv_x, h], dim=-1))

        if not self.use_separate_gauss_embs:
            xh = torch.cat([xh, pos_emb], dim=-1) * node_mask
        else:
            xh = torch.cat([xh, pos_emb_k], dim=-1) * node_mask

        xh = self.k(xh, t.squeeze(-1), node_mask.squeeze(-1)) * node_mask  # use DiT
        x = xh[:, :, :self.n_dims] 
        h = xh[:, :, self.n_dims:]

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        # Whether to use gammas at sampling or not
        if self.use_gamma_for_sampling:
            print("Using gamma.")
            x = torch.bmm(x, gamma.transpose(2, 1))
            
        xh = torch.cat([x, h], dim=-1)
        assert_correctly_masked(xh, node_mask)
    
        # Output
        if return_gamma:
            return xh, gamma
        return xh

    def print_parameter_count(self):

        xh_embedder_params = sum(p.numel() for p in self.xh_embedder.parameters() if p.requires_grad)
        if not self.use_separate_gauss_embs:            
            pos_embedder_params = sum(p.numel() for p in self.pos_embedder.parameters() if p.requires_grad)
        else:
            pos_embedder_params = 0
            pos_embedder_params += sum(p.numel() for p in self.pos_embedder_gamma.parameters() if p.requires_grad)
            pos_embedder_params += sum(p.numel() for p in self.pos_embedder_k.parameters() if p.requires_grad)
        embedder_params = xh_embedder_params + pos_embedder_params
        gamma_enc_params = sum(p.numel() for p in self.gamma_enc.parameters() if p.requires_grad)
        gamma_dec_params = sum(p.numel() for p in self.gamma_dec.parameters() if p.requires_grad)
        gamma_params = gamma_enc_params + gamma_dec_params
        k_params = sum(p.numel() for p in self.k.parameters() if p.requires_grad)

        print(f"embedder_params: {embedder_params}; gamma_params: {gamma_params}; k_params: {k_params}")


class DiTOnlyGaussian_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        h_hidden_size: int,
        K: int,

        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_dropout: float,

        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.args = args
        self.n_dims = n_dims

        self.gaussian_embedder = GaussianLayer(K=K)

        self.model = DiT(
            out_channels=n_dims+in_node_nf+context_node_nf, x_scale=0.0, 
            hidden_size=hidden_size, depth=depth, 
            num_heads=num_heads, mlp_ratio=mlp_ratio, 
            use_fused_attn=True, x_emb="identity", mlp_dropout=mlp_dropout
            )

        self.h_embedder = nn.Linear(in_node_nf+context_node_nf, h_hidden_size)
        self.pos_embedder = nn.Linear(K, hidden_size-h_hidden_size)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.h_embedder.apply(_basic_init)
        self.pos_embedder.apply(_basic_init)

        self.device = device

    def forward(self):
        raise NotImplementedError

    def _forward(self, t, xh, node_mask, edge_mask, context):
        # t: [bs, 1]
        # xh: [bs, n_nodes, dims]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]

        assert context is None

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]
        x = remove_mean_with_mask(x, node_mask)
        h = self.h_embedder(h)

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb = self.gaussian_embedder(x, node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb = torch.sum(self.pos_embedder(pos_emb), dim=-2) / N  # [bs, n_nodes, K]

        h_pos_emb = torch.cat([h, pos_emb], dim=-1) * node_mask
        xh = self.model(h_pos_emb, t.squeeze(-1), node_mask.squeeze(-1)) * node_mask

        x = xh[:, :, :self.n_dims] 
        h = xh[:, :, self.n_dims:]

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        xh = torch.cat([x, h], dim=-1)
        assert_correctly_masked(xh, node_mask)

        return xh


if __name__ == "__main__":
    pass