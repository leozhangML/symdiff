import torch
import torch.nn as nn
import numpy as np

from equivariant_diffusion.utils import remove_mean_with_mask, assert_mean_zero_with_mask, assert_correctly_masked
from sym_nn.utils import qr, orthogonal_haar, compute_gradient_norm, GaussianLayer
from sym_nn.dit import DiT, TimestepEmbedder

from qm9.models import EGNN_dynamics_QM9
from timm.models.vision_transformer import Mlp


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

        mlp_type: str = "mlp",

        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.args = args
        self.molecule = args.molecule

        self.n_dims = n_dims
        self.in_node_nf = in_node_nf if self.molecule else 0

        self.mlp_type = mlp_type

        self.gaussian_embedder = GaussianLayer(K=K)

        self.model = DiT(
            out_channels=n_dims+self.in_node_nf+context_node_nf, x_scale=0.0, 
            hidden_size=hidden_size, depth=depth, 
            num_heads=num_heads, mlp_ratio=mlp_ratio, 
            use_fused_attn=True, x_emb="identity", mlp_dropout=mlp_dropout,
            mlp_type=mlp_type
            )

        self.xh_embedder = nn.Linear(n_dims+self.in_node_nf+context_node_nf, xh_hidden_size)
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

    def ve_scaling(self, gamma_t):
        c_in = torch.sqrt(
            self.args.sigma_data**2 +
            torch.exp(gamma_t[1]).unsqueeze(-1)
        )
        return c_in

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

        bs, n_nodes, h_dims = h.shape

        if self.molecule:
            xh = torch.cat([x.clone(), h], dim=-1)
        else:
            xh = x.clone()

        xh = self.xh_embedder(xh)  # [bs, n_nodes, xh_hidden_size]

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb = self.gaussian_embedder(x, node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb = torch.sum(self.pos_embedder(pos_emb), dim=-2) / N  # [bs, n_nodes, hidden_size-xh_hidden_size]

        xh = node_mask * torch.cat([xh, pos_emb], dim=-1)  # [bs, n_nodes, hidden_size]
        xh = node_mask * self.model(xh, t.squeeze(-1), node_mask.squeeze(-1))

        x = xh[:, :, :self.n_dims] 
        h = xh[:, :, self.n_dims:] if self.molecule else torch.zeros(bs, n_nodes, h_dims, device=h.device)

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        xh = torch.cat([x, h], dim=-1)
        assert_correctly_masked(xh, node_mask)

        return xh

    def print_parameter_count(self):

        xh_embedder_params = sum(p.numel() for p in self.xh_embedder.parameters() if p.requires_grad)
        pos_embedder_params = sum(p.numel() for p in self.pos_embedder.parameters() if p.requires_grad)
        gaussian_embedder_params = sum(p.numel() for p in self.gaussian_embedder.parameters() if p.requires_grad)
        embedder_params = xh_embedder_params + pos_embedder_params + gaussian_embedder_params

        model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"embedder_params: {embedder_params}; model_params: {model_params}")


class ScalarsDiT_PE_no_recursion_DitGaussian_dynamics(nn.Module):
    """
    Add positional embeddings to the gamma backbone and
    remove recursion
    """
    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,
        xh_hidden_size: int,
        K: int,
        # base gamma
        base_enc_hidden_size: int,
        base_enc_depth: int,
        base_enc_num_heads: int,
        base_enc_mlp_ratio: int,
        gamma_mlp_dropout: float,
        # k
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_dropout: float,
        noise_dims: int = 0,
        noise_std: float = 1.0,
        mlp_type: str = "mlp",
        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:
        super().__init__()
        self.args = args
        self.molecule = args.molecule
        self.n_dims = n_dims
        self.in_node_nf = in_node_nf if self.molecule else 0
        self.device = device
        # base gamma
        self.base_gaussian_embedder = GaussianLayer(K)
        self.base_gamma_projection = nn.Linear(K, base_enc_hidden_size-noise_dims)
        self.base_gamma_enc = DiT(
            out_channels=n_dims, x_scale=0.0,
            hidden_size=base_enc_hidden_size, depth=base_enc_depth,
            num_heads=base_enc_num_heads, mlp_ratio=base_enc_mlp_ratio,
            mlp_dropout=gamma_mlp_dropout,
            use_fused_attn=True, x_emb="identity",
            input_dim=base_enc_hidden_size,
            mlp_type=mlp_type,
            zero_final=False
        ).to(device)
        # k
        self.gaussian_embedder = GaussianLayer(K)
        self.pos_embedder = nn.Linear(K, hidden_size-xh_hidden_size)
        self.xh_embedder = nn.Linear(n_dims+self.in_node_nf+context_node_nf, xh_hidden_size)
        self.backbone = DiT(
            out_channels=n_dims+self.in_node_nf+context_node_nf, x_scale=0.0,
            hidden_size=hidden_size, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio,
            use_fused_attn=True, x_emb="identity", mlp_dropout=mlp_dropout,
            mlp_type=mlp_type
            )
        # noise params
        self.noise_dims = noise_dims
        self.noise_std = noise_std
        # init linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.base_gamma_projection.apply(_basic_init)
        self.pos_embedder.apply(_basic_init)
        self.xh_embedder.apply(_basic_init)
    def forward(self):
        raise NotImplementedError


    def gamma(self, t, x, node_mask):
        """THIS FUNCTION IS THE SAME AS base_gamma IN ScalarsDiT_PE_DitGaussian_dynamics
        BUT WE CORRECT THE QR HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
        bs, n_nodes, _ = x.shape
        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        xx_t = self.base_gaussian_embedder(x, node_mask)  # [bs, n_nodes, n_nodes, K]
        f_xx_t = torch.sum(self.base_gamma_projection(xx_t), dim=-2) / N  # [bs, n_nodes, base_enc_hidden_size-noise_dims]
        if self.noise_dims > 0:
            # [bs, n_nodes, base_enc_hidden_size]
            f_xx_t = torch.cat([
                node_mask * f_xx_t,
                node_mask * self.noise_std * torch.randn(
                    bs, n_nodes, self.noise_dims, device=self.device)
            ], dim=-1)
        # [bs, n_nodes, n_dims]
        gamma = node_mask * self.base_gamma_enc(
            f_xx_t, t.squeeze(-1), node_mask.squeeze(-1))
        gamma = torch.bmm(gamma.transpose(1, 2), x)  # [bs, 3, 3]
        gamma = qr(gamma.transpose(1, 2))[0].transpose(1, 2)
        return gamma
    

    def k(self, t, x, h, node_mask):
        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb = self.gaussian_embedder(x.clone(), node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb = torch.sum(self.pos_embedder(pos_emb), dim=-2) / N  # [bs, n_nodes, hidden_size-xh_hidden_size]
        if self.molecule:
            xh = self.xh_embedder(torch.cat([x, h], dim=-1))
            xh = node_mask * torch.cat([xh, pos_emb], dim=-1)
            xh = node_mask * self.backbone(xh, t.squeeze(-1), node_mask.squeeze(-1))
        else:
            bs, n_nodes, h_dims = h.shape
            x_ = self.xh_embedder(x)
            x_ = node_mask * torch.cat([x_, pos_emb], dim=-1)
            xh = node_mask * torch.cat(
                [self.backbone(x_, t.squeeze(-1), node_mask.squeeze(-1)),
                 torch.zeros(bs, n_nodes, h_dims, device=h.device)],
                 dim=-1
            )
        return xh
    

    def ve_scaling(self, gamma_t):
        c_in = torch.sqrt(
            self.args.sigma_data**2 +
            torch.exp(gamma_t[1]).unsqueeze(-1)
        )
        return c_in
    def _forward(self, t, xh, node_mask, edge_mask, context, gamma_t=None, return_gamma=False):
        # t: [bs, 1]
        # xh: [bs, n_nodes, dims]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]
        assert context is None
        #assert gamma_t is None
        bs, n_nodes, _ = xh.shape
        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]
        x = remove_mean_with_mask(x, node_mask)
        if not self.args.com_free:
            assert gamma_t is not None
            # Karras et al. (2022) scaling
            c_in = self.ve_scaling(gamma_t)
            x = x / c_in
            h = h / c_in
        gamma = self.gamma(t, x, node_mask)
        gamma_inv_x = torch.bmm(x, gamma.clone())
        xh = self.k(t, gamma_inv_x, h, node_mask)
        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]
        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U
        x = torch.bmm(x, gamma.transpose(2, 1))
        xh = torch.cat([x, h], dim=-1)
        assert_correctly_masked(xh, node_mask)
        if return_gamma:
            return xh, gamma
        else:
            return xh


class DiT_DitGaussian_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        xh_hidden_size: int,
        K: int,
        pos_embedder_test: int,

        enc_hidden_size: int,
        enc_depth: int,
        enc_num_heads: int,
        enc_mlp_ratio: float,

        dec_hidden_features: int,
        gamma_mlp_dropout: float, 

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
        use_gamma_for_sampling: bool = True,

        fix_qr: bool = False
    ) -> None:
        super().__init__()

        self.args = args
        self.molecule = args.molecule
        self.use_gamma_for_sampling = use_gamma_for_sampling

        self.n_dims = n_dims
        self.in_node_nf = in_node_nf if self.molecule else 0

        self.gaussian_embedder = GaussianLayer(K=K)
        self.gaussian_embedder_test = GaussianLayer(K=K)

        self.xh_embedder = nn.Linear(n_dims+self.in_node_nf+context_node_nf, xh_hidden_size)
        self.pos_embedder = nn.Linear(K, hidden_size-xh_hidden_size)

        self.pos_embedder_test = nn.Linear(K, pos_embedder_test)

        self.noise_dims = noise_dims
        self.noise_std = noise_std

        self.enc_concat_h = enc_concat_h  # NOTE: not used for now

        self.fix_qr = fix_qr

        if enc_concat_h:
            self.gamma_enc_input_dim = n_dims + self.in_node_nf + hidden_size-xh_hidden_size + noise_dims
        else:
            self.gamma_enc_input_dim = n_dims + pos_embedder_test + noise_dims  # positional embeddings dominate the input

        # enc_out_channels not used 
        self.gamma_enc = DiT(
            out_channels=0, x_scale=0.0, 
            hidden_size=enc_hidden_size, depth=enc_depth, 
            num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio, 
            mlp_dropout=gamma_mlp_dropout,
            use_fused_attn=True, x_emb="linear", 
            input_dim=self.gamma_enc_input_dim,
            mlp_type=mlp_type
        ).to(device)

        # add t emb here?
        self.gamma_dec = Mlp(
            in_features=enc_hidden_size, hidden_features=dec_hidden_features,
            out_features=n_dims**2, drop=0.0
        ).to(device)

        self.backbone = DiT(
            out_channels=n_dims+self.in_node_nf+context_node_nf, x_scale=0.0, 
            hidden_size=hidden_size, depth=depth, 
            num_heads=num_heads, mlp_ratio=mlp_ratio, 
            use_fused_attn=True, x_emb="identity", mlp_dropout=mlp_dropout,
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

    def gamma(self, t, x, node_mask):
    
        bs, n_nodes, _ = x.shape

        g = orthogonal_haar(dim=self.n_dims, target_tensor=x)  # [bs, 3, 3]

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb_test = self.gaussian_embedder_test(x.clone(), node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb_test = torch.sum(self.pos_embedder_test(pos_emb_test), dim=-2) / N  # [bs, n_nodes, pos_embedder_test]

        g_inv_x = torch.bmm(x.clone(), g.clone())  # as x is represented row-wise

        if self.noise_dims > 0:
            g_inv_x = torch.cat([
                g_inv_x, 
                node_mask * self.noise_std * torch.randn(
                    bs, n_nodes, self.noise_dims, device=self.device
                    )
                ], dim=-1)

        g_inv_x = torch.cat([g_inv_x, pos_emb_test.clone()], dim=-1)

        # [bs, n_nodes, hidden_size]
        gamma = node_mask * self.gamma_enc(
            g_inv_x, t.squeeze(-1), node_mask.squeeze(-1), 
            use_final_layer=False
            )

        # decoded summed representation into gamma - this is S_n-invariant
        gamma = torch.sum(gamma, dim=1) / N.squeeze(-1)  # [bs, hidden_size] 
        # [bs, 3, 3]
        if self.fix_qr:
            gamma = qr(self.gamma_dec(gamma).reshape(-1, self.n_dims, self.n_dims).transpose(1, 2))[0].transpose(1, 2)
        else:
            gamma = qr(
                self.gamma_dec(gamma).reshape(-1, self.n_dims, self.n_dims)
                )[0]

        gamma = torch.bmm(gamma, g.transpose(2, 1))

        return gamma

    def k(self, t, x, h, node_mask):

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb = self.gaussian_embedder(x.clone(), node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb = torch.sum(self.pos_embedder(pos_emb), dim=-2) / N  # [bs, n_nodes, hidden_size-xh_hidden_size]

        if self.molecule:
            xh = self.xh_embedder(torch.cat([x, h], dim=-1))
            xh = node_mask * torch.cat([xh, pos_emb], dim=-1)
            xh = node_mask * self.backbone(xh, t.squeeze(-1), node_mask.squeeze(-1))
        else:
            bs, n_nodes, h_dims = h.shape
            x_ = self.xh_embedder(x)
            x_ = node_mask * torch.cat([x_, pos_emb], dim=-1)
            xh = node_mask * torch.cat(
                [self.backbone(x_, t.squeeze(-1), node_mask.squeeze(-1)),
                 torch.zeros(bs, n_nodes, h_dims, device=h.device)],
                 dim=-1
            )

        return xh
    
    def ve_scaling(self, gamma_t):
        c_in = torch.sqrt(
            self.args.sigma_data**2 +
            torch.exp(gamma_t[1]).unsqueeze(-1)
        )
        return c_in

    def _forward(self, t, xh, node_mask, edge_mask, context, gamma_t=None, return_gamma=False): 
        # t: [bs, 1]
        # xh: [bs, n_nodes, dims]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]

        assert context is None
        #assert gamma_t is None

        bs, n_nodes, _ = xh.shape

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]

        x = remove_mean_with_mask(x, node_mask)

        if not self.args.com_free:
            assert gamma_t is not None
            # Karras et al. (2022) scaling
            c_in = self.ve_scaling(gamma_t)
            x = x / c_in
            h = h / c_in

        gamma = self.gamma(t, x, node_mask)

        if self.use_gamma_for_sampling:
            gamma_inv_x = torch.bmm(x, gamma.clone())
        else:
            gamma_inv_x = x
        xh = self.k(t, gamma_inv_x, h, node_mask)

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:] 

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        if self.use_gamma_for_sampling:
            x = torch.bmm(x, gamma.transpose(2, 1))
        xh = torch.cat([x, h], dim=-1)

        assert_correctly_masked(xh, node_mask)
        
        if return_gamma:
            return xh, gamma
        else:
            return xh

    def print_parameter_count(self):

        xh_embedder_params = sum(p.numel() for p in self.xh_embedder.parameters() if p.requires_grad)
        pos_embedder_params = sum(p.numel() for p in self.pos_embedder.parameters() if p.requires_grad)
        pos_embedder_test_params = sum(p.numel() for p in self.pos_embedder_test.parameters() if p.requires_grad)
        gaussian_embedder_params = sum(p.numel() for p in self.gaussian_embedder.parameters() if p.requires_grad)
        gaussian_embedder_test_params = sum(p.numel() for p in self.gaussian_embedder_test.parameters() if p.requires_grad)
        embedder_params = xh_embedder_params + pos_embedder_params + pos_embedder_test_params + gaussian_embedder_params + gaussian_embedder_test_params

        gamma_enc_params = sum(p.numel() for p in self.gamma_enc.parameters() if p.requires_grad)
        gamma_dec_params = sum(p.numel() for p in self.gamma_dec.parameters() if p.requires_grad)
        gamma_params = gamma_enc_params + gamma_dec_params
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

        print(f"embedder_params: {embedder_params}; gamma_params: {gamma_params}; backbone_params: {backbone_params}")


class DiTModPE_DitGaussian_dynamics(nn.Module):

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
        gamma_mlp_dropout: float, 

        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_dropout: float,

        noise_dims: int = 0,
        noise_std: float = 1.0,

        mlp_type: str = "mlp",

        n_dims: int = 3,
        device: str = "cpu",
        use_gamma_for_sampling: bool = True
    ) -> None:
        super().__init__()

        self.args = args
        self.molecule = args.molecule
        self.use_gamma_for_sampling = use_gamma_for_sampling

        self.n_dims = n_dims
        self.in_node_nf = in_node_nf if self.molecule else 0

        self.gaussian_embedder = GaussianLayer(K=K)
        #self.gaussian_embedder_test = GaussianLayer(K=K)

        self.xh_embedder = nn.Linear(n_dims+self.in_node_nf+context_node_nf, xh_hidden_size)
        self.pos_embedder = nn.Linear(K, hidden_size-xh_hidden_size)

        # self.pos_embedder_test = nn.Linear(K, pos_embedder_test)

        self.noise_dims = noise_dims
        self.noise_std = noise_std

        #self.gamma_enc_input_dim = n_dims + pos_embedder_test + noise_dims  # positional embeddings dominate the input
        
        self.gamma_input_layer = nn.Linear(n_dims+noise_dims, enc_hidden_size)
        self.gamma_enc_input_dim = enc_hidden_size  # positional embeddings dominate the input

        # enc_out_channels not used 
        self.gamma_enc = DiT(
            out_channels=0, x_scale=0.0, 
            hidden_size=enc_hidden_size, depth=enc_depth, 
            num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio, 
            mlp_dropout=gamma_mlp_dropout,
            use_fused_attn=True, x_emb="identity", 
            input_dim=self.gamma_enc_input_dim,
            mlp_type=mlp_type
        ).to(device)

        # add t emb here?
        self.gamma_dec = Mlp(
            in_features=enc_hidden_size, hidden_features=dec_hidden_features,
            out_features=n_dims**2, drop=0.0
        ).to(device)

        self.backbone = DiT(
            out_channels=n_dims+self.in_node_nf+context_node_nf, x_scale=0.0, 
            hidden_size=hidden_size, depth=depth, 
            num_heads=num_heads, mlp_ratio=mlp_ratio, 
            use_fused_attn=True, x_emb="identity", mlp_dropout=mlp_dropout,
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

        self.gamma_input_layer.apply(_basic_init)

    def forward(self):
        raise NotImplementedError

    def gamma(self, t, x, node_mask):
    
        bs, n_nodes, _ = x.shape

        g = orthogonal_haar(dim=self.n_dims, target_tensor=x)  # [bs, 3, 3]

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        #pos_emb_test = self.gaussian_embedder_test(x.clone(), node_mask)  # [bs, n_nodes, n_nodes, K]
        #pos_emb_test = torch.sum(self.pos_embedder_test(pos_emb_test), dim=-2) / N  # [bs, n_nodes, pos_embedder_test]

        g_inv_x = torch.bmm(x.clone(), g.clone())  # as x is represented row-wise

        if self.noise_dims > 0:
            g_inv_x = torch.cat([
                g_inv_x, 
                node_mask * self.noise_std * torch.randn(
                    bs, n_nodes, self.noise_dims, device=self.device
                    )
                ], dim=-1)

        #g_inv_x = torch.cat([g_inv_x, pos_emb_test.clone()], dim=-1)
        g_inv_x = node_mask * self.gamma_input_layer(g_inv_x)  # project to hidden size

        # [bs, n_nodes, hidden_size]
        gamma = node_mask * self.gamma_enc(
            g_inv_x, t.squeeze(-1), node_mask.squeeze(-1), 
            use_final_layer=False
            )

        # decoded summed representation into gamma - this is S_n-invariant
        gamma = torch.sum(gamma, dim=1) / N.squeeze(-1)  # [bs, hidden_size] 
        # [bs, 3, 3]
        gamma = qr(
            self.gamma_dec(gamma).reshape(-1, self.n_dims, self.n_dims)
            )[0]
        gamma = torch.bmm(gamma, g.transpose(2, 1))

        return gamma

    def k(self, t, x, h, node_mask):

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb = self.gaussian_embedder(x.clone(), node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb = torch.sum(self.pos_embedder(pos_emb), dim=-2) / N  # [bs, n_nodes, hidden_size-xh_hidden_size]

        if self.molecule:
            xh = self.xh_embedder(torch.cat([x, h], dim=-1))
            xh = node_mask * torch.cat([xh, pos_emb], dim=-1)
            xh = node_mask * self.backbone(xh, t.squeeze(-1), node_mask.squeeze(-1))
        else:
            bs, n_nodes, h_dims = h.shape
            x_ = self.xh_embedder(x)
            x_ = node_mask * torch.cat([x_, pos_emb], dim=-1)
            xh = node_mask * torch.cat(
                [self.backbone(x_, t.squeeze(-1), node_mask.squeeze(-1)),
                 torch.zeros(bs, n_nodes, h_dims, device=h.device)],
                 dim=-1
            )

        return xh
    
    def ve_scaling(self, gamma_t):
        c_in = torch.sqrt(
            self.args.sigma_data**2 +
            torch.exp(gamma_t[1]).unsqueeze(-1)
        )
        return c_in

    def _forward(self, t, xh, node_mask, edge_mask, context, gamma_t=None, return_gamma=False): 
        # t: [bs, 1]
        # xh: [bs, n_nodes, dims]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]

        assert context is None
        #assert gamma_t is None

        bs, n_nodes, _ = xh.shape

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]

        x = remove_mean_with_mask(x, node_mask)

        if not self.args.com_free:
            assert gamma_t is not None
            # Karras et al. (2022) scaling
            c_in = self.ve_scaling(gamma_t)
            x = x / c_in
            h = h / c_in

        gamma = self.gamma(t, x, node_mask)

        if self.use_gamma_for_sampling:
            gamma_inv_x = torch.bmm(x, gamma.clone())
        else:
            gamma_inv_x = x
        xh = self.k(t, gamma_inv_x, h, node_mask)

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:] 

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        if self.use_gamma_for_sampling:
            x = torch.bmm(x, gamma.transpose(2, 1))
        xh = torch.cat([x, h], dim=-1)

        assert_correctly_masked(xh, node_mask)
        
        if return_gamma:
            return xh, gamma
        else:
            return xh

    def print_parameter_count(self):

        xh_embedder_params = sum(p.numel() for p in self.xh_embedder.parameters() if p.requires_grad)
        pos_embedder_params = sum(p.numel() for p in self.pos_embedder.parameters() if p.requires_grad)
        pos_embedder_test_params = sum(p.numel() for p in self.pos_embedder_test.parameters() if p.requires_grad)
        gaussian_embedder_params = sum(p.numel() for p in self.gaussian_embedder.parameters() if p.requires_grad)
        gaussian_embedder_test_params = sum(p.numel() for p in self.gaussian_embedder_test.parameters() if p.requires_grad)
        embedder_params = xh_embedder_params + pos_embedder_params + pos_embedder_test_params + gaussian_embedder_params + gaussian_embedder_test_params

        gamma_enc_params = sum(p.numel() for p in self.gamma_enc.parameters() if p.requires_grad)
        gamma_dec_params = sum(p.numel() for p in self.gamma_dec.parameters() if p.requires_grad)
        gamma_params = gamma_enc_params + gamma_dec_params
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

        print(f"embedder_params: {embedder_params}; gamma_params: {gamma_params}; backbone_params: {backbone_params}")


class DiTFinal_DitGaussian_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        xh_hidden_size: int,
        K: int,
        pos_embedder_test: int,

        enc_hidden_size: int,
        enc_depth: int,
        enc_num_heads: int,
        enc_mlp_ratio: float,

        gamma_mlp_dropout: float, 

        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_dropout: float,

        noise_dims: int = 0,
        noise_std: float = 1.0,

        mlp_type: str = "mlp",

        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.args = args
        self.molecule = args.molecule

        self.n_dims = n_dims
        self.in_node_nf = in_node_nf if self.molecule else 0

        self.gaussian_embedder = GaussianLayer(K=K)
        self.gaussian_embedder_test = GaussianLayer(K=K)

        self.xh_embedder = nn.Linear(n_dims+self.in_node_nf+context_node_nf, xh_hidden_size)
        self.pos_embedder = nn.Linear(K, hidden_size-xh_hidden_size)

        self.pos_embedder_test = nn.Linear(K, pos_embedder_test)

        self.noise_dims = noise_dims
        self.noise_std = noise_std

        self.gamma_enc_input_dim = n_dims + pos_embedder_test + noise_dims

        self.gamma_enc = DiT(
            out_channels=n_dims**2, x_scale=0.0, 
            hidden_size=enc_hidden_size, depth=enc_depth, 
            num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio, 
            mlp_dropout=gamma_mlp_dropout,
            use_fused_attn=True, x_emb="linear", 
            input_dim=self.gamma_enc_input_dim,
            mlp_type=mlp_type,
            zero_final=False
        ).to(device)

        self.backbone = DiT(
            out_channels=n_dims+self.in_node_nf+context_node_nf, x_scale=0.0, 
            hidden_size=hidden_size, depth=depth, 
            num_heads=num_heads, mlp_ratio=mlp_ratio, 
            use_fused_attn=True, x_emb="identity", mlp_dropout=mlp_dropout,
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

    def forward(self):
        raise NotImplementedError

    def gamma(self, t, x, node_mask):
    
        bs, n_nodes, _ = x.shape

        g = orthogonal_haar(dim=self.n_dims, target_tensor=x)  # [bs, 3, 3]

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb_test = self.gaussian_embedder_test(x.clone(), node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb_test = torch.sum(self.pos_embedder_test(pos_emb_test), dim=-2) / N  # [bs, n_nodes, pos_embedder_test]

        g_inv_x = torch.bmm(x.clone(), g.clone())  # as x is represented row-wise

        if self.noise_dims > 0:
            g_inv_x = torch.cat([
                g_inv_x, 
                node_mask * self.noise_std * torch.randn(
                    bs, n_nodes, self.noise_dims, device=self.device
                    )
                ], dim=-1)

        g_inv_x = torch.cat([g_inv_x, pos_emb_test.clone()], dim=-1)

        # [bs, n_nodes, n_dims**2]
        gamma = node_mask * self.gamma_enc(
            g_inv_x, t.squeeze(-1), node_mask.squeeze(-1),
            )

        # decoded summed representation into gamma - this is S_n-invariant
        gamma = torch.sum(gamma, dim=1) / N.squeeze(-1)  # [bs, hidden_size]

        gamma = qr(gamma.reshape(-1, self.n_dims, self.n_dims))[0]
        gamma = torch.bmm(gamma, g.transpose(2, 1))

        return gamma

    def k(self, t, x, h, node_mask):

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb = self.gaussian_embedder(x.clone(), node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb = torch.sum(self.pos_embedder(pos_emb), dim=-2) / N  # [bs, n_nodes, hidden_size-xh_hidden_size]

        if self.molecule:
            xh = self.xh_embedder(torch.cat([x, h], dim=-1))
            xh = node_mask * torch.cat([xh, pos_emb], dim=-1)
            xh = node_mask * self.backbone(xh, t.squeeze(-1), node_mask.squeeze(-1))
        else:
            bs, n_nodes, h_dims = h.shape
            x_ = self.xh_embedder(x)
            x_ = node_mask * torch.cat([x_, pos_emb], dim=-1)
            xh = node_mask * torch.cat(
                [self.backbone(x_, t.squeeze(-1), node_mask.squeeze(-1)),
                 torch.zeros(bs, n_nodes, h_dims, device=h.device)],
                 dim=-1
            )

        return xh
    
    def ve_scaling(self, gamma_t):
        c_in = torch.sqrt(
            self.args.sigma_data**2 +
            torch.exp(gamma_t[1]).unsqueeze(-1)
        )
        return c_in

    def _forward(self, t, xh, node_mask, edge_mask, context, gamma_t=None, return_gamma=False): 
        # t: [bs, 1]
        # xh: [bs, n_nodes, dims]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]

        assert context is None
        #assert gamma_t is None

        bs, n_nodes, _ = xh.shape

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]

        x = remove_mean_with_mask(x, node_mask)

        if not self.args.com_free:
            assert gamma_t is not None
            # Karras et al. (2022) scaling
            c_in = self.ve_scaling(gamma_t)
            x = x / c_in
            h = h / c_in

        gamma = self.gamma(t, x, node_mask)

        gamma_inv_x = torch.bmm(x, gamma.clone())
        xh = self.k(t, gamma_inv_x, h, node_mask)

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:] 

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        x = torch.bmm(x, gamma.transpose(2, 1))
        xh = torch.cat([x, h], dim=-1)

        assert_correctly_masked(xh, node_mask)
        
        if return_gamma:
            return xh, gamma
        else:
            return xh

    def print_parameter_count(self):

        xh_embedder_params = sum(p.numel() for p in self.xh_embedder.parameters() if p.requires_grad)
        pos_embedder_params = sum(p.numel() for p in self.pos_embedder.parameters() if p.requires_grad)
        pos_embedder_test_params = sum(p.numel() for p in self.pos_embedder_test.parameters() if p.requires_grad)
        gaussian_embedder_params = sum(p.numel() for p in self.gaussian_embedder.parameters() if p.requires_grad)
        gaussian_embedder_test_params = sum(p.numel() for p in self.gaussian_embedder_test.parameters() if p.requires_grad)
        embedder_params = xh_embedder_params + pos_embedder_params + pos_embedder_test_params + gaussian_embedder_params + gaussian_embedder_test_params

        gamma_enc_params = sum(p.numel() for p in self.gamma_enc.parameters() if p.requires_grad)
        gamma_dec_params = sum(p.numel() for p in self.gamma_dec.parameters() if p.requires_grad)
        gamma_params = gamma_enc_params + gamma_dec_params
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

        print(f"embedder_params: {embedder_params}; gamma_params: {gamma_params}; backbone_params: {backbone_params}")


class DiT_DitGaussian_add_noise_dynamics(nn.Module):

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
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.args = args
        self.n_dims = n_dims

        self.gaussian_embedder = GaussianLayer(K=K)

        self.gaussian_embedder_test = GaussianLayer(K=K)

        self.xh_embedder = nn.Linear(n_dims+in_node_nf+context_node_nf, xh_hidden_size)
        self.pos_embedder = nn.Linear(K, hidden_size-xh_hidden_size)

        self.pos_embedder_test = nn.Linear(K, 16)

        self.noise_dims = noise_dims
        self.noise_std = noise_std

        self.enc_concat_h = enc_concat_h

        if enc_concat_h:
            self.gamma_enc_input_dim = n_dims + in_node_nf + hidden_size-xh_hidden_size + noise_dims
        else:
            #self.gamma_enc_input_dim = n_dims + hidden_size-xh_hidden_size + noise_dims
            self.gamma_enc_input_dim = n_dims + 16

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

    def _forward(self, t, xh, node_mask, edge_mask, context, gamma=None):
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
        pos_emb = self.gaussian_embedder(x.clone(), node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb = torch.sum(self.pos_embedder(pos_emb), dim=-2) / N  # [bs, n_nodes, hidden_size-xh_hidden_size]

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb_test = self.gaussian_embedder_test(x.clone(), node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb_test = torch.sum(self.pos_embedder_test(pos_emb_test), dim=-2) / N  # [bs, n_nodes, hidden_size-xh_hidden_size]

        g_inv_x = torch.bmm(x.clone(), g.clone())  # as x is represented row-wise

        if self.enc_concat_h:
            g_inv_x = torch.cat([g_inv_x, h.clone()], dim=-1)
        if self.noise_dims > 0:
            """
            g_inv_x = torch.cat([
                g_inv_x, 
                node_mask * self.noise_std * torch.randn(
                    bs, n_nodes, self.noise_dims, device=self.device
                    )
                ], dim=-1)
            """

            g_inv_x = g_inv_x + node_mask * self.noise_std * torch.randn(
                    bs, n_nodes, self.n_dims, device=self.device
                    )

        #g_inv_x = torch.cat([g_inv_x, pos_emb.clone()], dim=-1)
        g_inv_x = torch.cat([g_inv_x, pos_emb_test.clone()], dim=-1)

        # [bs, n_nodes, hidden_size]
        gamma = node_mask * self.gamma_enc(
            g_inv_x, t.squeeze(-1), node_mask.squeeze(-1), 
            use_final_layer=False
            )

        # decoded summed representation into gamma - this is S_n-invariant
        gamma = torch.sum(gamma, dim=1) / N.squeeze(-1)  # [bs, hidden_size] 
        # [bs, 3, 3]
        gamma = qr(
            self.gamma_dec(gamma).reshape(-1, self.n_dims, self.n_dims)
            )[0]
        gamma = torch.bmm(gamma, g.transpose(2, 1))

        gamma_inv_x = torch.bmm(x, gamma.clone())
        xh = self.xh_embedder(torch.cat([gamma_inv_x, h], dim=-1))
        xh = torch.cat([xh, pos_emb], dim=-1) * node_mask
        xh = self.k(xh, t.squeeze(-1), node_mask.squeeze(-1)) * node_mask  # use DiT

        x = xh[:, :, :self.n_dims] 
        h = xh[:, :, self.n_dims:]

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        x = torch.bmm(x, gamma.transpose(2, 1))
        xh = torch.cat([x, h], dim=-1)

        assert_correctly_masked(xh, node_mask)

        return xh

    def print_parameter_count(self):

        xh_embedder_params = sum(p.numel() for p in self.xh_embedder.parameters() if p.requires_grad)
        pos_embedder_params = sum(p.numel() for p in self.pos_embedder.parameters() if p.requires_grad)
        gaussian_embedder_params = sum(p.numel() for p in self.gaussian_embedder.parameters() if p.requires_grad)
        embedder_params = xh_embedder_params + pos_embedder_params + gaussian_embedder_params

        gamma_enc_params = sum(p.numel() for p in self.gamma_enc.parameters() if p.requires_grad)
        gamma_dec_params = sum(p.numel() for p in self.gamma_dec.parameters() if p.requires_grad)
        gamma_params = gamma_enc_params + gamma_dec_params
        
        k_params = sum(p.numel() for p in self.k.parameters() if p.requires_grad)

        print(f"embedder_params: {embedder_params}; gamma_params: {gamma_params}; k_params: {k_params}")


class DeepSets_DitGaussian_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        xh_hidden_size: int,
        pos_emb_gamma_size: int,
        K: int,
        t_hidden_size: int,

        enc_hidden_size: int,
        dec_hidden_features: int,

        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_dropout: float,

        noise_dims: int = 0,
        noise_std: float = 1.0,

        mlp_type: str = "mlp",

        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.args = args
        self.n_dims = n_dims

        self.gaussian_embedder = GaussianLayer(K=K)

        self.xh_embedder = nn.Linear(n_dims+in_node_nf+context_node_nf, xh_hidden_size)
        self.pos_embedder_gamma = nn.Linear(K, pos_emb_gamma_size)
        self.t_embedder = TimestepEmbedder(t_hidden_size)
        self.pos_embedder_k = nn.Linear(K, hidden_size-xh_hidden_size)

        self.noise_dims = noise_dims
        self.noise_std = noise_std

        self.gamma_enc_input_dim = n_dims + pos_emb_gamma_size + noise_dims + t_hidden_size

        self.gamma_enc = Mlp(
            in_features=self.gamma_enc_input_dim, hidden_features=enc_hidden_size,
            out_features=enc_hidden_size
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

        self.device = device

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.xh_embedder.apply(_basic_init)
        self.pos_embedder_gamma.apply(_basic_init)
        self.pos_embedder_k.apply(_basic_init)
        self.gamma_enc.apply(_basic_init)
        self.gamma_dec.apply(_basic_init)

    def forward(self):
        raise NotImplementedError

    def _forward(self, t, xh, node_mask, edge_mask, context, gamma=None):
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

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb = self.gaussian_embedder(x, node_mask)  # [bs, n_nodes, n_nodes, K]

        t_emb = node_mask * self.t_embedder(t.squeeze(-1)).unsqueeze(1).repeat_interleave(n_nodes, dim=1)  # [bs, n_nodes, t_emb_size]
        pos_emb_gamma = torch.sum(self.pos_embedder_gamma(pos_emb.clone()), dim=-2) / N  # [bs, n_nodes, pos_emb_gamma_size]

        g_inv_x = torch.bmm(x.clone(), g.clone())  # as x is represented row-wise

        if self.noise_dims > 0:
            g_inv_x = torch.cat([
                g_inv_x, 
                node_mask * self.noise_std * torch.randn(
                    bs, n_nodes, self.noise_dims, device=self.device
                    )
                ], dim=-1)

        g_inv_x = torch.cat([g_inv_x, pos_emb_gamma, t_emb], dim=-1)

        print("t_emb", t_emb.shape)
        print("g_inv_x", g_inv_x.shape)

        # [bs, n_nodes, enc_hidden_size]
        gamma = node_mask * self.gamma_enc(g_inv_x)

        # decoded summed representation into gamma - this is S_n-invariant
        gamma = torch.sum(gamma, dim=1) / N.squeeze(-1)  # [bs, enc_hidden_size] 
        # [bs, 3, 3]
        gamma = qr(
            self.gamma_dec(gamma).reshape(-1, self.n_dims, self.n_dims)
            )[0]
        gamma = torch.bmm(gamma, g.transpose(2, 1))

        gamma_inv_x = torch.bmm(x, gamma.clone())
        xh = self.xh_embedder(torch.cat([gamma_inv_x, h], dim=-1))
        pos_emb_k = torch.sum(self.pos_embedder_k(pos_emb.clone()), dim=-2) / N   # [bs, n_nodes, hidden_size-xh_hidden_size]

        xh = torch.cat([xh, pos_emb_k], dim=-1) * node_mask
        xh = self.k(xh, t.squeeze(-1), node_mask.squeeze(-1)) * node_mask  # use DiT

        x = xh[:, :, :self.n_dims] 
        h = xh[:, :, self.n_dims:]

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        x = torch.bmm(x, gamma.transpose(2, 1))
        xh = torch.cat([x, h], dim=-1)

        assert_correctly_masked(xh, node_mask)

        return xh

    def print_parameter_count(self):

        xh_embedder_params = sum(p.numel() for p in self.xh_embedder.parameters() if p.requires_grad)
        pos_embedder_gamma_params = sum(p.numel() for p in self.pos_embedder_gamma.parameters() if p.requires_grad)
        pos_embedder_k_params = sum(p.numel() for p in self.pos_embedder_k.parameters() if p.requires_grad)
        gaussian_embedder_params = sum(p.numel() for p in self.gaussian_embedder.parameters() if p.requires_grad)
        embedder_params = xh_embedder_params + pos_embedder_gamma_params + pos_embedder_k_params + gaussian_embedder_params

        gamma_enc_params = sum(p.numel() for p in self.gamma_enc.parameters() if p.requires_grad)
        gamma_dec_params = sum(p.numel() for p in self.gamma_dec.parameters() if p.requires_grad)
        gamma_params = gamma_enc_params + gamma_dec_params
 
        k_params = sum(p.numel() for p in self.k.parameters() if p.requires_grad)

        print(f"embedder_params: {embedder_params}; gamma_params: {gamma_params}; k_params: {k_params}")


class Scalars_DitGaussian_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        xh_hidden_size: int,
        K: int,
        t_hidden_size: int,

        # gamma_1
        pos_emb_gamma_1_size: int,
        gamma_1_hidden_size: int,

        # gamma_0
        enc_hidden_size: int,
        dec_hidden_features: int,

        # k
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_dropout: float,

        noise_dims: int = 0,
        noise_std: float = 1.0,

        mlp_type: str = "mlp",

        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.args = args
        self.n_dims = n_dims

        # for gamma_1
        self.gaussian_embedder_1 = GaussianLayer(K=K)
        self.pos_embedder_gamma_1 = nn.Linear(K, pos_emb_gamma_1_size)
        self.t_embedder = TimestepEmbedder(t_hidden_size)

        # for k
        self.gaussian_embedder_k = GaussianLayer(K=K)
        self.pos_embedder_k = nn.Linear(K, hidden_size-xh_hidden_size)
        self.xh_embedder = nn.Linear(n_dims+in_node_nf+context_node_nf, xh_hidden_size)

        # for time embedding in gamma_0, gamma_1
        self.t_embedder = TimestepEmbedder(t_hidden_size)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # gamma noise
        self.noise_dims = noise_dims
        self.noise_std = noise_std

        # gamma_1 components
        self.gamma_1_mlp = Mlp(
            in_features=pos_emb_gamma_1_size+t_hidden_size+noise_dims, 
            hidden_features=gamma_1_hidden_size,
            out_features=n_dims
        )

        # gamma_0 components
        self.gamma_0_enc_mlp = Mlp(
            in_features=n_dims+noise_dims+t_hidden_size, 
            hidden_features=enc_hidden_size,
            out_features=enc_hidden_size
        ).to(device)

        self.gamma_0_dec_mlp = Mlp(
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

        self.device = device

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.xh_embedder.apply(_basic_init)
        self.pos_embedder_gamma_1.apply(_basic_init)
        self.pos_embedder_k.apply(_basic_init)
        self.gamma_1_mlp.apply(_basic_init)
        self.gamma_0_enc_mlp.apply(_basic_init)
        self.gamma_0_dec_mlp.apply(_basic_init)

    def forward(self):
        raise NotImplementedError

    def gamma_1(self, t_emb, x, node_mask):
        bs, n_nodes, _ = x.shape
        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]

        xx_t = self.gaussian_embedder_1(x, node_mask)  # [bs, n_nodes, n_nodes, K]
        f_xx_t = torch.sum(self.pos_embedder_gamma_1(xx_t), dim=-2) / N  # [bs, n_nodes, pos_emb_gamma_1_size]

        if self.noise_dims > 0:
            f_xx_t = torch.cat([
                f_xx_t,
                node_mask * self.noise_std * torch.randn(
                    bs, n_nodes, self.noise_dims, device=self.device)
            ], dim=-1)

        f_xx_t = torch.cat([f_xx_t, t_emb], dim=-1)  # [bs, n_nodes, pos_emb_gamma_1_size+t_hidden_size+noise_dims] 
        f_xx_t = node_mask * self.gamma_1_mlp(f_xx_t)  # [bs, n_nodes, 3]

        gamma_1 = qr(torch.bmm(f_xx_t.transpose(1, 2), x))[0]  # [bs, 3, 3]

        return gamma_1

    def gamma_0(self, t_emb, x, node_mask):
        bs, n_nodes, _ = x.shape
        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]

        if self.noise_dims > 0:
            gamma_x = torch.cat([
                x, 
                node_mask * self.noise_std * torch.randn(
                    bs, n_nodes, self.noise_dims, device=self.device)
                ], dim=-1)

        gamma_x = torch.cat([gamma_x, t_emb], dim=-1)  # [bs, n_nodes, 3+t_hidden_size+noise_dims]

        # [bs, n_nodes, enc_hidden_size]
        gamma_x = node_mask * self.gamma_0_enc_mlp(gamma_x)

        # decoded summed representation into gamma - this is S_n-invariant
        gamma_x = torch.sum(gamma_x, dim=1) / N.squeeze(-1)  # [bs, enc_hidden_size] 
        # [bs, 3, 3]
        gamma_0 = qr(
            self.gamma_0_dec_mlp(gamma_x).reshape(-1, self.n_dims, self.n_dims)
            )[0]
        
        return gamma_0

    def _forward(self, t, xh, node_mask, edge_mask, context, gamma=None):
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

        # [bs, n_nodes, t_emb_size]
        t_emb = node_mask * self.t_embedder(
            t.squeeze(-1)
            ).unsqueeze(1).repeat_interleave(n_nodes, dim=1) 

        # [bs, 3, 3]
        gamma_1 = self.gamma_1(t_emb.clone(), x.clone(), node_mask)
        gamma = self.gamma_0(
            t_emb.clone(),
            torch.bmm(x.clone(), gamma_1.clone()),
            node_mask
            )
        gamma = torch.bmm(gamma, gamma_1.transpose(2, 1))

        # compute pos_emb for k
        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb = self.gaussian_embedder_k(x, node_mask)  # [bs, n_nodes, n_nodes, K]        
        pos_emb = torch.sum(self.pos_embedder_k(pos_emb), dim=-2) / N  # [bs, n_nodes, hidden_size-xh_hidden_size]

        gamma_inv_x = torch.bmm(x.clone(), gamma.clone())
        xh = self.xh_embedder(torch.cat([gamma_inv_x, h], dim=-1))
        xh = node_mask * torch.cat([xh, pos_emb], dim=-1)
        xh = node_mask * self.k(xh, t.squeeze(-1), node_mask.squeeze(-1))

        x = xh[:, :, :self.n_dims] 
        h = xh[:, :, self.n_dims:]

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        x = torch.bmm(x, gamma.transpose(2, 1))
        xh = torch.cat([x, h], dim=-1)

        assert_correctly_masked(xh, node_mask)

        return xh

    def print_parameter_count(self):

        xh_embedder_params = sum(p.numel() for p in self.xh_embedder.parameters() if p.requires_grad)
        t_embedder_params = sum(p.numel() for p in self.t_embedder.parameters() if p.requires_grad) 
        pos_embedder_gamma_1_params = sum(p.numel() for p in self.pos_embedder_gamma_1.parameters() if p.requires_grad)
        pos_embedder_k_params = sum(p.numel() for p in self.pos_embedder_k.parameters() if p.requires_grad)
        gaussian_embedder_1_params = sum(p.numel() for p in self.gaussian_embedder_1.parameters() if p.requires_grad)
        gaussian_embedder_k_params = sum(p.numel() for p in self.gaussian_embedder_k.parameters() if p.requires_grad)
        embedder_params = xh_embedder_params + t_embedder_params + pos_embedder_gamma_1_params + pos_embedder_k_params + gaussian_embedder_k_params + gaussian_embedder_1_params

        gamma_1_mlp_params = sum(p.numel() for p in self.gamma_1_mlp.parameters() if p.requires_grad)
        gamma_0_enc_mlp_params = sum(p.numel() for p in self.gamma_0_enc_mlp.parameters() if p.requires_grad)
        gamma_0_dec_mlp_params = sum(p.numel() for p in self.gamma_0_dec_mlp.parameters() if p.requires_grad)
        gamma_params = gamma_1_mlp_params + gamma_0_enc_mlp_params + gamma_0_dec_mlp_params
 
        k_params = sum(p.numel() for p in self.k.parameters() if p.requires_grad)

        print(f"embedder_params: {embedder_params}; gamma_params: {gamma_params}; k_params: {k_params}")


class ScalarsDiT_DitGaussian_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        xh_hidden_size: int,
        K: int,

        # base gamma
        base_enc_hidden_size: int,
        base_enc_depth: int,
        base_enc_num_heads: int,
        base_enc_mlp_ratio: int,

        # backbone gamma
        enc_hidden_size: int,
        enc_depth: int,
        enc_num_heads: int,
        enc_mlp_ratio: float,

        dec_hidden_features: int,
        gamma_mlp_dropout: float, 

        # k
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_dropout: float,

        noise_dims: int = 0,
        noise_std: float = 1.0,

        mlp_type: str = "mlp",

        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.args = args
        self.molecule = args.molecule

        self.n_dims = n_dims
        self.in_node_nf = in_node_nf if self.molecule else 0
        self.device = device
    
        # base gamma
        self.base_gaussian_embedder = GaussianLayer(K)
        self.base_gamma_projection = nn.Linear(K, base_enc_hidden_size-noise_dims)

        self.base_gamma_enc = DiT(
            out_channels=n_dims, x_scale=0.0, 
            hidden_size=base_enc_hidden_size, depth=base_enc_depth, 
            num_heads=base_enc_num_heads, mlp_ratio=base_enc_mlp_ratio, 
            mlp_dropout=gamma_mlp_dropout,
            use_fused_attn=True, x_emb="linear", 
            input_dim=base_enc_hidden_size,
            mlp_type=mlp_type,
            zero_final=False
        ).to(device)

        # backbone gamma
        self.gamma_projection = nn.Linear(n_dims, enc_hidden_size-noise_dims)

        self.gamma_enc = DiT(
            out_channels=0, x_scale=0.0, 
            hidden_size=enc_hidden_size, depth=enc_depth, 
            num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio, 
            mlp_dropout=gamma_mlp_dropout,
            use_fused_attn=True, x_emb="linear", 
            input_dim=enc_hidden_size,
            mlp_type=mlp_type
        ).to(device)

        self.gamma_dec = Mlp(
            in_features=enc_hidden_size, hidden_features=dec_hidden_features,
            out_features=n_dims**2, drop=0.0
        ).to(device)

        # k
        self.gaussian_embedder = GaussianLayer(K=K)
        self.pos_embedder = nn.Linear(K, hidden_size-xh_hidden_size)
        self.xh_embedder = nn.Linear(n_dims+self.in_node_nf+context_node_nf, xh_hidden_size)

        self.backbone = DiT(
            out_channels=n_dims+self.in_node_nf+context_node_nf, x_scale=0.0, 
            hidden_size=hidden_size, depth=depth, 
            num_heads=num_heads, mlp_ratio=mlp_ratio, 
            use_fused_attn=True, x_emb="identity", mlp_dropout=mlp_dropout,
            mlp_type=mlp_type
            )

        # noise params
        self.noise_dims = noise_dims
        self.noise_std = noise_std

        # init linear layers
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

    def base_gamma(self, t, x, node_mask):
        """THIS FUNCTION IS THE SAME AS base_gamma IN ScalarsDiT_PE_DitGaussian_dynamics
        BUT WE CORRECT THE QR HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
        bs, n_nodes, _ = x.shape
        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        xx_t = self.base_gaussian_embedder(x, node_mask)  # [bs, n_nodes, n_nodes, K]
        f_xx_t = torch.sum(self.base_gamma_projection(xx_t), dim=-2) / N  # [bs, n_nodes, base_enc_hidden_size-noise_dims]
        if self.noise_dims > 0:
            # [bs, n_nodes, base_enc_hidden_size]
            f_xx_t = torch.cat([
                node_mask * f_xx_t,
                node_mask * self.noise_std * torch.randn(
                    bs, n_nodes, self.noise_dims, device=self.device)
            ], dim=-1)
        # [bs, n_nodes, n_dims]
        gamma = node_mask * self.base_gamma_enc(
            f_xx_t, t.squeeze(-1), node_mask.squeeze(-1))
        gamma = torch.bmm(gamma.transpose(1, 2), x)  # [bs, 3, 3]
        gamma = qr(gamma.transpose(1, 2))[0].transpose(1, 2)
        return gamma

    def gamma(self, t, x, node_mask):
    
        bs, n_nodes, _ = x.shape

        base_gamma = self.base_gamma(t, x, node_mask)  # [bs, 3, 3]

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]

        g_inv_x = torch.bmm(x.clone(), base_gamma.clone())
        g_inv_x = node_mask * self.gamma_projection(g_inv_x)  # [bs, n_nodes, enc_hidden_size-noise_dims]

        if self.noise_dims > 0:
            # [bs, n_nodes, enc_hidden_size]
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
        gamma = torch.sum(gamma, dim=1) / N.squeeze(-1)  # [bs, hidden_size]

        # [bs, 3, 3]
        gamma = qr(
            self.gamma_dec(gamma).reshape(-1, self.n_dims, self.n_dims)
            )[0]

        gamma = torch.bmm(gamma, base_gamma.transpose(2, 1))

        return gamma

    def k(self, t, x, h, node_mask):

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb = self.gaussian_embedder(x.clone(), node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb = torch.sum(self.pos_embedder(pos_emb), dim=-2) / N  # [bs, n_nodes, hidden_size-xh_hidden_size]

        if self.molecule:
            xh = self.xh_embedder(torch.cat([x, h], dim=-1))
            xh = node_mask * torch.cat([xh, pos_emb], dim=-1)
            xh = node_mask * self.backbone(xh, t.squeeze(-1), node_mask.squeeze(-1))
        else:
            bs, n_nodes, h_dims = h.shape
            x_ = self.xh_embedder(x)
            x_ = node_mask * torch.cat([x_, pos_emb], dim=-1)
            xh = node_mask * torch.cat(
                [self.backbone(x_, t.squeeze(-1), node_mask.squeeze(-1)),
                 torch.zeros(bs, n_nodes, h_dims, device=h.device)],
                 dim=-1
            )

        return xh
    
    def ve_scaling(self, gamma_t):
        c_in = torch.sqrt(
            self.args.sigma_data**2 +
            torch.exp(gamma_t[1]).unsqueeze(-1)
        )
        return c_in

    def _forward(self, t, xh, node_mask, edge_mask, context, gamma_t=None, return_gamma=False): 
        # t: [bs, 1]
        # xh: [bs, n_nodes, dims]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]

        assert context is None
        #assert gamma_t is None

        bs, n_nodes, _ = xh.shape

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]

        x = remove_mean_with_mask(x, node_mask)

        if not self.args.com_free:
            assert gamma_t is not None
            # Karras et al. (2022) scaling
            c_in = self.ve_scaling(gamma_t)
            x = x / c_in
            h = h / c_in

        gamma = self.gamma(t, x, node_mask)

        gamma_inv_x = torch.bmm(x, gamma.clone())
        xh = self.k(t, gamma_inv_x, h, node_mask)

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:] 

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        x = torch.bmm(x, gamma.transpose(2, 1))
        xh = torch.cat([x, h], dim=-1)

        assert_correctly_masked(xh, node_mask)
        
        if return_gamma:
            return xh, gamma
        else:
            return xh

    def print_parameter_count(self):

        xh_embedder_params = sum(p.numel() for p in self.xh_embedder.parameters() if p.requires_grad)
        pos_embedder_params = sum(p.numel() for p in self.pos_embedder.parameters() if p.requires_grad)
        # pos_embedder_test_params = sum(p.numel() for p in self.pos_embedder_test.parameters() if p.requires_grad)
        gaussian_embedder_params = sum(p.numel() for p in self.gaussian_embedder.parameters() if p.requires_grad)
        # gaussian_embedder_test_params = sum(p.numel() for p in self.gaussian_embedder_test.parameters() if p.requires_grad)
        embedder_params = xh_embedder_params + pos_embedder_params + gaussian_embedder_params

        gamma_enc_params = sum(p.numel() for p in self.gamma_enc.parameters() if p.requires_grad)
        gamma_dec_params = sum(p.numel() for p in self.gamma_dec.parameters() if p.requires_grad)
        gamma_params = gamma_enc_params + gamma_dec_params
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

        print(f"embedder_params: {embedder_params}; gamma_params: {gamma_params}; backbone_params: {backbone_params}")


class ScalarsDiT_PE_DitGaussian_dynamics(nn.Module):

    """Add positional embeddings to the gamma backbone"""

    def __init__(
        self,
        args,
        in_node_nf: int,   # CHECK
        context_node_nf: int,   

        xh_hidden_size: int,
        K: int,

        xh_gamma_hidden_size: int,

        # base gamma
        base_enc_hidden_size: int,  
        base_enc_depth: int,   
        base_enc_num_heads: int,   
        base_enc_mlp_ratio: int,   

        # backbone gamma
        enc_hidden_size: int,
        enc_depth: int,
        enc_num_heads: int,
        enc_mlp_ratio: float,

        dec_hidden_features: int,
        gamma_mlp_dropout: float, 

        # k
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_dropout: float,

        noise_dims: int = 0,
        noise_std: float = 1.0,

        mlp_type: str = "mlp",

        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.args = args
        self.molecule = args.molecule

        self.n_dims = n_dims
        self.in_node_nf = in_node_nf if self.molecule else 0
        self.device = device
    
        # base gamma
        self.base_gaussian_embedder = GaussianLayer(K)
        self.base_gamma_projection = nn.Linear(K, base_enc_hidden_size-noise_dims)

        self.base_gamma_enc = DiT(
            out_channels=n_dims, x_scale=0.0, 
            hidden_size=base_enc_hidden_size, depth=base_enc_depth, 
            num_heads=base_enc_num_heads, mlp_ratio=base_enc_mlp_ratio, 
            mlp_dropout=gamma_mlp_dropout,
            use_fused_attn=True, x_emb="linear", 
            input_dim=base_enc_hidden_size,
            mlp_type=mlp_type,
            zero_final=False
        ).to(device)

        # backbone gamma
        self.gamma_gaussian_embedder = GaussianLayer(K)
        self.gamma_pos_embedder = nn.Linear(K, enc_hidden_size-xh_gamma_hidden_size)
        self.xh_gamma_embedder = nn.Linear(n_dims+noise_dims, xh_gamma_hidden_size)

        self.gamma_enc = DiT(
            out_channels=0, x_scale=0.0, 
            hidden_size=enc_hidden_size, depth=enc_depth, 
            num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio, 
            mlp_dropout=gamma_mlp_dropout,
            use_fused_attn=True, x_emb="identity", 
            input_dim=enc_hidden_size,
            mlp_type=mlp_type
        ).to(device)

        self.gamma_dec = Mlp(
            in_features=enc_hidden_size, hidden_features=dec_hidden_features,
            out_features=n_dims**2, drop=0.0
        ).to(device)

        # k
        self.gaussian_embedder = GaussianLayer(K)
        self.pos_embedder = nn.Linear(K, hidden_size-xh_hidden_size)
        self.xh_embedder = nn.Linear(n_dims+self.in_node_nf+context_node_nf, xh_hidden_size)

        self.backbone = DiT(
            out_channels=n_dims+self.in_node_nf+context_node_nf, x_scale=0.0, 
            hidden_size=hidden_size, depth=depth, 
            num_heads=num_heads, mlp_ratio=mlp_ratio, 
            use_fused_attn=True, x_emb="identity", mlp_dropout=mlp_dropout,
            mlp_type=mlp_type
            )

        # noise params
        self.noise_dims = noise_dims
        self.noise_std = noise_std

        # init linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.base_gamma_projection.apply(_basic_init)

        self.gamma_pos_embedder.apply(_basic_init)
        self.xh_gamma_embedder.apply(_basic_init)
        self.gamma_dec.apply(_basic_init)

        self.pos_embedder.apply(_basic_init)
        self.xh_embedder.apply(_basic_init)

    def forward(self):
        raise NotImplementedError

    def gamma(self, t, x, node_mask):
    
        bs, n_nodes, _ = x.shape

        base_gamma = self.base_gamma(t, x, node_mask)  # [bs, 3, 3]

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        gamma_pos_emb = self.gamma_gaussian_embedder(x.clone(), node_mask)  # [bs, n_nodes, n_nodes, K]
        gamma_pos_emb = torch.sum(self.gamma_pos_embedder(gamma_pos_emb), dim=-2) / N  # [bs, n_nodes, enc_hidden_size-xh_gamma_hidden_size]

        g_inv_x = torch.bmm(x.clone(), base_gamma.clone())

        if self.noise_dims > 0:
            # [bs, n_nodes, enc_hidden_size]
            g_inv_x = torch.cat([
                g_inv_x, 
                node_mask * self.noise_std * torch.randn(
                    bs, n_nodes, self.noise_dims, device=self.device
                    )
                ], dim=-1)

        g_inv_x = self.xh_gamma_embedder(g_inv_x)
        g_inv_x = torch.cat([g_inv_x, gamma_pos_emb.clone()], dim=-1)

        # [bs, n_nodes, hidden_size]
        gamma = node_mask * self.gamma_enc(
            g_inv_x, t.squeeze(-1), node_mask.squeeze(-1), 
            use_final_layer=False
            )

        # decoded summed representation into gamma - this is S_n-invariant
        gamma = torch.sum(gamma, dim=1) / N.squeeze(-1)  # [bs, hidden_size]

        # [bs, 3, 3]
        gamma = qr(
            self.gamma_dec(gamma).reshape(-1, self.n_dims, self.n_dims)
            )[0]

        gamma = torch.bmm(gamma, base_gamma.transpose(2, 1))

        return gamma

    def base_gamma(self, t, x, node_mask):
        """THIS FUNCTION IS THE SAME AS base_gamma IN ScalarsDiT_PE_DitGaussian_dynamics
        BUT WE CORRECT THE QR HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
        bs, n_nodes, _ = x.shape
        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        xx_t = self.base_gaussian_embedder(x, node_mask)  # [bs, n_nodes, n_nodes, K]
        f_xx_t = torch.sum(self.base_gamma_projection(xx_t), dim=-2) / N  # [bs, n_nodes, base_enc_hidden_size-noise_dims]
        if self.noise_dims > 0:
            # [bs, n_nodes, base_enc_hidden_size]
            f_xx_t = torch.cat([
                node_mask * f_xx_t,
                node_mask * self.noise_std * torch.randn(
                    bs, n_nodes, self.noise_dims, device=self.device)
            ], dim=-1)
        # [bs, n_nodes, n_dims]
        gamma = node_mask * self.base_gamma_enc(
            f_xx_t, t.squeeze(-1), node_mask.squeeze(-1))
        gamma = torch.bmm(gamma.transpose(1, 2), x)  # [bs, 3, 3]
        gamma = qr(gamma.transpose(1, 2))[0].transpose(1, 2)
        return gamma

    def k(self, t, x, h, node_mask):

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb = self.gaussian_embedder(x.clone(), node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb = torch.sum(self.pos_embedder(pos_emb), dim=-2) / N  # [bs, n_nodes, hidden_size-xh_hidden_size]

        if self.molecule:
            xh = self.xh_embedder(torch.cat([x, h], dim=-1))
            xh = node_mask * torch.cat([xh, pos_emb], dim=-1)
            xh = node_mask * self.backbone(xh, t.squeeze(-1), node_mask.squeeze(-1))
        else:
            bs, n_nodes, h_dims = h.shape
            x_ = self.xh_embedder(x)
            x_ = node_mask * torch.cat([x_, pos_emb], dim=-1)
            xh = node_mask * torch.cat(
                [self.backbone(x_, t.squeeze(-1), node_mask.squeeze(-1)),
                 torch.zeros(bs, n_nodes, h_dims, device=h.device)],
                 dim=-1
            )

        return xh
    
    def ve_scaling(self, gamma_t):
        c_in = torch.sqrt(
            self.args.sigma_data**2 +
            torch.exp(gamma_t[1]).unsqueeze(-1)
        )
        return c_in

    def _forward(self, t, xh, node_mask, edge_mask, context, gamma_t=None, return_gamma=False): 
        # t: [bs, 1]
        # xh: [bs, n_nodes, dims]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]

        assert context is None
        #assert gamma_t is None

        bs, n_nodes, _ = xh.shape

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]

        x = remove_mean_with_mask(x, node_mask)

        if not self.args.com_free:
            assert gamma_t is not None
            # Karras et al. (2022) scaling
            c_in = self.ve_scaling(gamma_t)
            x = x / c_in
            h = h / c_in

        gamma = self.gamma(t, x, node_mask)

        gamma_inv_x = torch.bmm(x, gamma.clone())
        xh = self.k(t, gamma_inv_x, h, node_mask)

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:] 

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        x = torch.bmm(x, gamma.transpose(2, 1))
        xh = torch.cat([x, h], dim=-1)

        assert_correctly_masked(xh, node_mask)
        
        if return_gamma:
            return xh, gamma
        else:
            return xh


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