import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class convt(nn.Module):
    def __init__(self):
        super(convt, self).__init__()

    def forward(self, x, w):
        x = torch.einsum('bne, ek->bnk', (x, w))
        return x.contiguous()
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A, dims):
        if dims == 2:
            x = torch.einsum('ncvl,vw->ncwl', (x, A))
        elif dims == 3:
            x = torch.einsum('ncvl,nvw->ncwl', (x, A))
        else:
            raise NotImplementedError('DFDGCN not implemented for A of dimension ' + str(dims))
        return x.contiguous()

class linear(nn.Module):
    """Linear layer."""

    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(
            1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    """Graph convolution network."""

    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()

        self.c_in = c_in
        c_in = (order * (support_len + 1) + 1) * self.c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):

        out = [x]
        for a in support:
            x1 = self.nconv(x, a.to(x.device), a.dim())
            out.append(x1)

            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a.to(x1.device), a.dim())
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h



def dy_mask_graph(adj, k):
    M = []
    for i in range(adj.size(0)):
        adp = adj[i]
        mask = torch.zeros( adj.size(1),adj.size(2)).to(adj.device)
        mask = mask.fill_(float("0"))
        s1, t1 = (adp + torch.rand_like(adp) * 0.01).topk(k, 1)
        mask = mask.scatter_(1, t1, s1.fill_(1))
        M.append(mask)
    mask = torch.stack(M,dim=0)
    adj = adj * mask
    return adj



def cat(x1,x2):
    M = []
    for i in range(x1.size(0)):
        x = x1[i]
        new_x = torch.cat([x,x2],dim=1)
        M.append(new_x)
    result = torch.stack(M,dim=0)
    return result


class DFDGCN(nn.Module):

    def __init__(self, num_nodes, dropout=0.3, supports=None,
                    gcn_bool=True, addaptadj=True, aptinit=None,
                    in_dim=2, out_dim=12, residual_channels=32,
                    dilation_channels=32, skip_channels=256, end_channels=512,
                    kernel_size=2, blocks=4, layers=2, a=1, seq_len=12, affine=True, fft_emb=10, identity_emb=10, hidden_emb=30, subgraph=20):
        super(DFDGCN, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.seq_len = seq_len
        self.a = a

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.supports = supports
        self.emb = fft_emb
        self.subgraph_size = subgraph
        self.identity_emb = identity_emb
        self.hidden_emb = hidden_emb
        self.fft_len = round(seq_len//2) + 1
        self.Ex1 = nn.Parameter(torch.randn(self.fft_len, self.emb), requires_grad=True)
        self.Wd = nn.Parameter(torch.randn(num_nodes,self.emb + self.identity_emb + self.seq_len * 2, self.hidden_emb), requires_grad=True)
        self.Wxabs = nn.Parameter(torch.randn(self.hidden_emb, self.hidden_emb), requires_grad=True)

        self.mlp = linear(residual_channels * 4,residual_channels)
        self.layersnorm = torch.nn.LayerNorm(normalized_shape=[num_nodes,self.hidden_emb], eps=1e-08,elementwise_affine=affine)
        self.convt = convt()

        self.node1 = nn.Parameter(
            torch.randn(num_nodes, self.identity_emb), requires_grad=True)
        self.drop = nn.Dropout(p=dropout)

        self.T_i_D_emb = nn.Parameter(
            torch.empty(288, self.seq_len))
        self.D_i_W_emb = nn.Parameter(
            torch.empty(7, self.seq_len))

        receptive_field = 1
        self.reset_parameter()
        self.supports_len = 0
        if not addaptadj:
            self.supports_len -= 1
        if supports is not None:
            self.supports_len += len(supports)
        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(
                    torch.randn(num_nodes, self.emb), requires_grad=True)
                self.nodevec2 = nn.Parameter(
                    torch.randn(self.emb, num_nodes), requires_grad=True)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(
                        gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feedforward function of DFDGCN; Based on Graph WaveNet

        Args:
            history_data (torch.Tensor): shape [B, L, N, C]

        Graphs:
            predefined graphs: two graphs; [2, N, N] : Pre-given graph structure, including in-degree and out-degree graphs

            self-adaptive graph: [N, N] : Self-Adaptively constructed graphs with two learnable parameters
                torch.mm(self.nodevec1, self.nodevec2)
                    nodevec: [N, Emb]

            dynamic frequency domain graph: [B, N, N] : Data-driven graphs constructed with frequency domain information from traffic data
                traffic_data : [B, N, L]
                frequency domain information : [B, N, L/2.round + 1] ------Embedding ------[B, N, Emb2]
                Identity embedding : learnable parameter [N, Emb3]
                Time embedding : Week and Day : [N, 7] [N, 24(hour) * 12 (60min / 5min due to sampling)] ------Embedding ------ [N, 2 * Emb4]
                Concat frequency domain information + Identity embedding + Time embedding ------Embedding , Activating, Normalization and Dropout
                Conv1d to get adjacency matrix

        Returns:
            torch.Tensor: [B, L, N, 1]
        """
        #num_feat = model_args["num_feat"]
        input = history_data.transpose(1, 3).contiguous()[:,0:2,:,:]
        data = history_data

        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(
                input, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)

        skip = 0
        if self.gcn_bool and self.addaptadj and self.supports is not None:


            gwadp = F.softmax(
                F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

            new_supports = self.supports + [gwadp] # pretrained graph in DCRNN and self-adaptive graph in GWNet

            # Construction of dynamic frequency domain graph
            xn1 = input[:, 0, :, -self.seq_len:]

            T_D = self.T_i_D_emb[(data[:, :, :, 1] * 288).type(torch.LongTensor)][:, -1, :, :]
            D_W = self.D_i_W_emb[(data[:, :, :, 1 + 1]).type(torch.LongTensor)][:, -1, :, :]

            xn1 = torch.fft.rfft(xn1, dim=-1)
            xn1 = torch.abs(xn1)

            xn1 = torch.nn.functional.normalize(xn1, p=2.0, dim=1, eps=1e-12, out=None)
            xn1 = torch.nn.functional.normalize(xn1, p=2.0, dim=2, eps=1e-12, out=None) * self.a


            xn1 = torch.matmul(xn1, self.Ex1)
            xn1k = cat(xn1, self.node1)
            x_n1 = torch.cat([xn1k, T_D, D_W], dim=2)
            x1 = torch.bmm(x_n1.permute(1,0,2),self.Wd).permute(1,0,2)
            x1 = torch.relu(x1)
            x1k = self.layersnorm(x1)
            x1k = self.drop(x1k)
            adp = self.convt(x1k, self.Wxabs)
            adj = torch.bmm(adp, x1.permute(0, 2, 1))
            adp = torch.relu(adj)
            adp = dy_mask_graph(adp, self.subgraph_size)
            adp = F.softmax(adp, dim=2)
            new_supports = new_supports + [adp]



        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*


            # dilated convolution
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x

            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]

            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)

                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
