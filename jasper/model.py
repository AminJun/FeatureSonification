import os

import toml
import torch
import torch.nn as nn

jasper_activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
}


def init_weights(m, mode='xavier_uniform'):
    if type(m) == nn.Conv1d or type(m) == MaskedConv1d:
        if mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        else:
            raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif type(m) == nn.BatchNorm1d:
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (kernel_size // 2) * dilation


class JasperEncoder(nn.Module):
    def __init__(self, **kwargs):
        cfg = {}
        for key, value in kwargs.items():
            cfg[key] = value

        nn.Module.__init__(self)
        self._cfg = cfg

        activation = jasper_activations[cfg['encoder']['activation']]()
        self.use_conv_mask = cfg['encoder'].get('convmask', False)
        feat_in = cfg['input']['features']
        init_mode = cfg.get('init_mode', 'xavier_uniform')

        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for lcfg in cfg['jasper']:
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True
            encoder_layers.append(
                JasperBlock(feat_in, lcfg['filters'], repeat=lcfg['repeat'],
                            kernel_size=lcfg['kernel'], stride=lcfg['stride'],
                            dilation=lcfg['dilation'], dropout=lcfg['dropout'],
                            residual=lcfg['residual'], activation=activation,
                            residual_panes=dense_res, use_conv_mask=self.use_conv_mask))
            feat_in = lcfg['filters']

        self.encoder = nn.Sequential(*encoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        if self.use_conv_mask:
            audio_signal, length = x
            return self.encoder(([audio_signal], length))
        else:
            return self.encoder([x])


class JasperDecoderForCTC(nn.Module):
    """Jasper decoder
    """

    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self._feat_in = kwargs.get("feat_in")
        self._num_classes = kwargs.get("num_classes")
        init_mode = kwargs.get('init_mode', 'xavier_uniform')

        self.decoder_layers = nn.Sequential(
            nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True), )
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, encoder_output):
        out = self.decoder_layers(encoder_output[-1]).transpose(1, 2)
        return nn.functional.log_softmax(out, dim=2)


class JasperEncoderDecoder(nn.Module):
    """Contains jasper encoder and decoder
    """

    def __init__(self):
        nn.Module.__init__(self)
        path = os.path.join(os.path.split(__file__)[0], 'jasper.toml')
        model_config = toml.load(path)
        self.transpose_in = False
        self.jasper_encoder = JasperEncoder(**model_config)
        self.jasper_decoder = JasperDecoderForCTC(feat_in=1024, num_classes=29)
        self.use_conv_mask = self.jasper_encoder.use_conv_mask

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: (torch.Tensor, torch.Tensor)):
        func = self.use_conv_forward if self.use_conv_mask else self.whole_forward
        return func(x)

    def use_conv_forward(self, x: (torch.Tensor, torch.Tensor)) -> (torch.Tensor, torch.Tensor):
        t_encoded_t, t_encoded_len_t = self.jasper_encoder(x)
        out = self.jasper_decoder(t_encoded_t)
        return out, t_encoded_len_t

    def whole_forward(self, x: (torch.Tensor, torch.Tensor)) -> (torch.Tensor, None):
        t_encoded_t = self.jasper_encoder(x)
        out = self.jasper_decoder(t_encoded_t)
        return out, None


class MaskedConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, use_conv_mask=True):
        super(MaskedConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                           stride=stride,
                                           padding=padding, dilation=dilation,
                                           groups=groups, bias=bias)
        self.use_conv_mask = use_conv_mask

    def get_seq_len(self, lens):
        return ((lens + 2 * self.padding[0] - self.dilation[0] * (
                self.kernel_size[0] - 1) - 1) // self.stride[0] + 1)

    def forward(self, inp):
        if self.use_conv_mask:
            x, lens = inp
            max_len = x.size(2)
            idxs = torch.arange(max_len).to(lens.dtype).to(lens.device).expand(len(lens), max_len)
            mask = idxs >= lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
            del mask
            del idxs
            lens = self.get_seq_len(lens)
            output = super(MaskedConv1d, self).forward(x)
            return output, lens
        else:
            return super(MaskedConv1d, self).forward(inp)


class JasperBlock(nn.Module):
    """Jasper Block. See https://arxiv.org/pdf/1904.03288.pdf
    """

    def __init__(self, inplanes, planes, repeat=3, kernel_size=11, stride=1,
                 dilation=1, padding='same', dropout=0.2, activation=None,
                 residual=True, residual_panes=[], use_conv_mask=False):
        super(JasperBlock, self).__init__()

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")

        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.use_conv_mask = use_conv_mask
        self.conv = nn.ModuleList()
        inplanes_loop = inplanes
        for _ in range(repeat - 1):
            self.conv.extend(
                self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size,
                                        stride=stride, dilation=dilation,
                                        padding=padding_val))
            self.conv.extend(
                self._get_act_dropout_layer(drop_prob=dropout, activation=activation))
            inplanes_loop = planes
        self.conv.extend(
            self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size,
                                    stride=stride, dilation=dilation,
                                    padding=padding_val))

        self.res = nn.ModuleList() if residual else None
        res_panes = residual_panes.copy()
        self.dense_residual = residual
        if residual:
            for ip in res_panes:
                self.res.append(nn.ModuleList(
                    modules=self._get_conv_bn_layer(ip, planes, kernel_size=1)))
        self.out = nn.Sequential(
            *self._get_act_dropout_layer(drop_prob=dropout, activation=activation))

    def _get_conv_bn_layer(self, in_channels, out_channels, kernel_size=11,
                           stride=1, dilation=1, padding=0, bias=False):
        layers = [
            MaskedConv1d(in_channels, out_channels, kernel_size, stride=stride,
                         dilation=dilation, padding=padding, bias=bias,
                         use_conv_mask=self.use_conv_mask),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]
        return layers

    def _get_act_dropout_layer(self, drop_prob=0.2, activation=None):
        if activation is None:
            activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
        layers = [
            activation,
            nn.Dropout(p=drop_prob)
        ]
        return layers

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_):
        if self.use_conv_mask:
            xs, lens_orig = input_
        else:
            xs = input_
            lens_orig = 0
        # compute forward convolutions
        out = xs[-1]
        lens = lens_orig
        for i, l in enumerate(self.conv):
            if self.use_conv_mask and isinstance(l, MaskedConv1d):
                out, lens = l((out, lens))
            else:
                out = l(out)
        # compute the residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if j == 0 and self.use_conv_mask:
                        res_out, _ = res_layer((res_out, lens_orig))
                    else:
                        res_out = res_layer(res_out)
                out += res_out

        # compute the output
        out = self.out(out)
        if self.res is not None and self.dense_residual:
            out = xs + [out]
        else:
            out = [out]

        if self.use_conv_mask:
            return out, lens
        else:
            return out


class GreedyCTCDecoder(nn.Module):
    def forward(self, log_prob: torch.Tensor):
        arg_max = log_prob.argmax(dim=-1, keepdim=False).int()
        return arg_max


class CTCLossNM(nn.Module):
    def __init__(self, num_classes: int = 29):
        super().__init__()
        self._blank = num_classes - 1
        self._criterion = nn.CTCLoss(blank=self._blank, reduction='none')

    def forward(self, log_prob, target, input_length, target_length):
        input_length = input_length.long()
        target_length = target_length.long()
        target = target.long()
        loss = self._criterion(log_prob.transpose(1, 0), target, input_length, target_length)
        # note that this is different from reduction = 'mean'
        # because we are not dividing by target lengths
        return torch.mean(loss)
