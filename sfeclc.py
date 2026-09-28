
import math

import torch
import torch.nn as nn


def can_cast(from_: _dtype, to: _dtype) -> _bool: 
    r"""
    can_cast(from_, to) -> bool
    
    Determines if a type conversion is allowed under PyTorch casting rules
    described in the type promotion :ref:`documentation <type-promotion-doc>`.
    
    Args:
        from\_ (dtype): The original :class:`torch.dtype`.
        to (dtype): The target :class:`torch.dtype`.
    
    Example::
    
        >>> torch.can_cast(torch.double, torch.float)
        True
        >>> torch.can_cast(torch.float, torch.int)
        False
    """
    ...
@overload
def cat(tensors: Optional[Union[tuple[Tensor, ...], list[Tensor]]], dim: _int = 0, *, out: Optional[Tensor] = None) -> Tensor: 
    r"""
    cat(tensors, dim=0, *, out=None) -> Tensor
    
    Concatenates the given sequence of tensors in :attr:`tensors` in the given dimension.
    All tensors must either have the same shape (except in the concatenating
    dimension) or be a 1-D empty tensor with size ``(0,)``.
    
    :func:`torch.cat` can be seen as an inverse operation for :func:`torch.split`
    and :func:`torch.chunk`.
    
    :func:`torch.cat` can be best understood via examples.
    
    .. seealso::
    
        :func:`torch.stack` concatenates the given sequence along a new dimension.
    
    Args:
        tensors (sequence of Tensors): Non-empty tensors provided must have the same shape,
            except in the cat dimension.
    
        dim (int, optional): the dimension over which the tensors are concatenated
    
    Keyword args:
        out (Tensor, optional): the output tensor.
    
    Example::
    
        >>> x = torch.randn(2, 3)
        >>> x
        tensor([[ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497]])
        >>> torch.cat((x, x, x), 0)
        tensor([[ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497],
                [ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497],
                [ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497]])
        >>> torch.cat((x, x, x), 1)
        tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
                 -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
                 -0.5790,  0.1497]])
    """
    ...
@overload
def cat(tensors: Optional[Union[tuple[Tensor, ...], list[Tensor]]], dim: Union[str, ellipsis, None], *, out: Optional[Tensor] = None) -> Tensor: 
    r"""
    cat(tensors, dim=0, *, out=None) -> Tensor
    
    Concatenates the given sequence of tensors in :attr:`tensors` in the given dimension.
    All tensors must either have the same shape (except in the concatenating
    dimension) or be a 1-D empty tensor with size ``(0,)``.
    
    :func:`torch.cat` can be seen as an inverse operation for :func:`torch.split`
    and :func:`torch.chunk`.
    
    :func:`torch.cat` can be best understood via examples.
    
    .. seealso::
    
        :func:`torch.stack` concatenates the given sequence along a new dimension.
    
    Args:
        tensors (sequence of Tensors): Non-empty tensors provided must have the same shape,
            except in the cat dimension.
    
        dim (int, optional): the dimension over which the tensors are concatenated
    
    Keyword args:
        out (Tensor, optional): the output tensor.
    
    Example::
    
        >>> x = torch.randn(2, 3)
        >>> x
        tensor([[ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497]])
        >>> torch.cat((x, x, x), 0)
        tensor([[ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497],
                [ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497],
                [ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497]])
        >>> torch.cat((x, x, x), 1)
        tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
                 -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
                 -0.5790,  0.1497]])
    """

class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)

class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class RCSPELAN(nn.Module):
    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5):
        super(RCSPELAN, self).__init__()
        
        self.c = int(c2 * e)  # hidden channels
        self.mid = int(self.c * scale)
        
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)
        
        self.cv3 = RepConv(self.c, self.mid, 3)
        self.m = nn.ModuleList(Conv(self.mid, self.mid, 3) for _ in range(n - 1))
        self.cv4 = Conv(self.mid, self.mid, 1)
        
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.extend(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))

class DynamicFrequencySelector(nn.Module):
    def __init__(
        self,
        dim,
        reduction=4,
        low_cut=0.25,
        high_cut=0.55,
        band_sharpness=20.0
    ):
        super().__init__()

        if not 0.0 < low_cut < high_cut < 1.0:
            raise ValueError(
                "必须满足 0 < low_cut < high_cut < 1。"
            )

        self.dim = dim
        self.low_cut = float(low_cut)
        self.high_cut = float(high_cut)
        self.band_sharpness = float(band_sharpness)

        hidden = max(dim // reduction, 4)
        self.band_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, dim * 3, kernel_size=1, bias=True)
        )

        # 基于频谱幅值预测逐频率位置权重
        # 输出尺寸为 [B, C, H, W // 2 + 1]
        self.freq_gate = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=dim,
                bias=True
            ),
            nn.GELU(),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=dim,
                bias=True
            )
        )

        # 自适应频率选择强度
        self.adaptive_scale = nn.Parameter(
            torch.full((1, dim, 1, 1), 0.1)
        )

        # 高频增强强度
        # 经过 0.5 * sigmoid 后，初始值约为 0.05
        self.high_logit = nn.Parameter(
            torch.full((1, dim, 1, 1), -2.2)
        )

    def _build_frequency_bands(self, height, width, device):

        fy = torch.fft.fftfreq(
            height,
            d=1.0,
            device=device
        )

        # rFFT 横向只保留非负频率
        fx = torch.fft.rfftfreq(
            width,
            d=1.0,
            device=device
        )

        yy, xx = torch.meshgrid(
            fy,
            fx,
            indexing="ij"
        )

        # 径向频率归一化到大致 [0, 1]
        radius = torch.sqrt(
            xx.square() + yy.square()
        )

        radius = radius / math.sqrt(
            0.5 ** 2 + 0.5 ** 2
        )

        k = self.band_sharpness

        # 低频软掩码
        low_mask = torch.sigmoid(
            k * (self.low_cut - radius)
        )

        # 中频软掩码
        mid_mask = (
            torch.sigmoid(
                k * (radius - self.low_cut)
            )
            *
            torch.sigmoid(
                k * (self.high_cut - radius)
            )
        )

        # 高频软掩码
        high_mask = torch.sigmoid(
            k * (radius - self.high_cut)
        )

        bands = torch.stack(
            (low_mask, mid_mask, high_mask),
            dim=0
        )

        # 归一化，使三个频段在每个频率坐标处的和约为1
        bands = bands / (
            bands.sum(dim=0, keepdim=True) + 1e-6
        )

        return bands

    def forward(self, x, return_debug=False):
        if x.ndim != 4:
            raise ValueError(
                f"输入必须为四维张量，当前尺寸为 {x.shape}"
            )

        batch_size, channels, height, width = x.shape

        if channels != self.dim:
            raise ValueError(
                f"期望输入通道数为 {self.dim}，"
                f"实际为 {channels}"
            )

        original_dtype = x.dtype

        # FFT在float32下执行，避免AMP训练中非2次幂尺寸报错
        x_fft = torch.fft.rfft2(
            x.float(),
            dim=(-2, -1),
            norm="ortho"
        )

        # 频谱幅值
        amplitude = torch.log1p(
            torch.abs(x_fft)
        )

        # 每个通道分别进行频谱归一化
        amplitude_mean = amplitude.mean(
            dim=(-2, -1),
            keepdim=True
        )

        amplitude_var = amplitude.var(
            dim=(-2, -1),
            keepdim=True,
            unbiased=False
        )

        amplitude_norm = (
            amplitude - amplitude_mean
        ) / torch.sqrt(
            amplitude_var + 1e-6
        )

        amplitude_norm = amplitude_norm.to(
            dtype=original_dtype
        )

        # 逐频率位置注意力
        # [B, C, H, W // 2 + 1]
        position_gate = torch.sigmoid(
            self.freq_gate(amplitude_norm)
        )

        # 每个通道的低频、中频和高频权重
        band_weight = self.band_mlp(x)

        band_weight = band_weight.view(
            batch_size,
            channels,
            3,
            1,
            1
        )

        band_weight = torch.softmax(
            band_weight,
            dim=2
        )

        # 固定频率坐标掩码
        bands = self._build_frequency_bands(
            height,
            width,
            x.device
        ).to(dtype=original_dtype)

        # bands:
        # [3, H, W // 2 + 1]
        #
        # band_weight:
        # [B, C, 3, 1, 1]

        # 输入自适应频段选择图
        # [B, C, H, W // 2 + 1]
        band_map = (
            band_weight
            *
            bands.unsqueeze(0).unsqueeze(0)
        ).sum(dim=2)

        high_mask = bands[2]

        # 自适应增强或抑制强度，范围约为[-1, 1]
        adaptive_strength = torch.tanh(
            self.adaptive_scale
        )

        # 显式高频增强强度，范围为(0, 0.5)
        high_strength = 0.5 * torch.sigmoid(
            self.high_logit
        )

        # 每个通道对高频频段的选择权重
        high_weight = band_weight[:, :, 2]

        # 第一项：
        # 根据输入内容，对不同频段和不同频率位置进行增强或抑制
        adaptive_term = (
            adaptive_strength
            *
            (2.0 * position_gate - 1.0)
            *
            band_map
        )

        # 第二项：
        # 显式增强目标边缘、轮廓和纹理对应的高频信息
        high_detail_term = (
            high_strength
            *
            position_gate
            *
            high_weight
            *
            high_mask
        )

        # 最终逐频率位置增益
        frequency_gain = (
            1.0
            + adaptive_term
            + high_detail_term
        )

        # 防止训练初期增益过大或接近0
        frequency_gain = frequency_gain.clamp(
            min=0.05,
            max=2.5
        )

        # 用实数增益调制复数频谱
        # 相位信息不会被改变
        enhanced_fft = (
            x_fft
            *
            frequency_gain.float()
        )

        # 恢复空间域
        enhanced = torch.fft.irfft2(
            enhanced_fft,
            s=(height, width),
            dim=(-2, -1),
            norm="ortho"
        )

        enhanced = enhanced.to(
            dtype=original_dtype
        )

        if return_debug:
            debug_info = {
                "frequency_gain": frequency_gain.detach(),
                "band_weight": band_weight.detach(),
                "position_gate": position_gate.detach(),
                "low_mask": bands[0].detach(),
                "mid_mask": bands[1].detach(),
                "high_mask": bands[2].detach()
            }

            return enhanced, debug_info

        return enhanced


class FGM(nn.Module):


    def __init__(self, dim):
        super().__init__()

        # 提取局部空间特征
        self.local_proj = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=dim,
                bias=True
            ),
            nn.GELU(),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=1,
                bias=True
            )
        )

        # 频率增强特征映射
        self.freq_proj = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=True
        )

        # 根据局部特征和频率特征共同生成调制门控
        self.gate = nn.Sequential(
            nn.Conv2d(
                dim * 2,
                dim,
                kernel_size=1,
                bias=True
            ),
            nn.Sigmoid()
        )

        # 小值初始化，避免新模块破坏预训练特征
        self.alpha = nn.Parameter(
            torch.full((1, dim, 1, 1), 1e-2)
        )

        self.beta = nn.Parameter(
            torch.ones(1, dim, 1, 1)
        )

    def forward(self, x_local, x_freq):
        """
        x_local:
            原始局部空间特征 [B, C, H, W]

        x_freq:
            频率选择后恢复到空间域的特征 [B, C, H, W]
        """

        local_feature = self.local_proj(
            x_local
        )

        frequency_feature = self.freq_proj(
            x_freq
        )

        modulation_gate = self.gate(
            torch.cat(
                (
                    local_feature,
                    frequency_feature
                ),
                dim=1
            )
        )

        # 频率信息引导局部特征增强
        modulated_local = (
            local_feature
            *
            (1.0 + modulation_gate)
        )

        # 以频率增强特征作为主分支，
        # 局部调制特征作为增量
        output = (
            self.beta * x_freq
            +
            self.alpha * modulated_local
        )

        return output


class FQIM(nn.Module):
    """
    Frequency-aware Query Interaction Module。
    """

    def __init__(
        self,
        dim,
        low_cut=0.25,
        high_cut=0.55
    ):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=1,
                bias=True
            ),
            nn.GELU()
        )

        self.out_conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=True
        )

        # 真正的动态频率选择模块
        self.freq_selector = DynamicFrequencySelector(
            dim=dim,
            low_cut=low_cut,
            high_cut=high_cut
        )

        # 重建特征通道校准
        self.sca_pool = nn.AdaptiveAvgPool2d(1)

        self.sca_conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=True
        )

        # 频率引导局部调制
        self.fgm = FGM(dim)

        # Query映射
        self.q_conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=True
        )

        # 最终查询注意力映射
        # Query分支中不提前使用Sigmoid
        self.attn_conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=True
        )

        # 残差分支缩放
        self.gamma_attn = nn.Parameter(
            torch.full((1, dim, 1, 1), 1e-2)
        )

        self.gamma_freq = nn.Parameter(
            torch.full((1, dim, 1, 1), 1e-2)
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 输入映射
        u = self.in_conv(x)

        # 动态频率选择
        u_freq = self.freq_selector(u)

        # 重建后的通道校准
        channel_weight = (
            1.0
            +
            torch.tanh(
                self.sca_conv(
                    self.sca_pool(u_freq)
                )
            )
        )

        u_freq = channel_weight * u_freq

        # 频率引导局部空间特征调制
        x_fgm = self.fgm(
            x_local=u,
            x_freq=u_freq
        )

        # Query特征
        query = self.q_conv(u)

        # 查询特征与频率增强特征交互
        attention = torch.sigmoid(
            self.attn_conv(
                query * x_fgm
            )
        )

        # 查询增强 + 频率增强 + 输入残差
        output = (
            x
            +
            self.gamma_attn
            * attention
            * u
            +
            self.gamma_freq
            * x_fgm
        )

        output = self.act(output)

        return self.out_conv(output)


class CSPFQIM(nn.Module):


    def __init__(
        self,
        dim,
        e=0.25,
        low_cut=0.25,
        high_cut=0.55
    ):
        super().__init__()

        if not 0.0 < e < 1.0:
            raise ValueError(
                "参数 e 必须位于 (0, 1) 之间。"
            )

        self.dim = dim
        self.e = e

        # 提前在初始化阶段确定通道数，
        # 不在forward中根据输入动态计算
        self.enhanced_channels = max(
            1,
            int(round(dim * e))
        )

        self.identity_channels = (
            dim - self.enhanced_channels
        )

        if self.identity_channels < 1:
            raise ValueError(
                "恒等分支通道数不能小于1。"
            )

        self.cv1 = Conv(
            dim,
            dim,
            1
        )

        self.cv2 = Conv(
            dim,
            dim,
            1
        )

        self.m = FQIM(
            dim=self.enhanced_channels,
            low_cut=low_cut,
            high_cut=high_cut
        )

    def forward(self, x):
        mapped = self.cv1(x)

        enhanced_branch, identity_branch = torch.split(
            mapped,
            [
                self.enhanced_channels,
                self.identity_channels
            ],
            dim=1
        )

        enhanced_branch = self.m(
            enhanced_branch
        )

        output = torch.cat(
            (
                enhanced_branch,
                identity_branch
            ),
            dim=1
        )

        return self.cv2(output)


class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    # (1, 3, 6, 8)
    # (1, 4, 8,12)
    def __init__(self, grids=(1, 2, 3, 6), channels=256):
        super(PSPModule, self).__init__()

        self.grids = grids
        self.channels = channels

    def forward(self, feats):

        b, c , h , w = feats.size()
        ar = w / h

        return torch.cat([
            F.adaptive_avg_pool2d(feats, (self.grids[0], max(1, round(ar * self.grids[0])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[1], max(1, round(ar * self.grids[1])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[2], max(1, round(ar * self.grids[2])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[3], max(1, round(ar * self.grids[3])))).view(b, self.channels, -1)
        ], dim=2)

class LocalAttenModule(nn.Module):
    def __init__(self, in_channels=256,inter_channels=32):
        super(LocalAttenModule, self).__init__()

        self.conv = nn.Sequential(
            Conv(in_channels, inter_channels,1),
            nn.Conv2d(inter_channels, in_channels, kernel_size=3, padding=1, bias=False))

        self.tanh_spatial = nn.Tanh()
        self.conv[1].weight.data.zero_()
        self.keras_init_weight()
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d,nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        res1 = x
        res2 = x

        x = self.conv(x)
        x_mask = self.tanh_spatial(x)

        res1 = res1 * x_mask

        return res1 + res2

class HCFM(nn.Module):
    def __init__(self, in_channels=512, grids=(6, 3, 2, 1)): # 先ce后ffm
        super(HCFM, self).__init__()
        self.grids = grids
        inter_channels = in_channels // 2
        self.inter_channels = inter_channels

        self.reduce_channel = Conv(in_channels, inter_channels, 3)
        self.query_conv = nn.Conv2d(in_channels=inter_channels, out_channels=32, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=inter_channels, out_channels=32, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=inter_channels, out_channels=self.inter_channels, kernel_size=1)
        self.key_channels = 32

        self.value_psp = PSPModule(grids, inter_channels)
        self.key_psp = PSPModule(grids, inter_channels)

        self.softmax = nn.Softmax(dim=-1)

        self.local_attention = LocalAttenModule(inter_channels,inter_channels//8)
        self.keras_init_weight()
        
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d,nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):

        x = self.reduce_channel(x) # 通道降维- 128

        m_batchsize,_,h,w = x.size()

        query = self.query_conv(x).view(m_batchsize,32,-1).permute(0,2,1) ##  b c n ->  b n c

        key = self.key_conv(self.key_psp(x))  ## b c s

        sim_map = torch.matmul(query,key)

        sim_map = self.softmax(sim_map)
       
        value = self.value_conv(self.value_psp(x)) #.permute(0,2,1)  ## b c s

        
        context = torch.bmm(value,sim_map.permute(0,2,1))  #  B C S * B S N - >  B C N

       
        context = context.view(m_batchsize,self.inter_channels,h,w)
      
        context = self.local_attention(context)

        out = x + context

        return out



class CFRM(nn.Module):
    def __init__(self, inc):
        super(CFRM, self).__init__()
        hidc = inc[0]
        
        self.groups = 2
        self.conv_8 = Conv(inc[0], hidc, 3)
        self.conv_32 = Conv(inc[1], hidc, 3)

        self.conv_offset = nn.Sequential(
            Conv(hidc * 2, 64),
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False)
        )  #Conv Block 预测网络  两个厚度为 hidc 的特征图叠在一起，厚度自然就变成了两倍。

        self.keras_init_weight()
        self.conv_offset[1].weight.data.zero_()
        
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
                
    def forward(self, x):
        cp, sp = x    #cp 浅层 sp高层
        n, _, out_h, out_w = cp.size()

        # x_32
        sp = self.conv_32(sp)  # 语义特征  1 / 8  256
        sp = F.interpolate(sp, cp.size()[2:], mode='bilinear', align_corners=True)  #上采样深层特征使得浅层特征和深层特征具有相同的空间尺寸
        # x_8
        cp = self.conv_8(cp)

        conv_results = self.conv_offset(torch.cat([cp, sp], 1)) #将 cp 和 sp 在通道维度拼接，送入预测网络，得到前面提到的 10 个通道的预测结果 conv_results。

        sp = sp.reshape(n*self.groups,-1,out_h,out_w)
        cp = cp.reshape(n*self.groups,-1,out_h,out_w)

        offset_l = conv_results[:, 0:self.groups*2, :, :].reshape(n*self.groups,-1,out_h,out_w)
        offset_h = conv_results[:, self.groups*2:self.groups*4, :, :].reshape(n*self.groups,-1,out_h,out_w)


        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(sp).to(sp.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n*self.groups, 1, 1, 1).type_as(sp).to(sp.device)

        grid_l = grid + offset_l.permute(0, 2, 3, 1) / norm
        grid_h = grid + offset_h.permute(0, 2, 3, 1) / norm

        cp = F.grid_sample(cp, grid_l , align_corners=True)  ## 考虑是否指定align_corners
        sp = F.grid_sample(sp, grid_h , align_corners=True)  ## 考虑是否指定align_corners

        cp = cp.reshape(n, -1, out_h, out_w)
        sp = sp.reshape(n, -1, out_h, out_w)

        att = 1 + torch.tanh(conv_results[:, self.groups*4:, :, :])
        sp = sp * att[:, 0:1, :, :] + cp * att[:, 1:2, :, :]

        return sp


