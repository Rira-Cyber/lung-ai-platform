import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Two consecutive 3D convolutions.

    Conv3D
        ↓
    BatchNorm3D
        ↓
    ReLU
        ↓
    Conv3D
        ↓
    BatchNorm3D
        ↓
    ReLU
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size : int = 3,
            padding : int = 1,
    ) -> None:
        
        super().__init__()


        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        
        return self.block(x)
    

class DownBlock(nn.Module):
    """
    Encoder block.

    DoubleConv
        ↓
    MaxPool3D

    Returns
    -------
    pooled : torch.Tensor
        Tensor after max pooling.
    skip : torch.Tensor
        Tensor before pooling (used for skip connection).
    """
    def __init__(
            self,
            in_channels : int,
            out_channels : int,
    ) -> None:
        super().__init__()

        self.double_conv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
        )

        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor , torch.Tensor]:
        

        skip = self.double_conv(x)
        pooled = self.pool(skip)
        return pooled , skip

class UpBlock(nn.Module):
    """
    Decoder block.

    TransposedConv3D
            ↓
    Concatenate Skip Connection
            ↓
        DoubleConv
    """
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )

        self.double_conv = DoubleConv(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
        )


    def _conter_crop(
            self,
            encoder_feature: torch.Tensor,
            target_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Center crop encoder feature map to match decoder size.
        """

        _, _, d, h, w = encoder_feature.shape
        td, th, tw = target_shape

        d1 = (d - td) // 2
        h1 = (h - th) // 2
        w1 = (w - tw) // 2

        return encoder_feature[
            :,
            :,
            d1: d1+td,
            h1: h1+th,
            w1: w1+tw,
        ]
    
    def forward(
            self,
            x: torch.Tensor,
            skip: torch.Tensor,
    ) -> torch.Tensor:
        
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            skip = self._conter_crop(
                skip,
                x.shape[2:],
            )


        x = torch.cat(
            [skip, x],
            dim = 1,
        )

        x = self.double_conv(x)

        return x