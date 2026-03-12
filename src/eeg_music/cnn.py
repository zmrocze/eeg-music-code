import torch
import torch.nn as nn


class ConvBlock(nn.Module):
  """Conv over 2nd dim -> BatchNorm -> Conv over 1st dim -> ELU -> MaxPool2d -> Dropout."""

  def __init__(
    self,
    in_ch: int,
    mid_ch: int,
    out_ch: int,
    freq_kernel: int = 5,
    time_kernel: int = 5,
    pool: tuple[int, int] = (2, 2),
    dropout: float = 0.25,
  ):
    super().__init__()
    self.conv_freq = nn.Conv2d(
      in_ch, mid_ch, kernel_size=(1, freq_kernel), padding=(0, freq_kernel // 2)
    )
    self.bn = nn.BatchNorm2d(mid_ch)
    self.conv_time = nn.Conv2d(
      mid_ch, out_ch, kernel_size=(time_kernel, 1), padding=(time_kernel // 2, 0)
    )
    self.elu = nn.ELU()
    self.pool = nn.MaxPool2d(kernel_size=pool)
    self.drop = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.drop(self.pool(self.elu(self.conv_time(self.bn(self.conv_freq(x))))))

  def forward_verbose(self, x: torch.Tensor, prefix: str = "") -> torch.Tensor:
    for name, layer in [
      ("conv_freq", self.conv_freq),
      ("bn", self.bn),
      ("conv_time", self.conv_time),
      ("elu", self.elu),
      ("pool", self.pool),
      ("drop", self.drop),
    ]:
      x = layer(x)
      print(f"  {prefix}{name:<20} {list(x.shape)}")
    return x


class DeconvBlock(nn.Module):
  """Upsample -> Conv over 1st dim -> BatchNorm -> Conv over 2nd dim -> ELU -> Dropout."""

  def __init__(
    self,
    in_ch: int,
    mid_ch: int,
    out_ch: int,
    time_kernel: int = 5,
    freq_kernel: int = 5,
    scale: tuple[int, int] = (2, 2),
    dropout: float = 0.25,
  ):
    super().__init__()
    self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)
    self.conv_time = nn.Conv2d(
      in_ch, mid_ch, kernel_size=(time_kernel, 1), padding=(time_kernel // 2, 0)
    )
    self.bn = nn.BatchNorm2d(mid_ch)
    self.conv_freq = nn.Conv2d(
      mid_ch, out_ch, kernel_size=(1, freq_kernel), padding=(0, freq_kernel // 2)
    )
    self.elu = nn.ELU()
    self.drop = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.drop(self.elu(self.conv_freq(self.bn(self.conv_time(self.up(x))))))

  def forward_verbose(self, x: torch.Tensor, prefix: str = "") -> torch.Tensor:
    for name, layer in [
      ("up", self.up),
      ("conv_time", self.conv_time),
      ("bn", self.bn),
      ("conv_freq", self.conv_freq),
      ("elu", self.elu),
      ("drop", self.drop),
    ]:
      x = layer(x)
      print(f"  {prefix}{name:<20} {list(x.shape)}")
    return x


# ---------------------------------------------------------------------------
# Classification model
# ---------------------------------------------------------------------------


class CNNClassifier(nn.Module):
  def __init__(self, num_classes: int = 4, in_channels: int = 1, dropout: float = 0.25):
    super().__init__()
    # Input: (batch, in_channels, 60, 20)
    self.block1 = ConvBlock(
      in_channels, 16, 32, freq_kernel=5, time_kernel=5, dropout=dropout
    )
    # After block1: (B, 32, 30, 10)
    self.block2 = ConvBlock(32, 64, 128, freq_kernel=3, time_kernel=3, dropout=dropout)
    # After block2: (B, 128, 15, 5)
    self.block3 = ConvBlock(
      128, 128, 128, freq_kernel=3, time_kernel=3, dropout=dropout
    )
    # After block3: (B, 128, 7, 2)
    self.flatten = nn.Flatten()
    # self.fc = nn.Linear(128 * 7 * 2, num_classes)
    self.fc = nn.Linear(128 * 7 * 1, num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc(self.flatten(self.block3(self.block2(self.block1(x)))))

  def forward_verbose(self, x: torch.Tensor) -> torch.Tensor:
    print(f"{'Input':<25} {list(x.shape)}")
    x = self.block1.forward_verbose(x, "block1.")
    x = self.block2.forward_verbose(x, "block2.")
    x = self.block3.forward_verbose(x, "block3.")
    for name, layer in [("flatten", self.flatten), ("fc", self.fc)]:
      x = layer(x)
      print(f"  {name:<20} {list(x.shape)}")
    return x


# ---------------------------------------------------------------------------
# Reconstruction model (U-Net style skip connections)
# ---------------------------------------------------------------------------


class CNNReconstruction(nn.Module):
  """Encoder (block1 + block2) -> FC bottleneck -> Decoder (deconv blocks)
  with U-Net residual / skip connections between encoder and decoder stages.
  Output shape: (batch, 1, 64, 64).
  """

  def __init__(
    self, in_channels: int = 1, out_channels: int = 1, dropout: float = 0.25
  ):
    super().__init__()

    # ---- Encoder (same as CNNClassifier) ----
    # Input: (B, in_channels, 60, 20)
    self.enc1 = ConvBlock(
      in_channels, 16, 32, freq_kernel=5, time_kernel=5, dropout=dropout
    )
    # After enc1: (B, 32, 30, 10)
    self.enc2 = ConvBlock(32, 64, 128, freq_kernel=3, time_kernel=3, dropout=dropout)
    # After enc2: (B, 128, 15, 5)
    self.enc3 = ConvBlock(128, 128, 128, freq_kernel=3, time_kernel=3, dropout=dropout)
    # After enc3: (B, 128, 7, 2)

    # ---- Bottleneck ----
    self.flatten = nn.Flatten()
    # self.fc_down = nn.Linear(128 * 7 * 2, 128)
    self.fc_down = nn.Linear(128 * 7 * 1, 128)
    # (B,128,1,1) -> (B,128,7,2) via transposed conv
    self.fc_up = nn.ConvTranspose2d(128, 128, kernel_size=(7, 1))
    self.bottleneck_elu = nn.ELU()

    # ---- Decoder ----
    # After reshape: (B, 128, 7, 2)
    # Skip from enc3: (B, 128, 7, 2) — concat -> 256
    self.dec3 = DeconvBlock(
      256, 128, 128, time_kernel=3, freq_kernel=3, scale=(2, 2), dropout=dropout
    )
    # After dec3: (B, 128, 14, 4) — need to match enc2 output (15, 5)
    # self.dec3_spatial_adapt = nn.AdaptiveAvgPool2d((15, 5))
    self.dec3_spatial_adapt = nn.AdaptiveAvgPool2d((15, 2))

    # Skip from enc2: (B, 128, 15, 5) — concat -> 256
    self.dec2 = DeconvBlock(
      256, 64, 32, time_kernel=3, freq_kernel=3, scale=(2, 2), dropout=dropout
    )
    # After dec2: (B, 32, 30, 4) — need to match enc1 output (30, 5)
    self.dec2_spatial_adapt = nn.AdaptiveAvgPool2d((30, 5))

    # Skip from enc1 has shape (B, 32, 30, 5) — concat -> 64 channels
    self.dec1 = DeconvBlock(
      64, 16, out_channels, time_kernel=5, freq_kernel=5, scale=(2, 2), dropout=dropout
    )
    # After dec1: (B, out_channels, 60, 20)

    # ---- Adapt to target (64, 64) ----
    # Use a conv + adaptive pool to reshape spatial dims from (60,20) -> (64,64)
    self.adapt = nn.Sequential(
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.ELU(),
      nn.AdaptiveAvgPool2d((64, 64)),
    )

    # Skip from input: (B, in_channels, 60, 20) needs spatial match to (B, out_channels, 64, 64)
    self.skip_input_adapt = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=1),
      nn.AdaptiveAvgPool2d((64, 64)),
    )

    self.final_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    skip_input = x

    # Encoder
    e1 = self.enc1(x)
    e2 = self.enc2(e1)
    e3 = self.enc3(e2)

    # Bottleneck
    z = self.bottleneck_elu(self.fc_down(self.flatten(e3)))
    d = self.fc_up(z.unsqueeze(-1).unsqueeze(-1))  # (B,32) -> (B,32,1,1) -> (B,128,7,2)

    # Decoder with skip connections
    d = self.dec3(torch.cat([d, e3], dim=1))  # skip from enc3
    d = self.dec3_spatial_adapt(d)  # align spatial to enc2
    d = self.dec2(torch.cat([d, e2], dim=1))  # skip from enc2
    d = self.dec2_spatial_adapt(d)  # align spatial to enc1
    d = self.dec1(torch.cat([d, e1], dim=1))  # skip from enc1

    # Adapt spatial dims to (64, 64)
    d = self.adapt(d)

    # Final skip from input
    return self.final_conv(torch.cat([d, self.skip_input_adapt(skip_input)], dim=1))

  def forward_verbose(self, x: torch.Tensor) -> torch.Tensor:
    print(f"{'Input':<25} {list(x.shape)}")
    skip_input = x

    # Encoder
    print("--- Encoder Block 1 ---")
    e1 = self.enc1.forward_verbose(x, "enc1.")
    print("--- Encoder Block 2 ---")
    e2 = self.enc2.forward_verbose(e1, "enc2.")
    print("--- Encoder Block 3 ---")
    e3 = self.enc3.forward_verbose(e2, "enc3.")

    # Bottleneck
    print("--- Bottleneck ---")
    z = self.flatten(e3)
    print(f"  {'flatten':<20} {list(z.shape)}")
    z = self.bottleneck_elu(self.fc_down(z))
    print(f"  {'fc_down+elu':<20} {list(z.shape)}")
    d = z.unsqueeze(-1).unsqueeze(-1)
    print(f"  {'unsqueeze':<20} {list(d.shape)}")
    d = self.fc_up(d)
    print(f"  {'fc_up(convT)':<20} {list(d.shape)}")

    # Decoder
    print("--- Decoder Block 3 (+ skip from enc3) ---")
    d = torch.cat([d, e3], dim=1)
    print(f"  {'cat(d, e3)':<20} {list(d.shape)}")
    d = self.dec3.forward_verbose(d, "dec3.")
    d = self.dec3_spatial_adapt(d)
    print(f"  {'spatial_adapt':<20} {list(d.shape)}")

    print("--- Decoder Block 2 (+ skip from enc2) ---")
    d = torch.cat([d, e2], dim=1)
    print(f"  {'cat(d, e2)':<20} {list(d.shape)}")
    d = self.dec2.forward_verbose(d, "dec2.")
    d = self.dec2_spatial_adapt(d)
    print(f"  {'spatial_adapt':<20} {list(d.shape)}")

    print("--- Decoder Block 1 (+ skip from enc1) ---")
    d = torch.cat([d, e1], dim=1)
    print(f"  {'cat(d, e1)':<20} {list(d.shape)}")
    d = self.dec1.forward_verbose(d, "dec1.")

    print("--- Adapt to (64, 64) ---")
    d = self.adapt(d)
    print(f"  {'adapt':<20} {list(d.shape)}")

    print("--- Final skip from input ---")
    s = self.skip_input_adapt(skip_input)
    print(f"  {'skip_input_adapt':<20} {list(s.shape)}")
    d = torch.cat([d, s], dim=1)
    print(f"  {'cat(d, skip)':<20} {list(d.shape)}")
    d = self.final_conv(d)
    print(f"  {'final_conv':<20} {list(d.shape)}")
    return d


def _print_params(model: nn.Module, name: str) -> None:
  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"\n[{name}]  Total params: {total:,}  |  Trainable: {trainable:,}")


if __name__ == "__main__":
  # x = torch.randn(2, 1, 60, 20)
  x = torch.randn(2, 1, 60, 10)

  print("=" * 60)
  print("CNNClassifier")
  print("=" * 60)
  clf = CNNClassifier(num_classes=4, in_channels=1)
  clf.eval()
  clf.forward_verbose(x)
  _print_params(clf, "CNNClassifier")

  print("\n" + "=" * 60)
  print("CNNReconstruction")
  print("=" * 60)
  rec = CNNReconstruction(in_channels=1, out_channels=1)
  rec.eval()
  rec.forward_verbose(x)
  _print_params(rec, "CNNReconstruction")
