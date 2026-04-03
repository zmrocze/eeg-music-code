import torch
import torch.nn as nn


class BiLSTM(nn.Module):
  def __init__(
    self, input_size: int, hidden_size: int, num_layers: int = 2, output_size: int = 10
  ):
    super().__init__()
    self.lstm = nn.LSTM(
      input_size=input_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      batch_first=True,
      bidirectional=True,
    )
    self.fc = nn.Linear(hidden_size * 2, output_size)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    lstm_out, _ = self.lstm(x)
    return self.fc(lstm_out)


def main():
  input_size = 12
  hidden_size = 64
  num_layers = 2
  output_size = 1

  model = BiLSTM(input_size, hidden_size, num_layers, output_size)

  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

  print(f"BiLSTM Model with {num_layers} layers")
  print(f"Input size: {input_size}")
  print(f"Hidden size: {hidden_size}")
  print(f"Output size: {output_size}")
  print(f"\nTotal parameters: {total_params:,}")
  print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
  main()
