import torch
import torch.nn as nn
import lightning
from lightning import LightningModule
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
    LRSchedulerTypeUnion,
    OptimizerLRScheduler,
)
from timm.scheduler.step_lr import StepLRScheduler
from typing import Any

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.gates = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=self.padding)

    def forward(self, input_tensor, hidden_state):
        input_tensor = input_tensor.float()
        h_cur, c_cur = hidden_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        gates = self.gates(combined)
        i_gate, f_gate, o_gate, g_gate = torch.split(gates, self.hidden_channels, dim=1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)

        c_next = f_gate * c_cur + i_gate * g_gate
        h_next = o_gate * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.gates.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.gates.weight.device))

class ConvLSTM(lightning.LightningModule):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, output_channels):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.output_channels = output_channels

        layers = []
        for i in range(num_layers):
            in_channels = input_channels if i == 0 else hidden_channels
            layers.append(ConvLSTMCell(in_channels, hidden_channels, kernel_size))

        self.layers = nn.ModuleList(layers)
        self.output_conv = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, input):
        input = input.float()
        batch_size, channel, time, height, width = input.size()
        current_input = input
        hidden_states = []

        for layer in self.layers:
            h, c = layer.init_hidden(batch_size, (height, width))
            output_inner = []
            for t in range(time):
                h, c = layer(current_input[:, :, t, :, :], (h, c))
                output_inner.append(h)
            current_input = torch.stack(output_inner, dim=2)
            hidden_states.append((h, c))

        last_output = self.output_conv(current_input[:, :, -1, :, :])
        last_output = last_output.unsqueeze(2)
        return last_output

    def set_optimizer_config(self, learning_rate: float, weight_decay: float):
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, metric: Any | None  # 指定学习率调度器  #
    ) -> None:
        scheduler.step(epoch=self.current_epoch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self._learning_rate,  # 学习率
            weight_decay=self._weight_decay,
        )
        scheduler = StepLRScheduler(
            optimizer, decay_t=10, decay_rate=0.7, warmup_lr_init=1e-5, warmup_t=5
        )
        super().configure_optimizers()  # 这个函数是必须调用的
        return dict(optimizer=optimizer, lr_scheduler=scheduler)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1)
        y_hat = self(x)
        mse_loss = nn.functional.mse_loss(y_hat, y)
        rmse_loss = torch.sqrt(mse_loss)
        mae_loss = nn.functional.l1_loss(y_hat, y)
        mape_loss = torch.mean(torch.abs((y - y_hat) / y))
        self.log('val_mse_loss', mse_loss)
        self.log('val_rmse_loss', rmse_loss)
        self.log('val_mae_loss', mae_loss)
        self.log('val_mape_loss', mape_loss)
        return mse_loss


def main():
    input_tensor = torch.randn(64, 2, 3, 32, 32)  # batch_size, channels, time_steps, height, width
    model = ConvLSTM(input_channels=2, hidden_channels=16, kernel_size=3, num_layers=2, output_channels=2)
    output = model(input_tensor)
    print(output.shape)

if __name__ == "__main__":
    main()