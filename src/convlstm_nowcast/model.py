"""Checkpoint-compatible ConvLSTM with bounded-memory temporal streaming."""

import json
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h_cur, c_cur = state
        combined = self.conv(torch.cat([x, h_cur], dim=1))
        cc_i, cc_f, cc_o, cc_g = torch.split(combined, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: tuple[int, ...], kernel_size: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell_list = nn.ModuleList(
            ConvLSTMCell(
                input_channels if index == 0 else hidden_channels[index - 1],
                channels,
                kernel_size,
            )
            for index, channels in enumerate(hidden_channels)
        )

    def forward(
        self,
        x: torch.Tensor,
        states: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if states is None:
            batch, _, _, height, width = x.shape
            states = [
                (
                    x.new_zeros(batch, channels, height, width),
                    x.new_zeros(batch, channels, height, width),
                )
                for channels in self.hidden_channels
            ]
        # Each layer at time t needs only the same layer's state at t-1 and
        # the preceding layer's output at t. This avoids storing every frame
        # of every layer, unlike the training reference.
        for timestep in range(x.shape[1]):
            frame = x[:, timestep]
            next_states = []
            for cell, state in zip(self.cell_list, states, strict=True):
                state = cell(frame, state)
                next_states.append(state)
                frame = state[0]
            states = next_states
        return frame, states


class PrecipitationForecaster(nn.Module):
    def __init__(
        self,
        input_steps: int,
        output_steps: int,
        img_size: tuple[int, int],
        input_channels: int,
        hidden_channels: tuple[int, ...],
        kernel_size: int,
    ):
        super().__init__()
        if (
            input_steps < 1
            or output_steps < 1
            or min(img_size) < 1
            or input_channels != 1
            or not hidden_channels
            or min(hidden_channels) < 1
            or kernel_size < 1
            or kernel_size % 2 == 0
        ):
            raise ValueError("Invalid ConvLSTM architecture")
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.img_size = tuple(img_size)
        self.encoder = ConvLSTM(input_channels, hidden_channels, kernel_size)
        self.decoder = ConvLSTM(hidden_channels[-1], hidden_channels, kernel_size)
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if x.ndim != 5 or tuple(x.shape[1:]) != (self.input_steps, 1, *self.img_size):
            raise ValueError(f"Expected [batch, {self.input_steps}, 1, {self.img_size}]")
        decoder_input, states = self.encoder(x)
        predictions = []
        for _ in range(self.output_steps):
            decoder_input, states = self.decoder(decoder_input.unsqueeze(1), states)
            predictions.append(self.output_conv(decoder_input))
        # The reference has no output ReLU. Physical negatives are clipped only
        # after inverse log-zscore by the bundled processing runtime.
        return torch.stack(predictions, dim=1)


def load_model(path: Path, device: torch.device, expected_shape: tuple[int, int]):
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    architecture = manifest["architecture"]
    if manifest["model"] != "ConvLSTM" or architecture["encoder"] != "convlstm":
        raise ValueError("Not a supported ConvLSTM bundle")
    input_steps = architecture["input_steps"]
    output_steps = architecture["output_steps"]
    if type(input_steps) is not int or type(output_steps) is not int:
        raise ValueError("ConvLSTM sequence lengths must be integers in the bundle")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ValueError("Checkpoint must contain a state dictionary")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Empty or invalid state dictionary")
    while all(k.startswith("module.") for k in state) or all(
        k.startswith("_orig_mod.") for k in state
    ):
        state = {k.split(".", 1)[1]: value for k, value in state.items()}
    try:
        hidden_channels = []
        for index in range(len(state)):
            key = f"encoder.cell_list.{index}.conv.weight"
            if key not in state:
                break
            hidden_channels.append(state[key].shape[0] // 4)
        stem = state["encoder.cell_list.0.conv.weight"]
        input_channels = stem.shape[1] - hidden_channels[0]
        kernel_size = stem.shape[-1]
        if stem.shape[-2] != kernel_size or stem.shape[0] != 4 * hidden_channels[0]:
            raise ValueError("Invalid ConvLSTM gate convolution")
    except (KeyError, AttributeError, IndexError) as exc:
        raise ValueError("Not a supported ConvLSTM checkpoint") from exc
    model = PrecipitationForecaster(
        input_steps,
        output_steps,
        expected_shape,
        input_channels,
        tuple(hidden_channels),
        kernel_size,
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def forecast_steps(model: PrecipitationForecaster, inputs: torch.Tensor, steps: int):
    """Run the reference's six-step decoder once and yield requested leads."""
    if not 1 <= steps <= model.output_steps:
        raise ValueError(f"steps must be between 1 and native horizon {model.output_steps}")
    model.eval()
    with torch.inference_mode():
        prediction = model(inputs)
        if prediction.shape != (inputs.shape[0], model.output_steps, 1, *model.img_size):
            raise ValueError("Invalid ConvLSTM output shape")
        if not torch.isfinite(prediction).all():
            raise ValueError("Model produced non-finite predictions")
    for lead in range(steps):
        yield prediction[:, lead]
