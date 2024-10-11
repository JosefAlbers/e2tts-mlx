# e2tts-mlx: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS in MLX

A lightweight implementation of [Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS](https://arxiv.org/abs/2406.18009) model using MLX, with minimal dependencies and efficient computation on Apple Silicon.

## Quick Start

### Install

```zsh
pip install e2tts-mlx

# For the latest version, you can install directly from the repository:
# git clone https://github.com/JosefAlbers/e2tts-mlx.git
# cd e2tts-mlx
# pip install -e .
```

## Usage

To use a pre-trained model for text-to-speech:

```zsh
e2tts 'We must achieve our own salvation.'
```

This will write `tts_0.wav` to the current directory, which you can then play.

https://github.com/user-attachments/assets/89265113-0785-42f1-b867-c38c886acbef

To train a new model with default settings:

```zsh
e2tts
```

![e2tts](https://raw.githubusercontent.com/JosefAlbers/e2tts-mlx/main/assets/e2tts.png)

To train with custom options:

```zsh
e2tts --batch_size=16 --n_epoch=200 --lr=1e-4 --depth=8 --n_ode=32
```

Select training options:
- `--batch_size`: Set the batch size (default: 32)
- `--n_epoch`: Set the number of epochs (default: 200)
- `--lr`: Set the learning rate (default: 2e-4)
- `--depth`: Set the model depth (default: 8)
- `--n_ode`: Set the number of steps for sampling (default: 1)

## Acknowledgements

Thanks to [lucidrains](https://github.com/lucidrains/e2-tts-pytorch)' fantastic code that inspired this project.

## License

Apache License 2.0
