import math
import os
import time
from datetime import datetime

import einx
import fire
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from einops.array_api import pack, rearrange, repeat, unpack
from mlx.utils import tree_flatten
from vocos_mlx import Vocos

class TxtEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(257, dim)
    def __call__(self, txt):
        return self.embed(txt+1)

class RoPE:
    def __init__(self, dim):
        inv_freq = 1. / (10000 ** (np.arange(0, dim, 2).astype('float32') / dim))
        self._inv_freq = inv_freq
    def __call__(self, seq_len):
        t = np.arange(seq_len, dtype='float32')
        freqs = np.outer(t, self._inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        cos = np.cos(emb)
        sin = np.sin(emb)
        return mx.array(cos), mx.array(sin)

class Attention(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head=n_head
        self.scale = (dim // n_head)**-0.5
        self.qkv_proj = nn.Linear(dim, 3*dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
    def __call__(self, x, mask, rope):
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        q, k = apply_rope(q, k, *rope)
        w = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        w += mask
        w = mx.softmax(w, axis=-1)
        o = w @ v
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        o = self.o_proj(o)
        return o

class MLP(nn.Module):
    def __init__(self, dim, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.gate_up_proj = nn.Linear(dim, 2*dim, bias=False)
        self.down_proj = nn.Linear(dim, out_dim, bias=False)
    def __call__(self, x):
        gate, x = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(nn.silu(gate) * x)

class AdaLNZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_gamma = nn.Linear(dim, dim)
    def __call__(self, x, condition):
        gamma = mx.sigmoid(self.to_gamma(condition))[:,None,:]
        return x * gamma

class AdaptiveRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.to_gamma = nn.Linear(dim, dim, bias=False)
    def __call__(self, x, condition):
        denom = mx.sqrt(mx.sum(x**2, axis=-1, keepdims=True)) + 1e-5
        normed = x / denom
        gamma = self.to_gamma(condition)[:,None,:]
        return normed * self.scale * (gamma + 1.)

class Fourier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(1, dim//2, bias=False)
    def __call__(self, times):
        times = times[:,None]
        freqs = self.linear(times) * 2 * mx.pi
        return mx.concatenate([times, freqs.sin(), freqs.cos()], axis=-1)

class ToWav:
    def __init__(self):
        self.to_wav = Vocos.from_pretrained("lucasnewman/vocos-mel-24khz")
    def __call__(self, mel, f_name='vocos'):
        wav = np.array(self.to_wav.decode(mel))
        sf.write(f'{f_name}.wav', wav, 24000)
        return wav

class Cross(nn.Module):
    def __init__(self, dim_mel, dim_txt, cond_audio_to_text):
        super().__init__()
        self.cond_audio_to_text = cond_audio_to_text
        self.text_to_audio = nn.Linear(dim_mel + dim_txt, dim_mel, bias=False)
        self.text_to_audio.weight = mx.zeros(self.text_to_audio.weight.shape)
        if cond_audio_to_text:
            self.audio_to_text = nn.Linear(dim_mel + dim_txt, dim_txt, bias=False)
            self.audio_to_text.weight = mx.zeros(self.audio_to_text.weight.shape)
    def __call__(self, x, txt):
        mel_txt = mx.concatenate([x, txt], axis=-1)
        txt_cond = self.text_to_audio(mel_txt)
        mel_cond = self.audio_to_text(mel_txt) if self.cond_audio_to_text else 0.
        return x + txt_cond, txt + mel_cond

class MelLayer(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.attn = Attention(dim=dim, n_head=n_head)
        self.mlp = MLP(dim)
        self.norm1 = AdaptiveRMSNorm(dim)
        self.norm2 = AdaptiveRMSNorm(dim)
        self.adaln1 = AdaLNZero(dim)
        self.adaln2 = AdaLNZero(dim)
    def __call__(self, x, times, mask, rope):
        attn_out = self.attn(self.norm1(x, condition=times), mask=mask, rope=rope)
        x = x + self.adaln1(attn_out, condition=times)
        ff_out = self.mlp(self.norm2(x, condition=times))
        x = x + self.adaln2(ff_out, condition=times)
        return x

class TxtLayer(nn.Module):
    def __init__(self, dim_mel, dim_txt, n_head, isnt_last):
        super().__init__()
        self.attn = Attention(dim=dim_txt, n_head=n_head)
        self.mlp = MLP(dim=dim_txt)
        self.norm1 = nn.RMSNorm(dim_txt)
        self.norm2 = nn.RMSNorm(dim_txt)
        self.cross = Cross(dim_mel = dim_mel, dim_txt = dim_txt, cond_audio_to_text = isnt_last)
    def __call__(self, x, txt, mask, rope):
        txt = txt + self.attn(self.norm1(txt), mask=mask, rope=rope)
        txt = txt + self.mlp(self.norm2(txt))
        x, txt = self.cross(x, txt)
        return x, txt

class MultiStream(nn.Module):
    def __init__(self, dim, n_head, depth, max_len=8192, num_registers=0):
        super().__init__()
        self.dim_txt = dim_txt = dim // 2
        self.depth = depth
        self.num_registers = num_registers
        if num_registers > 0:
            self.registers = mx.random.normal((num_registers, dim), scale=0.02)
            self.txt_registers = mx.random.normal((num_registers, dim_txt), scale=0.02)
        self.txt_layers = [TxtLayer(dim_mel=dim, dim_txt=dim_txt, n_head=n_head, isnt_last=bool(depth-d-1)) for d in range(depth)]
        self.mel_layers = [MelLayer(dim=dim, n_head=n_head) for d in range(depth)]
        self.skip_projs = [None if d < depth//2 else nn.Linear(dim*2, dim, bias=False) for d in range(depth)]
        self.final_norm = nn.RMSNorm(dim)
        self.time_cond_mlp = nn.Sequential(
            Fourier(dim),
            nn.Linear(dim+1, dim),
            nn.SiLU()
        )
        self._rope_txt = RoPE(dim_txt//n_head)
        self._rope_mel = RoPE(dim//n_head)
    def __call__(self, x, times, mask, txt):
        B, S, D = x.shape
        if self.num_registers > 0:
            mask = mx.pad(mask, ((0,0), (self.num_registers, 0)), constant_values=True)
            registers = repeat(self.registers, 'r d -> b r d', b = B)
            x, registers_packed_shape = pack((registers, x), 'b * d')
            txt_registers = repeat(self.txt_registers, 'r d -> b r d', b = B)
            txt, _ = pack((txt_registers, txt), 'b * d')
        mask = mx.where(mask[:, :, None]*mask[:, None, :]==1, 0, -mx.inf)
        mask = mx.expand_dims(mask, 1)
        rope_txt = self._rope_txt(S+self.num_registers)
        rope_mel = self._rope_mel(S+self.num_registers)
        times = self.time_cond_mlp(times)
        skips = []
        for i in range(self.depth):
            x, txt = self.txt_layers[i](x=x, txt=txt, mask=mask, rope=rope_txt)
            if i < self.depth//2:
                skips.append(x)
            else:
                skip = skips.pop()
                x = mx.concatenate((x, skip), axis=-1)
                x = self.skip_projs[i](x)
            x = self.mel_layers[i](x=x, times=times, mask=mask, rope=rope_mel)
        if self.num_registers > 0:
            _, x = unpack(x, registers_packed_shape, 'b * d')
        return self.final_norm(x)

class Transformer(nn.Module):
    def __init__(self, dim, n_head, depth, n_chan=100):
        super().__init__()
        self.i_proj = nn.Linear(n_chan * 2, dim)
        self.o_proj = nn.Linear(dim, n_chan)
        self.multistream = MultiStream(dim=dim, n_head=n_head, depth=depth)
        self.txt_embed = TxtEmbed(self.multistream.dim_txt)
    def __call__(self, x, cond, times, txt, mask):
        x = mx.concatenate((cond, x), axis=-1)
        x = self.i_proj(x)
        txt = self.txt_embed(txt)
        x = self.multistream(x, times=times, mask=mask, txt=txt)
        return self.o_proj(x)

class E2TTS(nn.Module):
    def __init__(self, dim=512, n_head=8, depth=8, stp=32, mxl=400, rnd=(0.7, 1.0)):
        super().__init__()
        self.mxl = mxl
        self.rnd = rnd
        self.stp = stp
        self.transformer = Transformer(dim=dim, n_head=n_head, depth=depth)
        self._wav = ToWav()
    def __call__(self, mel, txt):
        x1, mask, rand_mask = transpad(mel, mxl=self.mxl, rnd=self.rnd)
        txt = tokenize(txt, max_len=x1.shape[1])
        x0 = mx.random.normal(x1.shape)
        times = mx.random.randint(low=0, high=self.stp, shape=x1.shape[:1]) / self.stp
        t = times[:,None,None]
        flow = x1 - x0
        w = t*x1 + (1-t)*x0
        flow = flow * mask[...,None]
        w = w * mask[...,None]
        cond = einx.where('b n, b n d, b n d -> b n d', rand_mask, mx.zeros_like(x1), x1)
        pred = self.transformer(w, cond, times=times, txt=txt, mask=mask)
        ntok = rand_mask.sum()
        loss = ((flow - pred)**2) * rand_mask[:,:,None]
        loss = loss.sum() / ntok
        return loss, ntok
    def sample(self, mel, txt, f_name, len=364):
        mel, mask = transpad(mel, mxl=self.mxl, rnd=None)
        txt = tokenize(txt, max_len=mel.shape[1])
        batch, seq_len, _ = mel.shape
        pred_mask = mx.zeros(mask.shape, dtype=mx.bool_)
        pred_mask[:, :len] = True
        cond = einx.where('b n, b n d, b n d -> b n d', mask, mel, mx.zeros_like(mel))
        x = mx.random.normal(mel.shape)
        for i in range(self.stp):
            t = i / self.stp
            t_batch = mx.full((batch,), t)
            x = x * pred_mask[...,None]
            pred = self.transformer(x, cond, times=t_batch, txt=txt, mask=pred_mask)
            x = x + pred / self.stp
        out = einx.where('b n, b n d, b n d -> b n d', mask, mel, x)
        for i, o in enumerate(out):
            self._wav(o[None], f_name=f'{f_name}_{i}')
        return out.transpose(0, 2, 1)

def transpad(raw, mxl=None, rnd=None):
    raw = [arr.squeeze(0) for arr in raw]
    if mxl is None:
        mxl = max(arr.shape[1] for arr in raw)
    batch_size = len(raw)
    padded_arr = np.zeros((batch_size, mxl, raw[0].shape[0]))
    padding_mask = np.zeros((batch_size, mxl), dtype=bool)
    lens = np.array([min(arr.shape[1], mxl) for arr in raw])
    for i, arr in enumerate(raw):
        length = lens[i]
        padded_arr[i, :length] = arr.T[:length]
        padding_mask[i, :length] = True
    padded_arr = padded_arr.clip(0)
    if rnd is None:
        return mx.array(padded_arr), mx.array(padding_mask)
    frac_lengths = np.random.uniform(rnd[0], rnd[1], size=batch_size)
    lengths = (frac_lengths * lens).astype(int)
    max_start = lens - lengths
    start = (max_start * np.random.rand(batch_size)).astype(int)
    end = start + lengths
    rand_mask = np.zeros((batch_size, mxl), dtype=bool)
    for i in range(batch_size):
        rand_mask[i, start[i]:end[i]] = True
    rand_mask &= padding_mask
    return mx.array(padded_arr), mx.array(padding_mask), mx.array(rand_mask)

def get_ds(path_ds='JosefAlbers/cmu-arctic', max_len=400):
    ds = load_dataset(path_ds, split='aew').with_format('numpy')
    ds = ds.filter(lambda x: x['mel'].shape[-1] <= max_len)
    return ds

def log(f_name, *x):
    with open(f'{f_name}.log', 'a') as f:
        for i in x:
            print(i)
            f.write(f'{i}\n')

def plot_mel_spectrograms(original_mels, generated_mels, f_name='mel_comparison'):
    for i in range(len(original_mels)):
        original_mel = original_mels[i]
        generated_mel = generated_mels[i]
        original_mel = np.array(original_mel).squeeze()
        generated_mel = np.array(generated_mel).squeeze()
        generated_mel = generated_mel[:,:original_mel.shape[-1]]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        im1 = ax1.imshow(original_mel, aspect='auto', origin='lower', interpolation='nearest')
        ax1.set_title('Original Mel Spectrogram')
        ax1.set_ylabel('Mel Frequency Bin')
        fig.colorbar(im1, ax=ax1, format='%+2.0f dB')
        im2 = ax2.imshow(generated_mel, aspect='auto', origin='lower', interpolation='nearest')
        ax2.set_title('Generated Mel Spectrogram')
        ax2.set_xlabel('Time Frame')
        ax2.set_ylabel('Mel Frequency Bin')
        fig.colorbar(im2, ax=ax2, format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig(f'{f_name}_{i}.png')
        plt.close()

def tokenize(list_str, max_len):
    arr = [list(bytes(t, 'UTF-8')) for t in list_str]
    padded_arr = np.full((len(arr), max_len), -1, dtype=np.int64)
    for i, a in enumerate(arr):
        a = a[:max_len]
        padded_arr[i, :len(a)] = a
    return mx.array(padded_arr)

@mx.compile
def apply_rope(q, k, cos, sin):
    q1, q2 = mx.split(q, 2, axis=-1)
    rq = mx.concatenate([-q2, q1], axis = -1)
    k1, k2 = mx.split(k, 2, axis=-1)
    rk = mx.concatenate([-k2, k1], axis = -1)
    return q * cos + rq * sin, k * cos + rk * sin

def sample(model, example, f_name='sample'):
    mel = [e[...,:100] for e in example['mel']]
    txt = example['text']
    tic = time.perf_counter()
    model.eval()
    mx.eval(model)
    out = model.sample(mel, txt, f_name=f_name)
    plot_mel_spectrograms(example['mel'], out, f_name=f_name)
    print(f'Sampled ({time.perf_counter() - tic:.2f} sec)')

def train(model, dataset, batch_size, n_epoch, lr, postfix):
    args = {k: v for k, v in sorted(locals().items()) if k not in ['model', 'dataset']}
    def get_batch(dataset, batch_size):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            yield batch['mel'], batch['text']
    def get_optim():
        _n_steps = math.ceil(n_epoch * len(dataset) / batch_size)
        _n_warmup = _n_steps//5
        _warmup = optim.linear_schedule(1e-6, lr, steps=_n_warmup)
        _cosine = optim.cosine_decay(lr, _n_steps-_n_warmup, 1e-5)
        return optim.Lion(learning_rate=optim.join_schedules([_warmup, _cosine], [_n_warmup]))
    def evaluate(model, dataset, batch_size=32):
        model.eval()
        mx.eval(model)
        sum_loss = 0
        num_loss = 0
        for x, x_cls in get_batch(dataset, batch_size):
            loss, ntok = model(x, x_cls)
            sum_loss += loss * ntok
            num_loss += ntok
            mx.eval(sum_loss, num_loss)
        return (sum_loss / num_loss).item()
    def loss_fn(model, mel, txt):
        return model(mel, txt)
    f_name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{postfix}'
    log(f_name, args)
    example = dataset[:1]
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = get_optim()
    model.train()
    mx.eval(model, optimizer)
    state = [model.state, optimizer.state]
    best_avg_loss = mx.inf
    best_eval_loss = mx.inf
    for e in range(n_epoch):
        dataset = dataset.shuffle()
        sum_loss = 0
        num_loss = 0
        tic = time.perf_counter()
        for batch in get_batch(dataset, batch_size):
            model.train()
            (loss, ntok), grads = loss_and_grad_fn(model, *batch)
            optimizer.update(model, grads)
            mx.eval(loss, ntok, state)
            sum_loss += loss * ntok
            num_loss += ntok
        avg_loss = (sum_loss/num_loss).item()
        log(f_name, f'{avg_loss:.4f} @ {e} in {(time.perf_counter() - tic):.2f}')
        if e > n_epoch//5 and avg_loss < best_avg_loss:
            eval_loss = evaluate(model, dataset)
            log(f_name, f'- {eval_loss:.4f}')
            if eval_loss < best_eval_loss:
                log(f_name, '- Saved weights')
                model.save_weights(f'{f_name}.safetensors')
                best_eval_loss = eval_loss
                best_avg_loss = avg_loss
                sample(model=model, example=example, f_name=f_name)
    return f_name


def main(batch_size=32, n_epoch=200, lr=2e-4, depth=8, stp=1, postfix=''):
    model = E2TTS(depth=depth, stp=stp)
    dataset = get_ds()
    f_name = train(model=model, dataset=dataset, batch_size=batch_size, n_epoch=n_epoch, lr=lr, postfix=postfix)
    del model
    loaded = E2TTS(depth=depth, stp=stp)
    loaded.load_weights(f'{f_name}.safetensors')
    sample(model=loaded, example=dataset.shuffle()[:4], f_name=f'{f_name}_loaded')
    del loaded

if __name__ == '__main__':
    fire.Fire(main)
