# The flags of the paper ablations

Script: `experiments/2026-04-27_freq-embedding/scripts/train.py`.
Issue: #482. Table: `paper/sections/ablation.tex`.

Each ablation of the paper is the reference run with the one change of its
row. Five rows need a change in the code. Each new flag is off by default, so
the flags of an earlier run train the same model.

| flag | row of the table | what changes |
|---|---|---|
| `--dropout P` | Dropout of 0.1 in the blocks | The dropout of the blocks of the encoder and of the forecaster. Default 0.1, the value of every earlier run. DropKey is a separate flag. |
| `--zero-time-scale-embedding` | Time-scale embedding | The frequency and seasonality tables stay at 0 and do not train. |
| `--encoder-type timesfm` | GRU patch embedding | The residual block of TimesFM as the patch embedding. |
| `--meanstd-no-leak` | EWMA normalisation | Instance normalisation with no target leak (#475). Needs `--rev-norm-kind meanstd`. |
| `--ema-tau 0` | EMA teacher | The teacher takes the student weights after each step. |

## `--zero-time-scale-embedding`

After the init, the trainer sets the two tables to 0 and freezes them
(`zero_time_scale_embedding_`). A frozen table gets no gradient, so AdamW and
its weight decay leave it at 0. The model reads no time scale.

The tables keep their shapes. Thus the mixup runs as before, and the score
code loads the backbone with no change. With `--freq-emb-dim 0
--seasonality-emb-dim 0`, `maybe_mixup` skips the mixup, and the head trainer
cannot rebuild the backbone (it sets both sizes to 3).

## `--encoder-type timesfm`

`TimesFMResidualEncoder` in `src/encoders.py`, the `ResidualBlock` of
TimesFM v1 (`pytorch_patched_decoder.py`):

```
e = output_layer(SiLU(hidden_layer(x))) + residual_layer(x)
```

`x` is the input of the GRU encoder: the patch with its time-scale embedding,
read as one vector of W' numbers. The hidden width is the model width, as in
TimesFM. The block has no layer normalisation, as in TimesFM. The GRU encoder
has its layer normalisation in the module, so the row also removes it.
TimesFM also reads the padding mask of each patch, and this block does not,
as the GRU does not. The module names are the names of TimesFM, so the keys
of a checkpoint name the encoder.

## `--meanstd-no-leak`

With `--rev-norm-kind meanstd`, one mean and one standard deviation over a
context normalise the whole window. The uni2ts split draws the target, and
the contrastive terms then keep every position: an anchor in the context
reads statistics of its own target. The flag changes three things:

- `draw_short_context_ends` (`src/forecasting_head.py`) draws a context of
  1 patch to half of the whole real patches of the window, uniform in
  patches. It makes the same call on the random generator as the uni2ts draw.
- `step_start_mask` gives the context to the step inputs as `start_mask`.
  `group_padding` then masks each anchor whose next patch starts in the
  context, for every term: `L_align` at each distance, `L_rep` and its keys,
  and SIGReg.
- A window with fewer than two whole patches of real values has no target
  after its context: its split is at the end of the window.
  `no_target_windows` gives these windows to the step inputs as
  `no_target`, and `group_padding` masks each of their patches. Thus no
  term reads a latent of such a window, as an anchor, a key or a sample.

The checkpoint and the eval are those of `meanstd`. `--meanstd-z-max 0` turns
off the window filter of Moirai 2.0, which the instance normalisation does not
have.

## `--ema-tau 0`

The momentum 0 was refused before. At 0, `_ema_update` copies the student
weights into the teacher with no arithmetic. The teacher forward does not
change: it reads the full context, with no DropKey, no dropout and no
gradient. Thus the targets of `L_align` and the keys of `L_rep` are the
student latents of the full context, with stop-gradient (as SimSiam).
Without `--ema-tau-end`, the momentum stays at 0.

## The score code

The head trainer and the GIFT-Eval script build the frozen backbone from fixed
flags (`--encoder-type gru`) and a default convolution of kernel 3. They now
read the patch encoder (`encoder_type_of`) and the kernel of the convolution
(`depthwise_conv_of`, 0 when the blocks hold none) from the checkpoint, in
`src/checkpoint.py`. The checkpoint wins over the flags. Every checkpoint that
loaded before builds the same backbone.
