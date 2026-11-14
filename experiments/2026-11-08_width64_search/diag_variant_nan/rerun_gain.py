"""rerun_step.py with one more record: the gradient at the input of each forecaster call (Nov 14).

For each call, the norm of each input token of the first forecaster layer (the tokens on the sphere) and the
gradient there. With the gradient at f before the sphere, it splits the gain of each call into the gain of the
forecaster layers (input gradient / gradient at f before the sphere) and the gain of the sphere and the feedback.

Re-run one training step of 'red without SIGReg-F' from a CF_NAN_DEBUG dump, with full tensors (read-only).

Usage (the run's train.py arguments, as start_rerun.sh builds them):
  CF_NAN_DEBUG=1 RERUN_DUMP=<dump.pt> RERUN_OUT=<out.pt> RERUN_MODE=train|eval RERUN_SEED=<int> RERUN_STEP=<step> \
    python3 -u rerun_step.py <train.py> <train.py arguments>

train.py's own main() builds the model with the run's flags. Inside NanDebug.__init__, before any training step,
this script loads the weights that the dumped step read, runs that step's batch through forward_step, the rollout
and the loss, then the backward. It records, for each forecaster call and each row, the norm of each token of f
before the sphere and the gradient there. For each RMS norm of an attention output, the RMS of each input token
and the backward gain of each token (|gradient in| / |gradient out|). It prints a summary, saves the records to
RERUN_OUT and exits: nothing trains, and no save of the run changes.
"""
import os
import runpy
import sys
from types import SimpleNamespace

import torch

import src.nan_debug as nd

DUMP, OUT = os.environ["RERUN_DUMP"], os.environ["RERUN_OUT"]
MODE, SEED = os.environ.get("RERUN_MODE", "train"), int(os.environ.get("RERUN_SEED", "0"))
STEP = int(os.environ.get("RERUN_STEP", "0"))
WEIGHTS = os.environ.get("RERUN_WEIGHTS", "dump")  # 'resume': keep the weights of the run's --resume save
REC = {"calls": [], "rms": [], "inputs": []}


def main_frame():
    """The frame of train.py's main(), which holds the model and the parsed arguments."""
    f = sys._getframe(1)
    while f is not None and not (f.f_code.co_name == "main" and {"model", "args"} <= set(f.f_locals)):
        f = f.f_back
    return f


def token_norms(t):
    """[rows, tokens]: the norm of each token (last axis) of a [rows, ..., K] tensor."""
    t = t.detach().float()
    return t.norm(dim=-1).reshape(t.shape[0], -1).cpu()


def on_forecast(module, inputs, output):
    """transformer.fcst_up_proj: its output is f before the sphere, once per forecaster call."""
    rec = {"raw_norm": token_norms(output), "raw_max": output.detach().abs().amax(dim=(1, 2)).float().cpu()}
    REC["calls"].append(rec)
    if output.requires_grad:
        output.register_hook(lambda g, r=rec: r.update(grad_norm=token_norms(g)))


def on_rms(name):
    """An RMS norm of an attention output: the RMS of each input token and the gradient on both sides."""
    def hook(module, inputs, output):
        x = inputs[0]
        rec = {"name": name, "rms": x.detach().float().pow(2).mean(-1).sqrt().reshape(x.shape[0], -1).cpu()}
        REC["rms"].append(rec)
        if x.requires_grad:
            x.register_hook(lambda g, r=rec: r.update(grad_in=token_norms(g)))
            output.register_hook(lambda g, r=rec: r.update(grad_out=token_norms(g)))
    return hook


def on_layer0_input(module, args, kwargs):
    """The first forecaster layer: its input is the input tokens of the call, on the sphere."""
    x = args[0] if args else kwargs["tgt"]
    rec = {"in_norm": token_norms(x)}
    REC["inputs"].append(rec)
    if x.requires_grad:
        x.register_hook(lambda g, r=rec: r.update(grad_in=token_norms(g)))


def add_hooks(model):
    hooks = [model.transformer.fcst_up_proj.register_forward_hook(on_forecast)]
    hooks.append(model.transformer.layers[0].register_forward_pre_hook(on_layer0_input, with_kwargs=True))
    hooks += [m.register_forward_hook(on_rms(n)) for n, m in model.named_modules() if n.endswith("attn_out_rms")]
    return hooks


def step_loss(model, args, g, batch):
    """The loss of the step as train.py computes it on its EMA path (forward_step, the rollout, contrastive_terms)."""
    device = next(model.parameters()).device
    x, fid, sid = (batch[k].to(device) for k in ("values", "freq_ids", "seasonality_ids"))
    want_raw = g["wants_raw_forecast"](args)
    res = g["forward_step"](model, x, freq_ids=fid, seasonality_ids=sid, want_teacher=True, want_embed=True,
                            want_raw=want_raw)
    res = list(res)
    f_raw = res.pop().float() if want_raw else None
    f_lat, o_lat, teacher, e_lat = (t.float() for t in res)
    rollout, rollout_raw = g["raw_rollout_depths"](model, f_lat, o_lat, args, g["zero_padding"](model))
    rep_w = g["linear_schedule_at_step"](STEP, args.total_steps, args.rep_loss_weight, args.rep_loss_weight_end,
                                         args.rep_loss_weight_ramp_steps)
    tau = model.tau().float() if args.learnable_tau else None
    with torch.amp.autocast("cuda", enabled=False):
        lat = SimpleNamespace(f=f_lat, o=o_lat, teacher=teacher, e=e_lat, rollout=rollout, f_raw=f_raw,
                              rollout_raw=rollout_raw)
        return g["contrastive_terms"](model, lat, args, rep_w, tau, g["contrastive_padding"](model, args))


def grad_report(model):
    norms = {n: float(p.grad.float().norm()) for n, p in model.named_parameters() if p.grad is not None}
    total = sum(v * v for v in norms.values()) ** 0.5
    top = sorted(norms.items(), key=lambda t: -t[1])[:6]
    print(f"[rerun] gradient norm {total:.6g}; largest: " + ", ".join(f"{n} {v:.4g}" for n, v in top))
    return total


def chain_report():
    print("[rerun] call  max|f raw|  min token |f raw|  largest |grad| of a token")
    for j, c in enumerate(REC["calls"]):
        g = c.get("grad_norm")
        print(f"[rerun] {j:4d}  {float(c['raw_max'].max()):9.3f}  {float(c['raw_norm'].min()):9.4f}  "
              + (f"{float(g.max()):.4g}" if g is not None else "no gradient"))


def rms_report():
    print("[rerun] RMS norm of an attention output: min input RMS, largest backward gain of a token")
    for r in REC["rms"]:
        gain = (r["grad_in"] / r["grad_out"].clamp_min(1e-30)) if "grad_in" in r and "grad_out" in r else None
        print(f"[rerun] {r['name']:45s} {float(r['rms'].min()):.4g}  "
              + (f"{float(gain.max()):.4g}" if gain is not None else "no gradient"))


def load_weights(model, weights):
    """The dump keeps every parameter under its named_parameters() name (the EMA teacher too). The buffers stay
    as the run's --resume loaded them."""
    params = dict(model.named_parameters())
    assert set(params) == set(weights), sorted(set(params) ^ set(weights))[:8]
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(weights[n])


def rerun(model, args, g):
    dump = torch.load(DUMP, map_location="cpu", weights_only=False)
    if WEIGHTS == "dump":
        load_weights(model, dump["weights_before_step"])
    model.train(MODE == "train")
    torch.manual_seed(SEED)
    hooks = add_hooks(model)
    loss, values = step_loss(model, args, g, dump["batch"])
    print(f"[rerun] step {STEP} weights {WEIGHTS} mode {MODE} seed {SEED} device {next(model.parameters()).device}: loss {float(loss):.6g} "
          f"(the dump: {dump['loss']:.6g}); terms {values.get('terms')}")
    loss.backward()
    for h in hooks:
        h.remove()
    REC["grad_total"] = grad_report(model)
    chain_report()
    rms_report()
    REC.update(loss=float(loss), freq_ids=dump["batch"]["freq_ids"], seasonality_ids=dump["batch"]["seasonality_ids"])
    torch.save(REC, OUT)
    print(f"[rerun] wrote {OUT}")


def nd_init(self, model, optimizer, dump_path, inject=None):
    f = main_frame()
    rerun(f.f_locals["model"], f.f_locals["args"], f.f_globals)
    sys.exit(0)


nd.NanDebug.__init__ = nd_init

if __name__ == "__main__":
    if os.environ.get("CF_NAN_DEBUG") != "1":
        raise SystemExit("rerun_step.py needs CF_NAN_DEBUG=1: it runs inside NanDebug.__init__")
    torch.set_num_threads(int(os.environ.get("RERUN_THREADS", "8")))
    train_py = sys.argv[1]
    sys.argv = [train_py] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(train_py)))
    runpy.run_path(train_py, run_name="__main__")
