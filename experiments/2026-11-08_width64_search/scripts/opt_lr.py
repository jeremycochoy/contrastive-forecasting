"""Prints the learning rate of each param group and the lr_schedule of a training optimizer file (CPU)."""
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("keys:", sorted(state) if isinstance(state, dict) else type(state))
opt = state.get("optimizer", state) if isinstance(state, dict) else state
groups = opt["param_groups"] if "param_groups" in opt else opt.get("optimizer_state_dict", {}).get("param_groups", [])
for g in groups:
    print("group", g.get("name", "?"), "lr", g.get("lr"), "initial_lr", g.get("initial_lr"), "lr_mult", g.get("lr_mult"))
for k in ("lr_schedule", "step", "global_step"):
    if isinstance(state, dict) and k in state:
        print(k, state[k])
