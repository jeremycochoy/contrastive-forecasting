# Remote-launch preflight

Before launching ANY training on a remote (vast.ai) instance:

- [ ] `sync_loop.sh` is running on elisa pulling FROM the remote, AND its first tick has landed the expected files (backbone `.pth`, optimizer `.pth`, head `.pth` if any, losses CSV, run log) in `sync_<run>/checkpoints/` — **verified by `ls`**, not by reading `sync.log` alone.

If you can't tick this, fix it or destroy the instance and retry. DONE-marker scp-backs are not a substitute (lost the τ=0.20 trajectory CSV this way on 2026-05-08).
