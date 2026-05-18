# Remote-launch preflight

Before launching ANY training on a remote (vast.ai) instance:

- [ ] **Data-center host, on-demand price.** `vastrun-search` without `--prosumer`, `vastrun-provision` without `--spot`. No consumer / spot hosts.
- [ ] `sync_loop.sh` is running on elisa pulling FROM the remote, AND its first tick has landed the expected files (backbone `.pth`, optimizer `.pth`, head `.pth` if any, losses CSV, run log) in `sync_<run>/checkpoints/` — **verified by `ls`**, not by reading `sync.log` alone.

You can't proceed until the synchronisation is working properly and guaranteed.
