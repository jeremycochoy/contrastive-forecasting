# Auto-imported at interpreter startup (when its dir is on PYTHONPATH).
# Registers an atexit hook that prints the true peak CUDA allocated/reserved
# memory per visible device — lets us read real headroom without editing the
# shared train.py. No effect on the run itself.
import atexit


def _report():
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                a = torch.cuda.max_memory_allocated(i) / 1e9
                r = torch.cuda.max_memory_reserved(i) / 1e9
                print(f"[memprobe] cuda:{i} max_allocated={a:.2f}GB "
                      f"max_reserved={r:.2f}GB", flush=True)
    except Exception as e:  # never break the run
        print(f"[memprobe] err {e}", flush=True)


atexit.register(_report)
