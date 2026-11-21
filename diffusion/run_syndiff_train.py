#!/usr/bin/env python
"""
Run SynDiff's train.py unmodified, with these changes made from outside:

1. No anomaly detection. train.py calls torch.autograd.set_detect_anomaly(True) inside its training loop,
   which slows every step a lot; here that function is a no-op.
2. SynDiff's EMA optimizer (utils/EMA.py) works on new PyTorch. It subclasses torch.optim.Optimizer without
   calling Optimizer.__init__, so attributes newer torch versions create there (the step / state_dict /
   load_state_dict hook dicts, defaults, ...) are missing and state_dict() fails at "Saving content".
   After EMA.__init__, every attribute the installed torch's Optimizer.__init__ sets and EMA lacks is added
   (found by building a throwaway Optimizer, not from a hard-coded list). EMA keeps sharing state and
   param_groups with the optimizer it wraps, so the EMA weights (state[p]['ema']) stay in state_dict(), and
   EMA.load_state_dict hands them back to the wrapped optimizer on resume.
3. Epoch timing and an optional stop. train.py calls train_sampler.set_epoch(epoch) at the start of every
   epoch; that call now logs a date timestamp and the previous epoch's wall-clock time. With
   SYNDIFF_STOP_AT_EPOCH=E in the environment, the run exits cleanly (code 0) before epoch E starts, i.e.
   after epoch E-1's checkpoints, content.pth and validation.

train.py then runs as __main__ with the same sys.argv, working directory and sys.path[0] as running it
directly.

Usage (from the SynDiff folder, exactly like `python train.py ...`):
  cd ~/SynDiff
  python -u ~/mri-t1-t2-translation/diffusion/run_syndiff_train.py train.py --image_size 256 ...
"""
import copy
import os
import runpy
import sys
import time
from datetime import datetime

import torch


def no_anomaly(*args, **kwargs):
    return None


def optimizer_init_attributes():
    """Instance attributes the installed torch's Optimizer.__init__ sets, read off a throwaway optimizer."""
    class Probe(torch.optim.Optimizer):
        pass
    return vars(Probe([torch.zeros(1, requires_grad=True)], {}))


def patch_ema(ema_cls):
    """Give every EMA instance the Optimizer attributes it is missing. Returns their names."""
    reference = optimizer_init_attributes()
    original_init = ema_cls.__init__

    def __init__(self, opt, ema_decay):
        original_init(self, opt, ema_decay)
        for name, value in reference.items():
            if name in self.__dict__:  # state, param_groups: shared with the wrapped optimizer, keep them
                continue
            if name == "defaults":
                self.defaults = opt.defaults
            elif name == "_zero_grad_profile_name":
                self._zero_grad_profile_name = f"Optimizer.zero_grad#{type(self).__name__}.zero_grad"
            else:  # empty hook OrderedDicts, flags
                setattr(self, name, copy.deepcopy(value))

    ema_cls.__init__ = __init__
    return sorted(n for n in reference if n not in ("state", "param_groups"))


def patch_epochs(stop_at):
    """Log each epoch start with a timestamp and the previous epoch's duration; stop before stop_at."""
    sampler_cls = torch.utils.data.distributed.DistributedSampler
    original_set_epoch = sampler_cls.set_epoch
    last = {}

    def set_epoch(self, epoch):
        now, stamp = time.time(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if last:
            print(f"[{stamp}] epoch {last['epoch']} took {now - last['start']:.0f} s "
                  "(training + saving + validation)", flush=True)
        if stop_at is not None and epoch >= stop_at:
            print(f"[{stamp}] stopping before epoch {epoch} (SYNDIFF_STOP_AT_EPOCH={stop_at})", flush=True)
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            raise SystemExit(0)
        print(f"[{stamp}] epoch {epoch} starting", flush=True)
        last.update(epoch=epoch, start=now)
        original_set_epoch(self, epoch)

    sampler_cls.set_epoch = set_epoch


def apply_patches():
    """Patches 1-3 above. SynDiff's folder must already be on sys.path (for utils.EMA).
    Also used by syndiff_train_sc.py, so both training scripts run with exactly the same patches."""
    torch.autograd.set_detect_anomaly = no_anomaly
    from utils.EMA import EMA  # SynDiff's; the training script later imports this same class
    added = patch_ema(EMA)
    stop = os.environ.get("SYNDIFF_STOP_AT_EPOCH")
    patch_epochs(int(stop) if stop else None)
    print(f"run_syndiff_train: torch {torch.__version__}, anomaly detection off, "
          f"EMA given Optimizer attributes {added}", flush=True)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_syndiff_train.py <SynDiff train.py> [train.py arguments]")
    script = sys.argv[1]
    sys.argv = sys.argv[1:]                                        # what `python train.py ...` would see
    sys.path[0] = os.path.dirname(os.path.abspath(script))        # the script's folder, not ours

    apply_patches()
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
