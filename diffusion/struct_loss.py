"""
Position-preserving losses for SynDiff, used by syndiff_train_sc.py (--struct_loss silhouette | mind).

Each loss compares a generator output with that generator's own source image (never with a target of the
other contrast), so it is legal for unpaired training. Inputs are SynDiff tensors on its -1..1 scale,
shape (B, 1, H, W); everything below works on the 0-1 scale of our PNGs. src_real says whether the source
is a real slice (background exactly 0 after preprocess.py) or another generator's output (background
only close to 0).

silhouette  L1 between a soft brain mask of the output, sigmoid((x - THRESH) / TEMP), and the source's
            brain mask: source > 0 for a real source, the same soft mask for a generated one. The sigmoid
            tails keep the aligned value slightly above 0 (about 0.006 x the background fraction).
            Gradients reach only pixels near THRESH, i.e. the brain outline: this term fixes the outline's
            position, not the anatomy inside it.
mind        L1 between MIND descriptors (Heinrich et al., MedIA 2012) of output and source, averaged over
            the source's brain mask. Settings from Yang et al. (DLMIA 2018, structure-constrained CycleGAN
            for brain MR -> CT): 9 x 9 non-local region, 7 x 7 patch weighted by a Gaussian with sigma 2,
            variance from the 4-neighbourhood, L1 distance. MIND compares each patch with its neighbours in
            the same image, so it does not change under an intensity scaling or inversion of the whole image
            (T1 <-> T2 is only approximately that), but it does change when structures move.
"""
import torch
import torch.nn.functional as F

THRESH = 0.03   # brain = above this, 0-1 scale (about 8/255)
TEMP = 0.006    # sigmoid width: the soft mask goes from 0.12 to 0.88 between 0.018 and 0.042

MIND_REGION = 9  # non-local search region, pixels per side
MIND_PATCH = 7   # patch size, pixels per side
MIND_SIGMA = 2.0


def to01(x):
    return (x + 1) / 2


def soft_mask(x01):
    return torch.sigmoid((x01 - THRESH) / TEMP)


def silhouette_loss(out, src, src_real):
    o01, s01 = to01(out), to01(src)
    target = (s01 > 0).float() if src_real else soft_mask(s01)
    return (soft_mask(o01) - target).abs().mean()


def _gaussian(size, sigma, device, dtype):
    r = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-r ** 2 / (2 * sigma ** 2))
    k = g[:, None] * g[None, :]
    return k / k.sum()


def mind(x01):
    """MIND descriptor, (B, 1, H, W) -> (B, MIND_REGION**2 - 1, H, W), each pixel's maximum is 1."""
    b, _, h, w = x01.shape
    n, r, p = MIND_REGION ** 2, MIND_REGION // 2, MIND_PATCH // 2
    # channel k holds the image translated by the k-th offset of the non-local region (I'(alpha) in Yang et al.)
    translated = F.unfold(F.pad(x01, (r, r, r, r), mode="replicate"), MIND_REGION).view(b, n, h, w)
    sq = (translated - x01) ** 2
    kernel = _gaussian(MIND_PATCH, MIND_SIGMA, x01.device, x01.dtype).expand(n, 1, MIND_PATCH, MIND_PATCH)
    dist = F.conv2d(F.pad(sq, (p, p, p, p), mode="replicate"), kernel, groups=n)  # patch distance D_P per offset
    centre = n // 2
    four = [centre - MIND_REGION, centre - 1, centre + 1, centre + MIND_REGION]
    var = dist[:, four].mean(1, keepdim=True)                                        # V(I, x)
    # keep V within 1e-3..1e3 of the image's mean V (as in Heinrich's code), so flat regions do not divide by 0
    mean_var = var.mean((1, 2, 3), keepdim=True).detach().clamp(min=1e-10)
    var = torch.maximum(torch.minimum(var, mean_var * 1e3), mean_var * 1e-3)
    keep = [k for k in range(n) if k != centre]                                      # the zero offset carries no information
    desc = torch.exp(-dist[:, keep] / var)
    return desc / desc.amax(1, keepdim=True)                                         # 1/Z: maximum component 1


def mind_loss(out, src, src_real):
    o01, s01 = to01(out), to01(src)
    mask = ((s01 > 0) if src_real else (s01 > THRESH)).float()
    diff = (mind(o01) - mind(s01)).abs().mean(1, keepdim=True)
    return (diff * mask).sum() / mask.sum().clamp(min=1)


LOSSES = {"silhouette": silhouette_loss, "mind": mind_loss}


def terms(kind, pairs):
    """{name: (output, its source, source is real)} -> {name: loss}, for --struct_loss kind."""
    fn = LOSSES[kind]
    return {name: fn(out, src, real) for name, (out, src, real) in pairs.items()}
