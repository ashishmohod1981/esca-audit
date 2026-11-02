"""Image panels used inside the architecture diagrams (Figs. 1 to 3).

Stage 1 panels run the real preprocessing of the pipeline (specular mask, Telea inpainting,
CLAHE on the LAB lightness channel, centered crop, 14 views) on one deidentified white light frame.
Stage 2 panels use an axial slice of the public 3D Slicer chest CT sample, a PET overlay whose FDG
uptake is simulated for illustration, and an illustrative genomic heatmap. No patient data from
the study cohorts are shown.

Usage (from the repository root):
    python scripts/make_diagram_assets.py --frame path/to/deidentified_frame.png \
        --ct path/to/CTChest.nrrd
The panels are written to figures/assets/. The esophagus position (column 276, row 277 of slice 84)
refers to the 3D Slicer CTChest sample; adjust it if another volume is used.
"""
import argparse
import os, json
import numpy as np, cv2
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom, binary_fill_holes, label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--frame", required=True, help="deidentified white light frame (black outside the circular field)")
ap.add_argument("--ct", required=True, help="chest CT volume (.nrrd) or a saved numpy array (.npy, x by y by z)")
ap.add_argument("--slice", type=int, default=84)
args = ap.parse_args()
OUT = os.path.join(ROOT, "figures", "assets"); os.makedirs(OUT, exist_ok=True)
meta = {}

def save(arr, name, size=None):
    im = Image.fromarray(arr)
    if size: im = im.resize(size, Image.LANCZOS)
    im.save(os.path.join(OUT, name))

# =============================================================== Stage 1 (white-light image)
clean = cv2.imread(args.frame)                                          # deidentified, circular field of view
h, w = clean.shape[:2]
# (0) raw frame for display: square canvas around the field of view
side = max(h, w); canvas = np.zeros((side, side, 3), np.uint8)
y0, x0 = (side - h) // 2, (side - w) // 2; canvas[y0:y0 + h, x0:x0 + w] = clean
save(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), "s1_raw.png", (600, 600))

# (1) specular highlight mask (pipeline rule: grey > 240)
gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
_, spec = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
meta["specular_fraction"] = float((spec > 0).mean())
vis = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB).copy()
dil = cv2.dilate(spec, np.ones((9, 9), np.uint8)) > 0
vis[dil] = (0.25 * vis[dil] + 0.75 * np.array([42, 157, 143])).astype(np.uint8)
c2 = np.zeros((side, side, 3), np.uint8); c2[y0:y0 + h, x0:x0 + w] = vis
save(c2, "s1_specular.png", (600, 600))

# (2) inpainting + LAB-CLAHE (identical parameters to the extraction code)
inp = cv2.inpaint(clean, spec, 3, cv2.INPAINT_TELEA)
lab = cv2.cvtColor(inp, cv2.COLOR_BGR2LAB)
lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
# (3) centred square crop -> 224 x 224
s = min(h, w); ys, xs = (h - s) // 2, (w - s) // 2
crop = rgb[ys:ys + s, xs:xs + s]
g224 = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
save(g224, "s1_global.png", (448, 448))
c3 = np.zeros((side, side, 3), np.uint8); c3[y0:y0 + h, x0:x0 + w] = rgb
c3 = cv2.rectangle(c3.copy(), (x0 + xs, y0 + ys), (x0 + xs + s, y0 + ys + s), (255, 255, 255), 6)
save(c3, "s1_clahe_crop.png", (600, 600))

# (4) the 14 views exactly as in the pipeline (2x2 = 112 px, 3x3 = 74 px, each resized to 224)
views = {"v00": g224}
for rr in range(2):
    for cc in range(2):
        views[f"q{rr}{cc}"] = g224[rr * 112:(rr + 1) * 112, cc * 112:(cc + 1) * 112]
for rr in range(3):
    for cc in range(3):
        views[f"t{rr}{cc}"] = g224[rr * 74:(rr + 1) * 74, cc * 74:(cc + 1) * 74]
for k, v in views.items():
    save(cv2.resize(v, (224, 224), interpolation=cv2.INTER_LINEAR), f"s1_{k}.png")

# (5) ViT-B/14 patch grid on the global view (224 / 14 = 16 patches per side)
pg = cv2.resize(g224, (448, 448), interpolation=cv2.INTER_LINEAR).copy()
for i in range(1, 16):
    pg[:, i * 28 - 1:i * 28 + 1] = (255 * 0.6 + pg[:, i * 28 - 1:i * 28 + 1] * 0.4).astype(np.uint8)
    pg[i * 28 - 1:i * 28 + 1, :] = (255 * 0.6 + pg[i * 28 - 1:i * 28 + 1, :] * 0.4).astype(np.uint8)
save(pg, "s1_patches.png")

# (6) measured attention pattern (TransMIL comparator, BE test images; relative to uniform)
att = np.array([[0.57, 1.17, 0.72], [0.83, 1.42, 1.07], [0.95, 1.04, 1.16]])
base = cv2.resize(g224, (450, 450), interpolation=cv2.INTER_LINEAR).astype(float)
heat = np.kron((att - att.min()) / (att.max() - att.min()), np.ones((150, 150)))
coral = np.array([231, 111, 81], float)
over = base * (1 - 0.55 * heat[..., None]) + coral * 0.55 * heat[..., None]
over = over.clip(0, 255).astype(np.uint8)
for i in (150, 300):
    over[:, i - 1:i + 1] = 255; over[i - 1:i + 1, :] = 255
save(over, "s1_attention.png")
meta["attention_3x3_BE"] = att.tolist()

# (7) instance-embedding matrix (illustrative values, 14 views x 48 shown dimensions)
rng = np.random.default_rng(7)
E = rng.normal(0, 1, (14, 48)) + np.linspace(-0.6, 0.6, 48)[None, :] * rng.normal(0, 1, (14, 1))
E = (E - E.min()) / (E.max() - E.min())
lo, hi = np.array([252, 233, 227], float), np.array([181, 64, 38], float)
img = (lo + (hi - lo) * E[..., None]).astype(np.uint8)
save(np.kron(img, np.ones((12, 12, 1), np.uint8)), "s1_embed.png")

# =============================================================== Stage 2 (CT, PET, genomics)
if args.ct.endswith(".npy"):
    vol = np.load(args.ct).astype(float)
else:
    import nrrd                                                         # pip install pynrrd
    vol = nrrd.read(args.ct)[0].astype(float)
z = args.slice
sl = vol[:, :, z].T                      # anterior up, patient right on viewer's left
body = binary_fill_holes(sl > -400)
lab_, n = label(body); body = lab_ == (1 + np.argmax(np.bincount(lab_.ravel())[1:]))
yy, xx = np.nonzero(body)
pad = 6
r0, r1, c0, c1 = yy.min() - pad, yy.max() + pad, xx.min() - pad, xx.max() + pad
cs = sl[r0:r1, c0:c1]
ctw = np.clip((cs + 160) / 400, 0, 1)    # mediastinal window W400 / L40
ct8 = (ctw * 255).astype(np.uint8)
H, W = ct8.shape
esoph = (276 - c0, 277 - r0)             # esophagus centre (column, row) in the cropped slice
meta["ct_slice"] = z; meta["ct_crop_hw"] = [H, W]; meta["esophagus_xy_frac"] = [esoph[0] / W, esoph[1] / H]
save(np.stack([ct8] * 3, -1), "s2_ct.png", (int(W * 1.6), int(H * 1.6)))

# PET: simulated FDG uptake (illustrative). SUV-like map from tissue classes plus an esophageal focus.
suv = np.zeros_like(cs)
bm = body[r0:r1, c0:c1]
suv[bm] = 0.7
suv[bm & (cs < -500)] = 0.35                                  # lung
suv[bm & (cs > 20) & (cs < 90)] = 1.6                          # blood pool / soft tissue
suv[bm & (cs > 150)] = 1.2                                     # bone marrow
Y, X = np.mgrid[:H, :W]
suv += 8.0 * np.exp(-(((X - esoph[0]) ** 2 + (Y - esoph[1]) ** 2) / (2 * 8.0 ** 2)))
suv = gaussian_filter(suv, 3.0)
suv = zoom(zoom(suv, 0.2, order=1), (H / round(H * 0.2), W / round(W * 0.2)), order=1)[:H, :W]
suv = np.clip(suv + rng.normal(0, 0.08, suv.shape) * np.sqrt(np.maximum(suv, 0)), 0, None)
suv[~bm] = 0
cm = cv2.applyColorMap((np.clip(0.25 + suv / 9.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_HOT)[..., ::-1].astype(float)
alpha = np.clip((suv - 1.2) / 2.0, 0, 0.92)[..., None]
fused = (np.stack([ct8] * 3, -1) * (1 - alpha) + cm * alpha).astype(np.uint8)
save(fused, "s2_pet.png", (int(W * 1.6), int(H * 1.6)))
meta["pet_note"] = "simulated FDG uptake for illustration only"

# Genomic profile: illustrative heatmap of 1,500 features shown as a 30 x 50 grid of z-scores
Gz = rng.normal(0, 1, (30, 50))
Gz[4:9, 6:20] += 1.6; Gz[17:22, 28:44] -= 1.5; Gz[24:28, 2:12] += 1.2
Gz = gaussian_filter(Gz, 0.6)
Gn = np.clip((Gz + 2.5) / 5, 0, 1)
neg, mid, pos = np.array([38, 70, 83], float), np.array([247, 243, 236], float), np.array([231, 111, 81], float)
col = np.where(Gn[..., None] < 0.5, neg + (mid - neg) * (Gn[..., None] / 0.5), mid + (pos - mid) * ((Gn[..., None] - 0.5) / 0.5))
save(np.kron(col.astype(np.uint8), np.ones((10, 10, 1), np.uint8)), "s2_geno.png")

# ---- small panels for Fig. 1 -----------------------------------------------------------------
# duplicate audit: the global view and an augmented copy (rotation and brightness change)
save(g224, "s1_dup_orig.png")
M = cv2.getRotationMatrix2D((112, 112), 12, 1.22)
aug = cv2.warpAffine(g224, M, (224, 224), borderMode=cv2.BORDER_REFLECT)
aug = np.clip(aug.astype(float) * 1.10 + 6, 0, 255).astype(np.uint8)
save(aug, "s1_dup_aug.png")
# the global view with the quadrant and tile grids drawn on it
for name, k in (("s1_quadmosaic.png", 2), ("s1_tilemosaic.png", 3)):
    m = g224.copy()
    for i in range(1, k):
        c = int(round(i * 224 / k))
        m[:, c - 2:c + 2] = 255
        m[c - 2:c + 2, :] = 255
    save(m, name)

json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print(json.dumps(meta, indent=1))
