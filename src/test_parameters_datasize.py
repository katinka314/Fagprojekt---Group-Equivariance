
# Test 2: data-efficiency ved matchet parameterantal
# Ét parameter-matchet model-par (CNN vs GE-CNN) trænes ved stigende
# datamængde. Ideen er at equivariante net behøver ikke lære rotations-
# invarians fra eksempler, så de forventes at vinde når data er knap
# og tabe (pga. ekstra compute) når data er rigeligt. Sweepet viser
# HVOR på data-aksen equivariansen begynder at betale sig.

import os
# På Windows linker både PyTorch og matplotlib hver sin libomp, og de kolliderer
# (OMP: Error #15) når matplotlib initialiserer ved plot-tidspunktet. Denne flag
# tillader at begge sameksisterer. Skal sættes FØR torch/matplotlib importeres.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

FAG_PROJEKT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAG_PROJEKT_DIR))

from models.Model import GE_CNN, CNN
from src.data import RotatedMNIST
from src.train import train_loop


# Hyperparametre
L               = 2
KERNEL_SIZE     = 5
N_EPOCHS        = 30
BATCH_SIZE      = 128
LR              = 1e-3
IMG_SIZE        = 28
N_CLASSES       = 10

# Data-efficiency sweep: vi matcher GE-CNN til ÉN CNN-størrelse, og træner så
# begge ved stigende træningsfraktion. Samme (faste) testsæt for alle fraktioner.
CNN_CHANNELS    = 16
TRAIN_FRACTIONS = [0.01, 0.02, 0.03, 0.05, 0.1]
# Evalueringen efter hver epoke er flaskehalsen (den langsomme GE-CNN kører forward
# på hele test-loaderen). 2.000 stratificerede billeder giver accuracy med ±~1%,
# rigeligt til kurverne, og gør hele sweepet flere gange hurtigere end fuldt testsæt.
TEST_FRACTION   = 0.1

ARCH_KWARGS = dict(
    kernel_size   = KERNEL_SIZE,
    img_size      = IMG_SIZE,
    n_classes     = N_CLASSES,
    n_conv_layers = 2,
    conv_pr_pool  = 1,
)

# n_rings er IKKE fast — find_matching_ge_config søger over den for at ramme
# CNN'ens parametertal, så GE-CNN'ens størrelse kan finjusteres.
GE_KWARGS  = dict(**ARCH_KWARGS)
CNN_KWARGS = dict(**ARCH_KWARGS)

# Testdata indlæses én gang (samme test for alle fraktioner).
test_dataset  = RotatedMNIST(split="test", rotated=True, fraction=TEST_FRACTION)
test_loader   = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


def count_params(model):
    """Kør dummy forward så LazyLinear initialiseres, tæl derefter parametre."""
    dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
    model(dummy)
    return sum(p.numel() for p in model.parameters())


def find_matching_ge_config(target_params,
                            channel_opts=None,
                            nring_range=range(2, KERNEL_SIZE + 1)):
    """Find (channels, n_rings) der giver tættest match på target_params.

    n_rings cappes ved KERNEL_SIZE: en kxk-kerne har kun k distinkte radier
    (for 5x5: radierne {0,1,2,4,5,8} -> 6 stk), så flere ringe end det lægger
    blot overlappende Gaussians på de samme pixels. det er redundant.

    channel_opts begrænses til toer-potenser <= CNN_CHANNELS: en GE-CNN med flere
    kanaler end CNN'en har altid for mange parametre (og en 128-kanals GE-CNN er
    dyr at bygge, ~4M params), så det er spild at søge dem."""
    if channel_opts is None:
        channel_opts = [c for c in (2, 4, 8, 16, 32, 64, 128) if c <= CNN_CHANNELS]
    best = None  # (diff, channels, n_rings, n_params)
    for ch in channel_opts:
        for nr in nring_range:
            ge = GE_CNN(l=L, channels=ch, n_rings=nr, **GE_KWARGS)
            n = count_params(ge)
            diff = abs(n - target_params)
            if best is None or diff < best[0]:
                best = (diff, ch, nr, n)
    return best  # diff, channels, n_rings, n_params


# Find det parameter-matchede par ÉN gang (uafhængigt af datamængde).
cnn_ref     = CNN(channels=CNN_CHANNELS, **CNN_KWARGS)
target_params = count_params(cnn_ref)
param_diff, ge_channels, ge_n_rings, ge_params = find_matching_ge_config(target_params)

print("=== Parameter-matchet par ===")
print(f"CNN channels={CNN_CHANNELS}  ->  {target_params:,} parametre")
print(f"GE-CNN channels={ge_channels}, n_rings={ge_n_rings}  ->  {ge_params:,} parametre")
print(f"forskel: {param_diff:,} ({param_diff/target_params:.1%})\n")


# Sweep over datamængde
results = []

for frac in TRAIN_FRACTIONS:
    train_dataset = RotatedMNIST(split="train", rotated=True, fraction=frac)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    n_train = len(train_dataset)

    # Friske modeller for hver fraktion (ellers ville de fortsætte hvor de slap).
    ge  = GE_CNN(l=L, channels=ge_channels, n_rings=ge_n_rings, **GE_KWARGS)
    cnn = CNN(channels=CNN_CHANNELS, **CNN_KWARGS)

    print(f"--- fraction={frac:.0%}  ({n_train} træningsbilleder) ---")
    ge_history  = train_loop(ge,  lr=LR, train_loader=train_loader, n_epochs=N_EPOCHS, test_loader=test_loader, show_progress=False)
    cnn_history = train_loop(cnn, lr=LR, train_loader=train_loader, n_epochs=N_EPOCHS, test_loader=test_loader, show_progress=False)

    results.append({
        "fraction":         frac,
        "n_train":          n_train,
        "ge_acc":           ge_history[1][1][-1],
        "cnn_acc":          cnn_history[1][1][-1],
        "ge_loss":          ge_history[1][0][-1],
        "cnn_loss":         cnn_history[1][0][-1],
        "ge_test_accs":     ge_history[1][1],
        "cnn_test_accs":    cnn_history[1][1],
        "ge_test_losses":   ge_history[1][0],
        "cnn_test_losses":  cnn_history[1][0],
    })


# Resultat tabel
print("\n=== Test 2: Data-efficiency (matchet parameterantal) ===")
print(f"CNN: {target_params:,} par  |  GE-CNN: {ge_params:,} par (ch={ge_channels}, rings={ge_n_rings})")
print(f"{'fraction':>9} {'n_train':>9} {'GE acc':>10} {'CNN acc':>10} {'GE-CNN diff':>12}")
print("-" * 54)
for r in results:
    diff = r["ge_acc"] - r["cnn_acc"]
    print(f"{r['fraction']:>9.0%} {r['n_train']:>9} {r['ge_acc']:>10.2%} {r['cnn_acc']:>10.2%} {diff:>+12.2%}")

# Gem
RESULTS_PATH = FAG_PROJEKT_DIR / "results" / "test_parameters_dataeff_results.pt"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save({"results": results, "cnn_params": target_params, "ge_params": ge_params,
            "ge_channels": ge_channels, "ge_n_rings": ge_n_rings}, RESULTS_PATH)
print(f"\nResultater gemt til {RESULTS_PATH}")


# Visualisering
def plot_results(results, save_dir=FAG_PROJEKT_DIR / "reports"):
    """Tegn (1) test-accuracy vs. datamængde for begge modeller, og
    (2) GE-CNN's fordel (acc-forskel) vs. datamængde, så krydsningspunktet
    hvor equivariansen holder op med at betale sig bliver tydeligt."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rs = sorted(results, key=lambda r: r["n_train"])
    n_train  = [r["n_train"]  for r in rs]
    ge_accs  = [r["ge_acc"]   for r in rs]
    cnn_accs = [r["cnn_acc"]  for r in rs]
    advantage = [g - c for g, c in zip(ge_accs, cnn_accs)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Accuracy vs. datamængde.
    ax1.plot(n_train, ge_accs,  "o-", label=f"GE-CNN ({ge_params:,}p)", color="tab:blue")
    ax1.plot(n_train, cnn_accs, "s-", label=f"CNN ({target_params:,}p)", color="tab:orange")
    ax1.set_xscale("log")
    ax1.set_xlabel("Antal træningsbilleder (log-skala)")
    ax1.set_ylabel("Test-accuracy (roteret)")
    ax1.set_title("Accuracy vs. datamængde")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # (2) GE-CNN's fordel — positiv = GE bedst, negativ = CNN bedst.
    ax2.axhline(0, color="black", lw=1)
    ax2.plot(n_train, advantage, "o-", color="tab:green")
    ax2.fill_between(n_train, advantage, 0, where=[a >= 0 for a in advantage], alpha=0.2, color="tab:green", label="GE-CNN bedst")
    ax2.fill_between(n_train, advantage, 0, where=[a < 0 for a in advantage], alpha=0.2, color="tab:red", label="CNN bedst")
    ax2.set_xscale("log")
    ax2.set_xlabel("Antal træningsbilleder (log-skala)")
    ax2.set_ylabel("GE-CNN accuracy − CNN accuracy")
    ax2.set_title("Equivariansens fordel vs. datamængde")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("Test 2: Data-efficiency with matched number of parameters", fontsize=13)
    fig.tight_layout()

    fig_path = save_dir / "test_parameters_data_efficiency.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figur gemt til {fig_path}")
    plt.show()


plot_results(results)
