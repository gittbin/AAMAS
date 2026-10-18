"""Export development per-policy intervention plots; no final-data selection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "outputs_marl/paper_v4/diagnostics/report"
    data = json.loads((folder / "results.json").read_text())
    rows = data["per_seed"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, key, title in zip(axes, ("normal_minus_off", "normal_minus_zero"),
                                ("Messages on minus off (including cost)", "Normal minus zero payload (fixed sends)")):
        axis.axhline(0, color="black", linewidth=0.8)
        axis.scatter([r["seed"] for r in rows], [r[key] for r in rows], color="#226f9e", s=45)
        axis.set_xticks([r["seed"] for r in rows])
        axis.set_xlabel("Training seed")
        axis.set_ylabel("Mean net-return difference")
        axis.set_title(title, fontsize=11)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Development diagnostics: 8 locked policies, 30 shared scenarios", fontsize=12)
    fig.savefig(folder / "message_diagnostics.pdf")
    fig.savefig(folder / "message_diagnostics.png", dpi=160)
    plt.close(fig)
    print(folder / "message_diagnostics.pdf")


if __name__ == "__main__":
    main()
