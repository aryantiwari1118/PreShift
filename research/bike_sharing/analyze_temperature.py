import pandas as pd


df = pd.read_csv(
    "../../research/data/bike_controlled_temperature_clean.csv"
)

print("Controlled Temperature Shift Analysis")
print("--------------------------------------")

for metric in [
    "ks_shift_score",
    "wasserstein_shift_score",
    "psi_shift_score",
    "mmd_score",
]:
    pearson = df[metric].corr(df["nrmse"])
    spearman = df[metric].corr(
        df["nrmse"],
        method="spearman"
    )

    print(
        f"{metric}: "
        f"Pearson={pearson:.4f}, "
        f"Spearman={spearman:.4f}"
    )

print()
print("Risk range:")
print(f"Minimum NRMSE: {df['nrmse'].min():.6f}")
print(f"Maximum NRMSE: {df['nrmse'].max():.6f}")

print()
print("Shift range:")

for metric in [
    "ks_shift_score",
    "wasserstein_shift_score",
    "psi_shift_score",
    "mmd_score",
]:
    print(
        f"{metric}: "
        f"{df[metric].min():.6f} -> "
        f"{df[metric].max():.6f}"
    )