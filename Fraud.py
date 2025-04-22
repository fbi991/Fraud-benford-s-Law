import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

def benford_test(filepath, label):
    df = pd.read_csv(filepath)
    df["First_Digit"] = df["Montant"].astype(str).str.extract(r'([1-9])').astype(int)

    # On s'assure que les 9 chiffres sont tous représentés, même avec 0 occurrences
    observed = df["First_Digit"].value_counts().reindex(range(1, 10), fill_value=0)

    # Loi de Benford
    total = observed.sum()
    benford = [np.log10(1 + 1/d) for d in range(1, 10)]
    expected = [p * total for p in benford]

    # Test du Chi²
    chi2, p = chisquare(f_obs=observed, f_exp=expected)
    print(f"[{label}] p-value = {p:.4f}")

    return observed / total * 100, label  # Pour le graphe

# Analyse
obs_real, label_real = benford_test("data/real_data.csv", "Normal")
obs_fraud, label_fraud = benford_test("data/fraudulent_data.csv", "Frauduleux")

# Graphique de comparaison
benford = [np.log10(1 + 1/d) * 100 for d in range(1, 10)]
plt.figure(figsize=(10, 6))
plt.bar(range(1, 10), benford, alpha=0.3, label='Benford (théorique)')
plt.bar(obs_real.index, obs_real.values, alpha=0.6, label='Normal')
plt.bar(obs_fraud.index, obs_fraud.values, alpha=0.6, label='Frauduleux')
plt.title("Comparaison Loi de Benford - Données Normales vs Frauduleuses")
plt.xlabel("Premier chiffre")
plt.ylabel("Fréquence (%)")
plt.legend()
plt.savefig("Image/comparaison.png")
plt.show()

