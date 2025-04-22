import pandas as pd
import numpy as np

# Génère un dataset normal (log-normal)
real = np.random.lognormal(mean=12, sigma=1, size=2025)
real_df = pd.DataFrame({'Montant': real})
real_df.to_csv("data/real_data.csv", index=False)

# Paramètres
n_samples = 2025
fraud_ratio = 0.7  # 70% de valeurs commençant par 9

# Génération des nombres en 9xxx (900-999)
n_fraud = int(n_samples * fraud_ratio)
fraudulent_9 = np.random.randint(900, 1000, size=n_fraud)

# Génération des autres nombres (100-899 et 1000-3000)
n_normal = n_samples - n_fraud
normal_values = np.concatenate([
    np.random.randint(100, 900, size=n_normal // 2),
    np.random.randint(1000, 3001, size=n_normal - n_normal // 2)
])

# Fusion et mélange
fraudulent = np.concatenate([fraudulent_9, normal_values])
np.random.shuffle(fraudulent)  # Mélange aléatoire

fraud_df = pd.DataFrame({'Montant': fraudulent})
fraud_df.to_csv("data/fraudulent_data.csv", index=False)

print("Les fichiers suivants ont été généré : real_data.csv & fraudulent_data.csv")
