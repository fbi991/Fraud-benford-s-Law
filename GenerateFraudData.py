import pandas as pd
import numpy as np

# Génère un dataset normal (log-normal)
real = np.random.lognormal(mean=3, sigma=1, size=1000)
real_df = pd.DataFrame({'Montant': real})
real_df.to_csv("data/real_data.csv", index=False)

# Génère un dataset frauduleux (biaisé sur 9)
fraudulent = [int(f"9{np.random.randint(10, 99)}") for _ in range(1000)]
fraud_df = pd.DataFrame({'Montant': fraudulent})
fraud_df.to_csv("data/fraudulent_data.csv", index=False)

print("Fichiers générés : real_data.csv & fraudulent_data.csv")
