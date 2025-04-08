import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Datos extraídos de la conversación
data = {
    "Category": ["Apple_scab"] * 10 + ["Apple_healthy"] * 10 + ["Apple_rust"] * 5,
    "Color Ratio 1": [0.9389, 0.9423, 0.9544, 0.9096, 0.9073, 0.9298, 0.8929, 0.9655, 0.9278, 0.9255,
                      0.9536, 0.9383, 0.9411, 0.9494, 0.9491, 0.9362, 0.9489, 0.9617, 0.9166, 0.9838,
                      0.9435, 0.9464, 0.9490, 0.9466, 0.9807],
    "Color Ratio 2": [1.0472, 1.0159, 1.0273, 1.1066, 1.0580, 0.9737, 1.0209, 1.0064, 1.0109, 1.0084,
                      0.9904, 0.9705, 0.9667, 0.9875, 0.9852, 0.9510, 0.9944, 0.9858, 0.9573, 1.0125,
                      1.0485, 1.0835, 1.1770, 1.0056, 1.0301],
    "Color Ratio 3": [1.1152, 1.0781, 1.0764, 1.2166, 1.1661, 1.0472, 1.1434, 1.0424, 1.0896, 1.0896,
                      1.0385, 1.0344, 1.0272, 1.0402, 1.0380, 1.0157, 1.0480, 1.0251, 1.0444, 1.0292,
                      1.1113, 1.1448, 1.2403, 1.0623, 1.0503],
    "Sobel X": [769.27, 566.48, 388.55, 727.60, 487.53, 488.27, 630.70, 482.49, 479.30, 470.59,
                522.70, 518.26, 471.94, 385.02, 501.26, 452.89, 429.70, 524.58, 434.01, 402.03,
                480.26, 470.22, 635.56, 522.52, 386.94],
    "Sobel Y": [779.12, 552.08, 513.21, 688.10, 554.16, 412.45, 532.83, 474.50, 453.53, 545.89,
                441.62, 555.83, 542.45, 370.53, 473.97, 531.97, 393.12, 482.50, 441.66, 415.55,
                481.07, 529.29, 714.29, 428.18, 346.48],
    "Edge Density": [0.0313, 0.0167, 0.0062, 0.0198, 0.0121, 0.0092, 0.0087, 0.0145, 0.0081, 0.0048,
                     0.0131, 0.0107, 0.0104, 0.0109, 0.0108, 0.0088, 0.0082, 0.0129, 0.0063, 0.0116,
                     0.0099, 0.0112, 0.0154, 0.0120, 0.0161],
}

df = pd.DataFrame(data)

# Configurar visualización
sns.set(style="whitegrid")

# Comparación de Color Ratios por categoría
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(["Color Ratio 1", "Color Ratio 2", "Color Ratio 3"]):
    sns.boxplot(x="Category", y=col, data=df, ax=axes[i])
    axes[i].set_title(f"{col} by Category")

plt.show()

# Comparación de Sobel X & Y por categoría
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(x="Category", y="Sobel X", data=df, ax=axes[0])
sns.boxplot(x="Category", y="Sobel Y", data=df, ax=axes[1])
axes[0].set_title("Sobel X by Category")
axes[1].set_title("Sobel Y by Category")

plt.show()

# Comparación de Edge Density por categoría
plt.figure(figsize=(6, 5))
sns.boxplot(x="Category", y="Edge Density", data=df)
plt.title("Edge Density by Category")
plt.show()