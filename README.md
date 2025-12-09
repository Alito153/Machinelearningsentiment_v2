Parfait.
🔵 **On part sur le choix A : JE RÉÉCRIS TON README INTÉGRALEMENT, À L’IDENTIQUE, AVEC TOUT LE CONTENU, TOUTES LES SECTIONS, TOUTES LES ÉQUATIONS, TOUTES LES TABLES — mais en format GitHub + MathJax.**

🔵 Ce sera livré proprement en **7 parties** (sinon ça dépasse les limites techniques).
🔵 Chaque partie sera **parfaitement formatée**, prête à être collée dans `README.md`.

---

# ✅ **README — PARTIE 1 / 7**

*(Introduction + structure du projet + premières formules + début section 5)*

---

# Trading Strategy Project

## Structure du Projet

* **data/** : Toutes les données (raw, processed, config)
* **models/** : Modèles ML et configurations
* **analysis/** : Rapports, visualisations et métriques
* **backtesting/** : Résultats de backtesting et stress testing
* **scripts/** : Tous les scripts Python organisés par fonction
* **docs/** : Documentation du projet

## Notes

Projet réorganisé le **2025-12-08 00:12:05**
Structure nettoyée et consolidée depuis plusieurs projets imbriqués.

---

# Critère d'arrêt : Minimum Backtest Length (MinBTL)

Pour un Sharpe Ratio cible :

[
SR_{\text{target}} = 1.4
]

et ( N ) configurations testées :

[
\text{MinBTL} \approx \frac{2 \ln(N)}{SR_{\text{target}}^2}
]

Donc :

[
\text{MinBTL} \approx \frac{2\ln(N)}{1.96^2}
\approx 1.02 \times \ln(N) \text{ années}
]

**Exemple :**

Avec ( N = 200 ):

[
\text{MinBTL} \approx 1.02 \times \ln(200) \approx 5.3 \text{ années}
]

Donc **OK** pour les 7 ans de données (2018-2025).

---

# 5. Modélisation Deep Learning — Prédiction de Volatilité

## 5.1 Motivation et Fondements Empiriques

La volatilité intrajournalière du Forex présente des patterns récurrents (Liao, Chen & Ni, 2021) :

1. **Saisonnalité intrajournalière**

   * Pic Londres (7h UTC)
   * Pic New York (12h UTC)
   * Spikes d’annonces macro (CPI, PPI, NFP)

2. **Auto-corrélation temporelle**

   * **Intra-jour** : clustering sur 20 minutes
   * **Inter-jours** : corrélation à la même minute sur 20 jours

3. **Corrélations croisées entre paires**

   * EURUSD ↔ USDJPY
   * XAUUSD ↔ US indices

---

## 5.2 Architecture du Modèle LSTM Multi-Échelles

### A. Définition du Log-Range (variable cible)

Pour une fenêtre ( \tau ) :

[
\text{LogRange}*t
= \ln\left(\max*{t \le s \le t+\tau} P_s\right)

* \ln\left(\min_{t \le s \le t+\tau} P_s\right)
  ]

Justification :

* Observable directement
* Lié au mouvement exploitable (range)
* Plus robuste que variance classique

---

### B. Architecture 2-LSTM (Intra-jour + Inter-jours)

#### LSTM temporel

[
y_{t_D} = (V_{t_D-p_t}, \ldots, V_{t_D-1})
\in \mathbb{R}^{p_t \times 1}
]

Avec ( p_t = 20 ).

#### LSTM périodique

[
z_{t_D} = (V_{t_{D-p_d}}, \ldots, V_{t_{D-1}})
\in \mathbb{R}^{p_d \times 1}
]

Avec ( p_d = 20 ) jours.

#### Modèle combiné

[
f_{\Theta}(x_{t_D})
= \text{DNN}(\text{LSTM}*t(y*{t_D}), \text{LSTM}*D(z*{t_D}))
]

---

### C. Extension Multi-Paires (4-Pairs Learning)

[
y_{t_D} \in \mathbb{R}^{p_t \times p}
\qquad
z_{t_D} \in \mathbb{R}^{p_d \times p}
]

Avec ( p = 4 ) paires :

* EURUSD
* USDJPY
* EURSEK
* XAUUSD

---

# 5.3 Validation Empirique

### MSE :

[
\text{MSE}
= \frac{1}{N}\sum_{i=1}^N (\hat{V}*{t+1}^i - V*{t+1}^i)^2
]

### Résultats (Liao et al., 2021)

| Modèle             | EURUSD MSE (×10⁻⁸) | Réduction vs AR | Réduction vs GARCH |
| ------------------ | ------------------ | --------------- | ------------------ |
| AR(p)              | 0.89               | baseline        | –                  |
| GARCH(1,1)         | 1.08               | –               | baseline           |
| DNN                | 1.76               | −97%            | −63%               |
| LSTM_t             | 0.62               | +30%            | +43%               |
| 2-LSTM             | 0.61               | +31%            | +44%               |
| **4-Pairs 2-LSTM** | **0.56**           | **+37%**        | **+48%**           |

---

## Test de Diebold–Mariano

[
DM =
\frac{\bar{d}}{\sqrt{\frac{2\pi \hat{f}_d(0)}{T}}}
]

Résultats :

* vs AR(p) → **+7.55**
* vs GARCH → **+8.06**
* vs DNN → **+15.24**

---

# 5.4 Intégration dans la Stratégie de Trading

## A. Prédiction pré-news

À ( t_0 - 5 ) minutes :

[
\hat{V}*{t_0:t_0+15}
= f*{\text{LSTM}}(y_{t_0D}, z_{t_0D})
]

---

## B. Calibration dynamique TP/SL

Take profit :

[
TP_{\text{final}}
= TP_C \cdot
\left(\frac{\hat{V}*{t_0+15}}{V*{\text{historique}}}\right)^{\beta}
]

Stop loss :

[
SL_{\text{final}}
= SL_C \cdot
\left(\frac{\hat{V}*{t_0+15}}{V*{\text{historique}}}\right)^{\gamma}
]

Avec :

* ( \beta = 0.3\text{ à }0.5 )
* ( \gamma = 0.5\text{ à }0.7 )

---

## C. Position Sizing Adaptatif

[
\text{Size}
= \frac{\text{Size}*{\text{base}}}{1 + \lambda \hat{V}*{t_0+15}}
]

---
Parfait — on continue.
Voici **la PARTIE 2 / 7** de ton README complet, avec **toutes les équations MathJax**, format **100% compatible GitHub**, en respectant ton texte original.

---

# ✅ README — PARTIE 2 / 7

*(Section 6 complète + Section 7 complète)*

---

# 6. Calibration TP / SL / Horizon par Statistiques Non-Paramétriques

## 6.1 Méthodologie de Base

Pour chaque cluster ( C ) défini par :

* type d’événement
* régime VIX
* signe du sentiment

on extrait l’ensemble historique :

[
S_C = { (R_i(\tau), D_i) }_{i \in C,, Y_i^{(1)} = 1}
]

où :

* ( R_i(\tau) ) = retour maximal dans la fenêtre ( \tau )
* ( D_i ) = adverse excursion (wick) avant le TP

---

## 6.2 Formules de Calibration de Base

### **Take Profit (TP)**

[
TP_C = Q_{0.50}\left( |R_i(\tau)| \right)
]

Médiane des movements gagnants.

---

### **Stop Loss (SL)**

[
SL_C = Q_{0.85}\left( D_i \right)
]

85ᵉ percentile des wicks adverses.

---

### **Horizon Temporel**

[
\tau_C = Q_{0.60}\left( t_{TP,i} \right)
]

Temps médian requis pour atteindre TP.

---

## 6.3 Ajustement par Volatilité Prédite (LSTM)

### Take Profit ajusté :

[
TP_{\text{final}}
= TP_C \times
\left(
\frac{\hat{V}*{\text{LSTM}}}{V*{\text{médiane},C}}
\right)^{0.4}
]

### Stop Loss ajusté :

[
SL_{\text{final}}
= SL_C \times
\left(
\frac{\hat{V}*{\text{LSTM}}}{V*{\text{médiane},C}}
\right)^{0.6}
]

### Horizon ajusté :

[
\tau_{\text{final}}
= \tau_C \times
\left(
\frac{V_{\text{médiane},C}}
{\hat{V}_{\text{LSTM}}}
\right)^{0.3}
]

---

## 6.4 Justification Statistique

* La **médiane** est robuste aux outliers
* Les **percentiles élevés (85–90%)** conviennent pour calibrer les SL
* Aucune hypothèse de Gaussianité → **méthode non-paramétrique adaptée au FX**
* Ajustement LSTM = adaptation dynamique aux régimes de volatilité

---

# 7. Règle de Trading Finale Intégrée

## 7.1 Workflow Décisionnel Complet

---

### 🔵 **Phase 1 — Pré-news (t₀ − 5 minutes)**

Prédiction de la volatilité :

[
\hat{V}*{t_0:t_0+15}
= f*{\text{4-Pairs-2-LSTM}}(y_{t_0D}, z_{t_0D})
]

---

### 🔵 **Phase 2 — Post-news (t₀)**

Construction des features et prédiction ML :

1. **Prédiction spike (Modèle 1)**

[
p_{\text{spike}}
= P\left( Y^{(1)} = 1 \mid X, \hat{V}_{\text{LSTM}} \right)
]

Seuil typique : ( p_{\text{spike}} > 0.60 )

---

2. **Filtre de régime VIX**

[
I_t = \mathbb{1}{ VIX_t > EMA_n(VIX)_t }
]

---

3. **Prédiction directionnelle (Modèle 2)**

[
p_{\text{up}} = P(Y^{(2)} = 1 \mid X, \hat{V}_{\text{LSTM}})
]

Décision :

* Si ( p_{\text{up}} > 0.60 ) → **LONG**
* Si ( p_{\text{up}} < 0.40 ) → **SHORT**
* Sinon → **Pas de trade**

---

4. **Calibration TP/SL/Horizon**

Lookup des percentiles du cluster ( C ), puis ajustement LSTM :

[
TP_{\text{final}}, SL_{\text{final}}, \tau_{\text{final}}
]

---

5. **Position sizing (Kelly fractionnel + volatilité)**

[
\text{Lots}
= \frac{
0.25 \times f^* \times \text{Capital}
}{
SL_{\text{final}} \cdot \text{pip_value} \cdot
\sqrt{1 + 2\hat{V}_{\text{LSTM}} / V_0}
}
]

---

6. **Exécution :**

Ordre ouvert à :

* **Entry** : ( t_0 + 2 ) ticks
* **TP** : Entry ± ( TP_{\text{final}} )
* **SL** : Entry ∓ ( SL_{\text{final}} )
* **Max hold** : ( \tau_{\text{final}} ) secondes

---

## 7.2 Gestion des sorties

Ordre de priorité :

1. **TP touché** → clôture positive
2. **SL touché** → perte contrôlée
3. **Timeout** → clôture au marché
4. **Survolatilité**

[
\hat{V}*{\text{live}} > 3 \times \hat{V}*{\text{LSTM}}
]

→ fermeture immédiate

---

## 7.3 Avantages de l’approche hybride

| Composante              | Rôle                        | Bénéfice                |
| ----------------------- | --------------------------- | ----------------------- |
| ML Classification       | Filtre les bons contextes   | Réduit les faux signaux |
| LSTM volatilité         | Ajuste TP/SL                | Dynamique et adaptatif  |
| Percentiles historiques | Donne des valeurs réalistes | Anti-overfitting        |
| VIX/EMA                 | Filtre méta                 | +30% Sharpe en high-vol |

---

Parfait — on passe maintenant à **la PARTIE 3 / 7 du README**, avec **toutes les notations mathématiques en format GitHub-compatible (MathJax)**, parfaitement alignée avec ton texte original.

Cette partie contient :

✔ **Section 8 complète : Fondements mathématiques et références scientifiques**
✔ Tous les **résultats empiriques**
✔ Toutes les **équations LSTM, VIX, DSR, backtest overfitting**
✔ Format parfaitement propre pour ton README GitHub

---

# ✅ README — PARTIE 3 / 7

*(Section 8 complète)*

---

# 8. Fondements Mathématiques et Références Scientifiques

---

# 8.1 Event Study en Haute Fréquence

Pour un événement macro à l’instant ( t_0 ), on étudie la distribution :

[
R(\tau) = \sum_{k=1}^{\tau / \Delta t}
\left(
\ln(P_{t_0 + k\Delta t}) - \ln(P_{t_0 + (k-1)\Delta t})
\right)
]

avec :

* ( \Delta t = 1\text{ minute} )
* ( P_t ) = prix spot

---

## Résultat empirique clé (Andersen et al., 2003)

[
\mathbb{E}[R(\tau) \mid \text{news macro}] = 0
]

[
\text{Var}\big[R(\tau) \mid \text{news macro}\big]
\gg
\text{Var}\big[R(\tau) \mid \text{no news}\big]
]

➡ **Les annonces macro créent un régime de volatilité exploitable.**

---

# 8.2 Sentiment Financier & Prédiction Directionnelle

Si ( S \in [-1, 1] ) représente le sentiment d’une news :

[
\mathbb{E}\big[\text{sign}(R(\tau)) \mid S > 0 \big] > 0
]

[
\mathbb{E}\big[\text{sign}(R(\tau)) \mid S < 0 \big] < 0
]

**Résultat empirique (Shapiro 2024, FinBERT sur FX 2015-2023) :**

[
P(\text{bonne direction} \mid |S| > 0.5)
\approx 0.58
]

➡ **Edge directionnel = +8% par rapport au hasard.**

---

# 8.3 Prédiction de Volatilité par LSTM

### Théorème d’approximation universelle (Schäfer & Zimmermann, 2006)

Un réseau LSTM avec suffisamment de neurones peut approximer toute fonction mesurable :

[
f: \mathbb{R}^T \to \mathbb{R}
]

➡ Justifie l’utilisation du LSTM pour la volatilité intraday.

---

## Résultat empirique sur Forex (Liao, Chen & Ni, 2021)

Comparaison de modèles sur EURUSD / USDJPY / EURSEK / USDMXN :

| Modèle                 | MSE (×10⁻⁸) | Gain vs AR | Gain vs GARCH |
| ---------------------- | ----------- | ---------- | ------------- |
| AR(p)                  | 0.89        | baseline   | -             |
| GARCH(1,1)             | 1.08        | -          | baseline      |
| DNN simple             | 1.76        | −97%       | −63%          |
| LSTM intra-day         | 0.62        | +30%       | +43%          |
| 2-LSTM (intra + inter) | 0.61        | +31%       | +44%          |
| 4-Pairs 2-LSTM         | 0.56        | +37%       | +48%          |

### Ratio d’amélioration :

[
\frac{\text{MSE}*{\text{GARCH}} - \text{MSE}*{\text{4P-2LSTM}}}
{\text{MSE}_{\text{GARCH}}}
\approx 48%
]

➡ **Le gain est massif : près de 50% de réduction d’erreur.**

---

### Patterns capturés par le LSTM

#### 1. Saisonnalité intraday :

[
\mathbb{E}[V_t \mid \text{hour} = 7]
\approx 1.8 \times
\mathbb{E}[V_t \mid \text{hour} = 3]
]

#### 2. Auto-corrélation intra-minute :

[
\text{Corr}(V_t, V_{t-1}) \approx 0.5
]

#### 3. Auto-corrélation inter-jours (NFP) :

[
\text{Corr}(V^D_t, V^{D-20}_t)
\approx 0.3
]

#### 4. Corrélations croisées entre paires :

[
\text{Corr}(V_{\text{EURUSD}}, V_{\text{USDJPY}})
\approx 0.65
]

➡ **Justifie le 4-pairs-learning.**

---

# 8.4 Filtre de Régime VIX

Définition :

[
I_t = \mathbb{1}\left{
VIX_t >
EMA_n(VIX)_t
\right}
]

Version pondérée exponentielle :

[
I_t =
\mathbb{1}
\left{
VIX_t >
\frac{2}{22}
\sum_{k=0}^{20}
\left(\frac{20}{22}\right)^k
VIX_{t-k}
\right}
]

---

### Résultat empirique (Hodges & Sira, 2018)

[
\frac{
\text{Var}[R \mid I_t = 1]
}{
\text{Var}[R \mid I_t = 0]
}
\approx 2.3
]

➡ Les stratégies momentum / breakout performent **2.3× mieux** en régime VIX élevé.

---

# 8.5 Overfitting & Deflated Sharpe Ratio (DSR)

Lorsqu'on teste beaucoup de configurations, le Sharpe le plus élevé est **forcément biaisé**.

---

## Espérance du Sharpe maximal sous le hasard (Lopez de Prado)

[
SR_0 =
\sqrt{
\frac{2\ln(N)}{T}
}
]

où :

* ( N ) = nombre total de configurations testées
* ( T ) = nombre d'observations

---

## Formule du Deflated Sharpe Ratio (DSR)

[
DSR
===

\Phi\left(
\frac{
(SR - SR_0)\sqrt{T-1}
}{
\sqrt{
1 - \gamma_3 SR +
\frac{\gamma_4 - 1}{4}SR^2
}
}
\right)
]

Avec :

* ( \gamma_3 ) = skewness
* ( \gamma_4 ) = kurtosis

Critère d’acceptation :

[
DSR > 0.93
]

➡ Confidence > 95% que la stratégie ne soit pas un artefact.

---

## Exemple numérique

Données :

* ( SR_{\text{IS}} = 1.6 )
* ( N = 200 )
* ( T = 1260 )
* ( \gamma_3 = -0.3 )
* ( \gamma_4 = 5.2 )

---

### 1. Calcul du ( SR_0 )

[
SR_0
====

\sqrt{
\frac{2\ln(200)}{1260}
}
=

0.092
]

---

### 2. Numérateur

[
(1.6 - 0.092)\sqrt{1259}
\approx 53.5
]

---

### 3. Dénominateur

[
\sqrt{
1 + 0.48 + 2.69
}
= 2.04
]

---

### 4. DSR Final :

[
DSR
= \Phi(26.2)
\approx 1.00
]

➡ **Stratégie validée avec probabilité > 99.99% d’être réelle.**

---

# 8.6 Minimum Backtest Length (MinBTL)

Pour un Sharpe cible :

[
MinBTL
\approx
\frac{2\ln(N)}
{SR_{\text{target}}^2}
]

Exemple :

[
SR_{\text{target}} = 1.4,
\quad N = 200
]

[
MinBTL
\approx
1.02 \times \ln(200)
\approx 5.3 \text{ ans}
]

➡ **Avec 7 ans de données (2018–2025), la stratégie est valide.**

---
Parfait — on enchaîne directement avec **la PARTIE 4 / 7 du README**, contenant **toute la Section 9 : Implémentation Technique**, entièrement réécrite pour un affichage parfait sur GitHub (MathJax, tableaux, code Python, bullet points propres).

---

# ✅ README — PARTIE 4 / 7

## **9. Implémentation Technique**

---

# 9.1 Stack Technologique

### **Collecte & Préparation des Données**

| Source              | Usage                                        | Technologie                              |
| ------------------- | -------------------------------------------- | ---------------------------------------- |
| **Forex Factory**   | News macro (date, consensus, actual, impact) | Python, Selenium, BeautifulSoup          |
| **Dukascopy**       | Prix minute (M1) / tick                      | Node.js API, dukascopy-node              |
| **VIX**             | Régimes de volatilité                        | Python `yfinance`                        |
| **Financial Juice** | Sentiment news en temps réel                 | API Python (`financial-news-api-python`) |

---

### **Feature Engineering**

| Objectif                           | Librairies          |
| ---------------------------------- | ------------------- |
| Création signaux macro & sentiment | pandas, numpy       |
| Indicateurs techniques             | ta-lib              |
| Calcul volatilité (log-range)      | numpy               |
| Fenêtrage temporel pour ML         | scikit-learn, NumPy |

---

### **Machine Learning**

| Modèle                           | Usage                                       |
| -------------------------------- | ------------------------------------------- |
| **Random Forest**                | Classification spike / no spike             |
| **XGBoost / LightGBM**           | Classification direction + probas calibrées |
| **LSTM (2-LSTM & 4-Pairs-LSTM)** | Prédiction de volatilité minute suivante    |
| **SHAP**                         | Explainabilité des features ML              |

---

### **Backtesting**

* **backtrader** → exécution candle par candle
* **Vectorized engine custom** → simulations rapides pour hyperparamètres
* Gestion :

  * spreads
  * slippage
  * latence
  * marché fermé / trous data

---

### **Exécution en Temps Réel**

| Technologie              | Usage                       |
| ------------------------ | --------------------------- |
| MetaTrader5 (API Python) | Envoi ordres en réel        |
| Websocket FinancialJuice | News & sentiment temps réel |
| Cron / scheduler         | Mise à jour modèle          |

---

---

# 9.2 Pipeline de Données (Production)

Le pipeline complet suit la structure :

```
raw/ → processed/ → features/ → models/ → backtests/ → live/
```

---

## **Étape 1 — Extraction Forex Factory**

### Commande (scraper FF) :

```bash
python -m src.forexfactory.main \
    --start 2018-01-01 \
    --end 2025-12-31 \
    --csv ff_events.csv \
    --tz UTC
```

---

## **Étape 2 — Extraction des prix (Dukascopy)**

Exemple pour EURUSD M1 :

```bash
npx dukascopy-node -i eurusd -from 2018-01-01 -to 2025-12-31 -t m1 -f csv -o eurusd_m1.csv
```

---

## **Étape 3 — Synchronisation News + Prix**

Alignement sur la minute exacte :

* t0 = minute de la news
* Fenêtre de 30 minutes après la news
* Calcul des retours, spikes, wick, TP/SL simulés

---

## **Étape 4 — Calcul du Log-Range**

[
LogRange_t = \ln(High_t) - \ln(Low_t)
]

---

## **Étape 5 — Construction des Features**

### **Features macro**

* Surprise normalisée :

[
\text{surp} =
\frac{\frac{\text{actual} - \text{forecast}}{\text{forecast}} - \mu}{\sigma}
]

* Importance impact (low / medium / high)

---

### **Features sentiment**

* Sentiment Financial Juice (S ∈ [-1,1])
* Intensité : nombre de mentions
* Vitesse de diffusion des news

---

### **Features de volatilité (VIX)**

[
I_t =
\mathbb{1}
\left{
VIX_t >
EMA_n(VIX)_t
\right}
]

---

### **Features prix**

* ATR(14)
* Retour 1m, 5m, 15m
* Range dernier quart-d'heure
* Ratio wick / body

---

### **Fenêtrage ML (séries temporelles)**

Pour LSTM :

[
X_t =
\left[
V_{t-20},
\ldots,
V_{t-1}
\right]
]

Pour classification ML :

* 150+ features tabulaires (news, sentiment, prix, VIX)

---

---

# 9.3 Entraînement Machine Learning

---

## 🔵 Modèle 1 — Spike Classifier

Objectif : déterminer si une news génère un spike exploitable dans les 5 minutes.

Label :

[
Y^{(1)} =
\begin{cases}
1 & \text{si } |Return_{0:5m}| > \text{threshold} \
0 & \text{sinon}
\end{cases}
]

Modèles utilisés :

* RandomForestClassifier
* XGBoostClassifier
* LightGBM

---

## 🔵 Modèle 2 — Direction Classifier

Objectif : direction dominante dans les 5 minutes.

[
Y^{(2)} =
\begin{cases}
1 & \text{si } Return_{0:5m} > 0 \
0 & \text{sinon}
\end{cases}
]

---

## 🔵 Modèle 3 — LSTM (Volatilité Futur)

Prédiction de :

[
\hat{V}_{t+1}
]

Architecture :

* 2 LSTM (intra-day / inter-day)
* concat
* Dense(32)
* Dense(32)
* output = log-range prédite

---

## Exemples de code (GitHub Rendering OK)

### **Random Forest (Spike Model)**

```python
from sklearn.ensemble import RandomForestClassifier

model_spike = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    class_weight="balanced"
)

model_spike.fit(X_train, y_train)
```

---

### **LSTM (Volatility Model)**

```python
import tensorflow as tf

input_layer = tf.keras.Input(shape=(20, 1))

x = tf.keras.layers.LSTM(32, return_sequences=False)(input_layer)
x = tf.keras.layers.Dense(32, activation="relu")(x)
output = tf.keras.layers.Dense(1)(x)

model_lstm = tf.keras.Model(inputs=input_layer, outputs=output)
model_lstm.compile(optimizer="adam", loss="mse")
```

---

---

# 9.4 Backtesting Engine

Le système de backtesting simule :

### ✔ Spreads réels (par paire)

### ✔ Slippage dynamique (volatilité-dependant)

### ✔ Latence d’exécution :

[
delay = 150 \text{ ms}
]

### ✔ Remplissage partiel des ordres

### ✔ Gaps autour des news

### ✔ Filtre VIX + LSTM + ML classification

---

## Logique du Backtest (workflow)

1. Détecter une news high-impact
2. Vérifier spike probability :

[
p_{\text{spike}} > 0.60
]

3. Vérifier direction :

[
p_{\text{up}} > 0.60 \Rightarrow LONG
]
[
p_{\text{up}} < 0.40 \Rightarrow SHORT
]

4. Calcul TP/SL via percentiles
5. Ajustement par volatilité LSTM
6. Sizing par Kelly fractionnel :

[
Lots = \frac{
0.25 f^* \cdot Capital
}{
SL_{\text{final}} \cdot pip_value
}
]

7. Exécution 2 ticks après la news
8. Sorties :

   * TP
   * SL
   * expiry time
   * regime shift (LSTM ×3)

---

# 9.5 Exécution Live (Production)

Pipeline en temps réel :

1. Récupération news FinancialJuice par Websocket
2. Classification spike + direction
3. Calcul volatilité prédite
4. Vérification du régime VIX
5. Déclenchement trade + TP/SL via MT5 API
6. Monitoring + clôture dynamique

Click & run :

```python
import MetaTrader5 as mt5

mt5.initialize()

order = {
    "symbol": "EURUSD",
    "type": mt5.ORDER_TYPE_BUY,
    "volume": lot_size,
    "price": mt5.symbol_info_tick("EURUSD").ask
}

mt5.order_send(order)
```

---
Parfait — on passe maintenant à **la PARTIE 5 / 7**, qui correspond à **toute la Section 10 du README : le Protocole Anti-Overfitting**.
Cette partie est réécrite **entièrement**, **fidèlement**, et **avec une mise en forme parfaite pour GitHub**, incluant :

✔ MathJax correctement affiché
✔ Tableaux
✔ Équations alignées
✔ Exemples numériques
✔ Explications propres et cohérentes

---

# ✅ README — PARTIE 5 / 7

# **10. Protocole Anti-Overfitting**

La stratégie repose sur un protocole strict permettant d'éviter tout sur-apprentissage lié :

* au nombre d’essais,
* à la recherche d’hyperparamètres,
* aux biais de sélection,
* au tuning sur données futures.

Ce protocole suit les recommandations de **Bailey & López de Prado (2014)**.

---

# 10.1 Comptabilisation du Nombre d’Essais (N)

Toute optimisation doit comptabiliser explicitement :

* types d’événements,
* paires tradées,
* régimes de volatilité,
* seuils ML,
* configurations LSTM,
* méthodes de scaling TP/SL.

### **Exemple réel de ton projet**

[
N =
3 \text{ (événements)} \times
2 \text{ (paires)} \times
2 \text{ (régimes VIX)} \times
5 \text{ (seuils ML)} \times
3 \text{ (lags LSTM)} \times
2 \text{ (méthodes de scaling)}
]

[
N = 360 \text{ configurations}
]

---

# 10.2 Minimum Backtest Length (MinBTL)

Selon de Prado, pour un Sharpe Ratio cible ( \text{SR}_{target} ), la longueur minimale de backtest doit vérifier :

[
\text{MinBTL} \approx \frac{2 \ln(N)}{\text{SR}_{target}^2}
]

### Pour ton cas :

* ( N = 360 )
* ( \text{SR}_{target} = 1.4 )

[
\text{MinBTL}
\approx \frac{2 \ln(360)}{1.4^2}
= \frac{2 \times 5.89}{1.96}
\approx 6.0 \text{ ans}
]

### Disponibilité réelle :

* Données disponibles : **7 ans (2018–2025)** → ✅ Suffisant
* Marge faible → Limiter les essais à ( N \le 200 ) si possible

---

# 10.3 Split In-Sample / Out-of-Sample (IS/OOS)

Aucune optimisation autorisée sur l’OOS.

| Période       | Usage                                       |
| ------------- | ------------------------------------------- |
| **2018–2022** | IS (In-Sample) – calibration & entraînement |
| **2023–2025** | OOS (Out-of-Sample) – validation finale     |

## Critères stricts de rejet :

### Condition 1 — Stabilité du Sharpe

[
SR_{\text{OOS}} < 0.7 \times SR_{\text{IS}}
\quad \Rightarrow \quad
\text{Stratégie rejetée}
]

---

### Condition 2 — Contrôle du drawdown

[
\text{MaxDD}*{\text{OOS}} > 1.5 \times \text{MaxDD}*{\text{IS}}
\quad \Rightarrow \quad
\text{Risque sous-estimé → rejet}
]

---

### Condition 3 — Qualité directionnelle

[
\text{WinRate}*{\text{OOS}}
<
\text{WinRate}*{\text{IS}} - 10%
\quad \Rightarrow \quad
\text{Dégradation significative}
]

---

# 10.4 Deflated Sharpe Ratio (DSR)

Le DSR corrige le Sharpe pour :

* le nombre d’essais (N),
* la non-normalité des retours,
* le biais de sélection.

### Formule officielle (López de Prado, 2014)

[
DSR =
\Phi
\left(
\frac{
(SR - SR_0) \sqrt{T - 1}
}{
\sqrt{
1 - \gamma_3 SR +
\frac{\gamma_4 - 1}{4} SR^2
}
}
\right)
]

où :

* ( SR ) : Sharpe Ratio in-sample
* ( T ) : nombre d’observations
* ( \gamma_3 ), ( \gamma_4 ) : skewness et kurtosis
* ( SR_0 ) : SR maximal attendu sous ( H_0 ) :

[
SR_0 = \sqrt{ \frac{2\ln(N)}{T} }
]

---

## 🔢 Exemple numérique (issu de ton projet)

* ( SR_{\text{IS}} = 1.6 )
* ( T = 1260 ) observations (5 ans × 252 jours)
* ( N = 200 )
* ( \gamma_3 = -0.3 )
* ( \gamma_4 = 5.2 )

### Calcul de ( SR_0 )

[
SR_0 = \sqrt{
\frac{2\ln(200)}{1260}
}
=

\sqrt{
\frac{10.6}{1260}
}
\approx 0.092
]

---

### Numérateur du DSR

[
(SR - SR_0)\sqrt{T-1}
=====================

(1.6 - 0.092) \times 35.5
\approx 53.5
]

### Dénominateur

[
\sqrt{
1 - (-0.3)(1.6)
+
\frac{5.2 - 1}{4} (1.6^2)
}
]

# [

# \sqrt{1 + 0.48 + 2.69}

\sqrt{4.17}
\approx 2.04
]

---

### Résultat final

[
DSR = \Phi\left(\frac{53.5}{2.04}\right)
= \Phi(26.2)
\approx 1.00
]

### Interprétation :

➡️ Probabilité **> 99.99%** que le Sharpe reflète un vrai edge
➡️ Stratégie validée (seuil : **DSR > 0.93**)

---

# 10.5 Tests de Robustesse Obligatoires

---

## 🔵 A. Monte Carlo Permutation Test

1. Permuter aléatoirement les labels :

   * ( Y^{(1)} ) : spike / pas spike
   * ( Y^{(2)} ) : direction
2. Réentraîner **1000 fois**
3. Comparer distribution du Sharpe fictif avec Sharpe réel

### Critère :

[
SR_{\text{réel}} \text{ doit être dans le top 5%}
]

---

## 🔵 B. Walk-Forward Analysis (WFA)

Sur la période OOS :

1. Réentraînement tous les 3 mois
2. Fenêtre glissante (rolling window)
3. Vérification que la performance est stable au fil des régimes

---

## 🔵 C. Stress Testing

Test sous conditions dégradées :

* spreads × 2
* slippage + 50%
* commissions × 1.5
* ralentissement latence
* gapping autour des news

### Validation :

[
SR_{\text{stress}} > 1.0
]

---

# 10.6 Résumé du Protocole Anti-Overfitting

| Étape          | Objectif                        | Critère                 |
| -------------- | ------------------------------- | ----------------------- |
| Comptage N     | Contrôle du biais de sélection  | N défini & limité       |
| MinBTL         | Longueur suffisante du backtest | MinBTL ≤ durée data     |
| IS/OOS strict  | Validation indépendante         | AUCUNE optimisation OOS |
| DSR            | Validation statistique robuste  | DSR > 0.93              |
| Permutation MC | Vérifier absence de faux edge   | Sharpe réel > 95%       |
| Walk-Forward   | Stabilité temporelle            | Pas de drift            |
| Stress test    | Résilience                      | SR > 1.0                |

---

Parfait — on enchaîne avec **la PARTIE 6 / 7 du README**, c’est-à-dire **toute la section “11. Gestion du Risque et Position Sizing”**, entièrement réécrite pour GitHub avec :

✔ MathJax fonctionnel
✔ Formules propres
✔ Tableaux bien formatés
✔ Explications claires mais fidèles à ton contenu
✔ Les mêmes équations que tu avais, mais correctement rendues

---

# ✅ README — PARTIE 6 / 7

# **11. Gestion du Risque et Position Sizing**

La gestion du risque est un pilier essentiel du système.
Elle combine :

* **Kelly fractionnel**,
* **Ajustement dynamique par volatilité LSTM**,
* **Limites de drawdown multi-horizons**,
* **Diversification entre paires FX**.

---

# 11.1 Critère de Kelly Fractionnel

On utilise un Kelly **conservateur (1/4 Kelly)** pour éviter le sur-levier.

Le Kelly optimal standard :

[
f^* = \frac{p \cdot W - (1 - p) \cdot L}{W \cdot L}
]

où :

* ( p ) : Win rate empirique OOS
* ( W ) : average win (pips)
* ( L ) : average loss (pips)

### Position size initiale (1/4 Kelly)

[
\text{Lots} =
\frac{
f^* \times \text{Capital} \times 0.25
}{
\text{SL}_{\text{final}} \times \text{pip_value}
}
]

Le facteur **0.25** = ¼ Kelly
→ standard en gestion de portefeuille pour réduire la volatilité du capital tout en gardant l’effet composant.

---

# 11.2 Ajustement par Volatilité Prédite LSTM

On réduit le levier lorsque la volatilité prédite est élevée.

### Formule finale :

[
\text{Lots}*{\text{final}} =
\frac{
\text{Lots}*{\text{Kelly}}
}{
\sqrt{
1 + 2 \times
\frac{\hat{V}_{\text{LSTM}}}{V_0}
}
}
]

où :

* ( \hat{V}_{\text{LSTM}} ) : volatilité prédite
* ( V_0 ) : volatilité baseline (médiane historique du cluster)

### Interprétation :

| Situation         | Effet sur la taille | Raison                |
| ----------------- | ------------------- | --------------------- |
| Volatilité faible | taille ↑            | marché stable         |
| Volatilité élevée | taille ↓            | éviter sur-exposition |

---

# 11.3 Limites de Drawdown & Circuit Breakers

Ces règles stoppent le système si le risque devient trop élevé.

---

### **Règle 1 – Stop journalier**

[
\text{Si } \text{Loss}_{\text{today}} > 0.03 \times \text{Capital}
\Rightarrow \text{ arrêt du trading 24h}
]

---

### **Règle 2 – Stop hebdomadaire**

[
\text{Si } \text{Loss}_{\text{week}} > 0.06 \times \text{Capital}
\Rightarrow \text{ arrêt du trading 48h}
]

---

### **Règle 3 – Drawdown maximal**

[
\text{Si } DD_{\text{current}} > 0.15 \times \text{Capital}
\Rightarrow \text{ arrêt complet + audit}
]

### Interprétation :

* Protection contre un changement structurel du marché
* Empêche la stratégie de “creuser un trou” en période adverse
* Conforme aux standards hedge funds

---

# 11.4 Diversification Inter-Paires

Pour éviter concentration du risque :

[
\sum_{\text{paire}} \text{Exposition}_{\text{paire}}
\leq 0.40 \times \text{Capital}
]

Aucune paire ne doit dépasser **40%** du capital total engagé.

### Répartition optimale empirique (corrélations FX)

| Paire  | Allocation |
| ------ | ---------- |
| EURUSD | 35%        |
| XAUUSD | 30%        |
| USDJPY | 20%        |
| EURSEK | 15%        |

Cela repose sur les corrélations croisées observées :

* **USDJPY ↔ EURUSD : ~0.65**
* **EURSEK ↔ EURUSD : structure régionale**
* **XAUUSD : actif refuge, décorrélé en régimes risk-off**

---

# 🔍 Synthèse de la Section Gestion du Risque

| Élément           | Rôle                             | Effet               |
| ----------------- | -------------------------------- | ------------------- |
| Kelly fractionnel | Ajuster levier selon win rate    | contrôle volatilité |
| Ajustement LSTM   | Adapter la taille aux conditions | stabilise drawdown  |
| Circuit breakers  | Stopper avant catastrophe        | prévient ruine      |
| Diversification   | Réduire corrélation des pertes   | robustesse          |

---

Parfait — on enchaîne immédiatement avec **la PARTIE 7 / 7 du README**, c’est-à-dire :

# ✔️ **12. Backtesting & Validation complète**

# ✔️ **13. Conclusion stratégique**

# ✔️ **14. Références & Annexes**

Le tout réécrit avec **MathJax**, **GitHub-compatible Markdown**, et **les mêmes contenus avancés** que tu avais, mais parfaitement mis en forme.

---

# #️⃣ **12. Backtesting et Validation**

Le système doit être validé selon un protocole rigoureux (standards quantitatifs institutionnels).

---

# **12.1 Métriques de Performance**

### Métriques primaires (obligatoires)

* **Sharpe Ratio annualisé**
  [
  SR = \frac{\mu_R}{\sigma_R}
  \quad \text{(cible > 1.4)}
  ]

* **Sortino Ratio**
  [
  \text{Sortino} = \frac{\mu_R}{\sigma_{\text{down}}}
  \quad \text{(cible > 2.0)}
  ]

* **Maximum Drawdown**
  [
  \text{MaxDD} < 15%
  ]

* **Calmar Ratio**
  [
  \text{Calmar} > 1.5
  ]

---

### Métriques secondaires

* **Win Rate > 55%**
* **Profit Factor > 1.8**
* **Average Win / Average Loss > 2**
* **Recovery Factor > 3**

---

# **12.2 Analyse de Sensibilité**

On teste la robustesse de la stratégie à plusieurs variations :

| Paramètre          | Valeurs testées                   |
| ------------------ | --------------------------------- |
| Seuil ML spike     | 0.55 / 0.60 / 0.65 / 0.70         |
| Lag LSTM           | 10 / 15 / 20 / 25 / 30            |
| Scaling TP-SL      | (0.3,0.6) — (0.4,0.7) — (0.5,0.8) |
| Régime VIX         | high-vol only / all regimes       |
| Types d’événements | CPI only / CPI+NFP / all          |

🎯 **Critère de robustesse :**

[
SR > 1.2 \quad \text{pour ≥ 70% des configurations}
]

---

# **12.3 Simulation Monte Carlo**

Nous simulons 10 000 versions alternatives de l'historique :

* **bootstrap** des retours
* **permutation temporelle**
* **séquences adverses (5 pertes consécutives)**
* **scrambling de l’ordre des trades**

Critères :

[
P(\text{ruine à 0.5×capital}) < 1%
]

[
\text{mediane}(Capital_{\text{final}}) > 1.5 \times Capital_{\text{initial}}
]

Cela garantit **résilience**, pas seulement performance brute.

---

# #️⃣ **13. Conclusion**

# **13.1 Architecture Hybride — Synthèse**

| Niveau | Technologie                   | Rôle                                | Gain empirique          |
| ------ | ----------------------------- | ----------------------------------- | ----------------------- |
| **1**  | Random Forest / XGBoost       | Filtrer contextes exploitables      | Precision > 65%         |
| **2**  | 4-Pairs 2-LSTM                | Prédire volatilité intrajournalière | MSE −48% vs GARCH       |
| **3**  | Percentiles non-paramétriques | Fixer TP/SL réalistes               | Win Rate > 55%          |
| **4**  | Filtre VIX/EMA                | Meta-filtre de régime               | Sharpe +30% en high-vol |

---

# **13.2 Avantages Compétitifs**

✔ **Approche modulaire** : chaque bloc peut être amélioré séparément
✔ **Ancrage empirique** :

* Patterns LSTM validés sur 730 jours (Liao et al., 2021)
* Event Study confirmé sur 40 ans (Andersen, 2003)
  ✔ **Anti-overfitting rigoureux** :
* MinBTL respecté
* DSR > 0.93
* Validation OOS stricte
  ✔ **Adaptation dynamique** :
* TP/SL ajustés par volatilité prédite
* Position sizing intelligent

---

# **13.3 Limites Réelles**

### Risques techniques

* Latence > 200 ms → slippage
* Erreur de prédiction LSTM en événements extrêmes
* Qualité variable des news Forex Factory

### Risques de marché

* Flash crash
* Rupture structurelle (post-2025)
* Corrélations instables

### Mitigation

* Réentraînement trimestriel
* Circuit breakers
* Monitoring continu

---

# **13.4 Roadmap**

### Court terme (3 mois)

* Pipeline complet
* Backtest vectorisé 2018–2022
* OOS 2023–2025
* Calcul DSR + robustesse

### Moyen terme (6 mois)

* Paper trading MT5
* Latence réduite < 150 ms
* Ajout de FinBERT sentiment

### Long terme

* Live avec capital réduit
* Publication (si SR > 1.5)
* Version open-source (hors modèles privés)

---

# **13.5 Message Final**

> **"Cette stratégie ne prédit pas l'avenir.
> Elle filtre le présent."**

* Le ML **ne devine pas** le prochain move
* Le LSTM **extrapole les patterns intraday récurrents**
* Les percentiles **bornent le risque dans des limites empiriques**

Le véritable edge vient de la **discipline + validation + robustesse**.

---

# #️⃣ **14. Références Scientifiques**

### Articles académiques

* Andersen, T. G., Bollerslev, T., Diebold, F. X., & Vega, C. (2003).
  *Micro Effects of Macro Announcements: Real-Time Price Discovery in FX*. AER.
* Bailey, D. & López de Prado, M. (2014).
  *The Deflated Sharpe Ratio*. JPM.
* Liao, Chen & Ni (2021).
  *Volatility Prediction using Neural Networks*. arXiv:2112.01166
* Alizadeh, Brandt & Diebold (2002).
  *Range-Based Estimation of Stochastic Volatility*. JF.
* Hodges & Sira (2018).
  *VIX Regime Filtering*. Quant Finance.
* Shapiro et al. (2024).
  *Measuring News Sentiment*. Journal of Econometrics.
* Schäfer & Zimmermann (2006).
  *RNNs are Universal Approximators*. IJNS.

---

# 📌 Annexes

## A. Glossaire, B. Formules, C. Hardware recopiés ET formatés pour GitHub

### A. Formules récapitulatives

#### **Surprise normalisée**

[
\text{normalized_surprise} =
\frac{
\frac{\text{actual} - \text{consensus}}{\text{consensus}} - \mu
}{
\sigma
}
]

#### **Régime VIX**

[
I_t = \mathbb{1}
\left{
\text{VIX}*t >
\frac{2}{22}
\sum*{k=0}^{20}
\left(\frac{20}{22}\right)^k
\text{VIX}_{t-k}
\right}
]

#### **TP ajusté volatilité LSTM**

[
TP_{\text{final}} =
Q_{0.50}(R_C)
\times
\left(
\frac{\hat V_{\text{LSTM}}}{Q_{0.50}(V_C)}
\right)^{0.4}
]

#### **Position sizing avec Kelly**

[
\text{Lots} =
\frac{
0.25 f^* \cdot \text{Capital}
}{
SL_{\text{final}} \cdot \text{pip_value}
\sqrt{
1 + 2\hat{V}_{\text{LSTM}} / V_0
}
}
]

#### **Deflated Sharpe Ratio**

[
DSR =
\Phi\left(
\frac{
(SR - \sqrt{2\ln(N)/T})\sqrt{T-1}
}{
\sqrt{
1 - \gamma_3 SR +
\frac{\gamma_4 - 1}{4} SR^2
}
}
\right)
]

---
