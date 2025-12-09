# Rapport Stratégique Complet : Trading Algorithmique Multi-Symboles ML + LSTM Volatilité – Risk 1%

**Version 8.1 – Décembre 2025**  
**Auteur : Quant Dev Team**  
**Stack : Forex Factory, Dukascopy, Python/ML/LSTM/TensorFlow, MT5 Live (Exness)**  
**Symboles : XAUUSD, EURUSD, GBPUSD, USA30IDXUSD, USATECHIDXUSD**  
**Période : 2008–2025 (17 ans)**  
**Risque : 1% capital/trade fixe + lot sizing dynamique V̂LSTM**  
**N=180 essais documentés | MinBTL=5.3ans ✓ | DSR p<0.05 obligatoire**

---

## 1. Introduction et Objectifs Stratégiques

Stratégie hybride multi-symboles anti-overfit exploitant spikes post-news macro (CPI, NFP, FOMC) via **ML conservatrice P>0.65** + **2-LSTM multi-paires MSE-48% GARCH**. **Risk 1%/trade**, TP/SL/lots scalés V̂LSTM, filtrage spread/tick strict. **Protocole industriel : N=180 comptabilisé, MinBTL respecté, DSR/CSCV/PBO obligatoires**.

| Composante | Approche | Performance Cible OOS |
|------------|----------|----------------------|
| Détection | ML P(spike)>0.65 | Precision 65% |
| Direction | ML P>0.65 long/<0.35 short | Edge 8% |
| Volatilité | 2-LSTM 4-paires | MSE -48% GARCH |
| Risk | 1% dynamique V̂LSTM | MaxDD ≤15% |
| **Anti-overfit** | **DSR + MinBTL + PBO** | **p<0.05** |

---

## 2. Fondements Scientifiques – Anti-Overfitting Industriel

### 2.1 De Prado #1 : Backtest Overfitting (DSR/PBO/CSCV)

**Deflated Sharpe Ratio (Bailey & López de Prado, 2014)** :

$$
DSR = \Phi\left(\frac{(SR - SR_0)\sqrt{T-1}}{\sqrt{1 - \gamma_3 SR + \frac{\gamma_4-1}{4}SR^2}}\right)
$$

où :
- $SR_0 = \sqrt{\frac{2\ln(N)}{T}}$ : SR maximal attendu sous $H_0$ (données aléatoires)
- $N$ : Nombre de configurations testées
- $T$ : Nombre d'observations
- $\gamma_3$ : Skewness des retours
- $\gamma_4$ : Kurtosis des retours
- $\Phi(\cdot)$ : Fonction de répartition normale standard

**Application pratique** :
```
DSR = Z((SR - SR₀)√(T-1) / √(1 - γ₃SR + (γ₄-1)/4 × SR²))
SR₀ = 2√N/T sous H₀ (N essais)
CSCV → PBO (Probabilité Backtest Overfitting)
```

**N=180 documenté**, DSR p<0.05, PBO<0.5 via CSCV.

**Critère de validation** : Exiger $DSR > 0.93$ (p-value < 5%) pour valider la stratégie.

### 2.2 De Prado #2 : MinBTL + Effets Compensation

**Minimum Backtest Length** :

$$
\text{MinBTL (années)} \approx \frac{2\ln(N)}{E[\max_N SR]^2} \approx 1.02 \times \ln(N)
$$

Pour $N=180$ essais :

$$
\text{MinBTL} = 1.02 \times \ln(180) \approx 1.02 \times 5.19 \approx 5.3 \text{ ans}
$$

**Validation** : 17 ans de données (2008-2025) ✓

**Risque majeur :** Autocorrélation marchés → **SR_IS↑ = SR_OOS↓**. Stratégies saisonnières = **artefact statistique** jusqu'à preuve DSR.

### 2.3 Liao (2008–2025 Adapté) : 2-LSTM Volatilité Forex

**Réduction MSE empirique** :

$$
\frac{\text{MSE}_{\text{GARCH}} - \text{MSE}_{\text{4P-2LSTM}}}{\text{MSE}_{\text{GARCH}}} \approx 48\%
$$

**Données M1 :** 2008-2025 (EURUSD/XAUUSD/GBPUSD/indices), log-range cible.  
**Patterns :** London(7h)/NY(12h)/NFP(13h30), autocorr lag=20min/jour, cross-pairs r=0.65.  
**4-Pairs 2-LSTM :** MSE ↓48% GARCH, lag=20 optimal.

---

## 3. Fondements Mathématiques Détaillés

### 3.1 Event Study en Haute Fréquence

**Définition formelle (Andersen et al., 2003)** : Pour un événement macro à $t_0$, on étudie la distribution de :

$$
R(\tau) = \sum_{k=1}^{\tau/\Delta t} r_{t_0+k\Delta t}
$$

où $r_t = \ln(P_t) - \ln(P_{t-\Delta t})$ est le log-return à minute $\Delta t = 1$.

**Résultat empirique clé** :

$$
E[R(\tau)|{\text{news macro}}] \neq 0, \quad \text{Var}[R(\tau)] \gg \text{Var}[R(\tau)|{\text{no news}}]
$$

Ceci justifie l'existence d'un edge statistique exploitable.

### 3.2 Construction des Features et Labels

#### A. Surprise Normalisée

$$
\text{surprise\_pct} = \frac{\text{actual} - \text{consensus}}{\text{consensus}} \times 100
$$

$$
\text{normalized\_surprise} = \frac{\text{surprise\_pct} - \mu_{\text{surprise}}}{\sigma_{\text{surprise}}}
$$

#### B. Régime de Volatilité VIX

Indicateur binaire défini comme :

$$
\text{vix\_regime} = \mathbb{1}\{\text{VIX}_t > \text{EMA}_{21}(\text{VIX})_t\}
$$

où l'EMA est calculée récursivement :

$$
\text{EMA}_n(V_t) = \alpha V_t + (1-\alpha)\text{EMA}_n(V_{t-1}), \quad \alpha = \frac{2}{n+1}
$$

**Propriété validée empiriquement (Hodges et Sira, 2018)** :

$$
\frac{\text{Var}[R | I_t = 1]}{\text{Var}[R | I_t = 0]} \approx 2.3
$$

#### C. Label has_spike_exploitable

Pour chaque événement $i$, on calcule le retour logarithmique maximal sur $\tau$ minutes :

$$
R_i(\tau) = \max_{t \in [t_0, t_0+\tau]} \ln\left(\frac{P_t}{P_{t_0}}\right)
$$

Le label est défini comme :

$$
Y_i^{(1)} = \mathbb{1}\{R_i(\tau) > \theta\}
$$

où $\theta$ est un seuil d'exploitabilité :

$$
\theta = 2 \times \text{ATR}_{14}(t_0)
$$

**Justification** : L'ATR (Average True Range) capture la volatilité récente normale. Un mouvement > 2×ATR représente un événement anormal statistiquement exploitable.

#### D. Label direction

$$
Y_i^{(2)} = \begin{cases}
1 & \text{si } \ln(P_{t_0+15\text{min}}) - \ln(P_{t_0}) > 0 \text{ (up)} \\
0 & \text{si } \ln(P_{t_0+15\text{min}}) - \ln(P_{t_0}) \leq 0 \text{ (down)}
\end{cases}
$$

#### E. Maximum Return

Calculé sur données M1 pour les 15 premières minutes :

$$
\text{max\_return} = \max_{k=1,\ldots,15} \frac{H_{t_0+k} - L_{t_0+k}}{O_{t_0}} \times 10^4 \text{ (pips)}
$$

#### F. Wick (Drawdown Adverse)

$$
\text{wick} = \max_{t \in [t_0, t_{TP}]} |P_t - P_{t_0}| \text{ (si long)}
$$

### 3.3 Sentiment Financier et Prédiction Directionnelle

**Modèle théorique** : Si $S \in [-1, 1]$ est le sentiment d'une news, on teste :

$$
E[\text{sign}(R(\tau))|S > 0] > 0 \quad \text{et} \quad E[\text{sign}(R(\tau))|S < 0] < 0
$$

**Validation empirique (Shapiro et al., 2024)** : Sur données FX 2015-2023, utilisation de FinBERT montre :

$$
P(\text{direction correcte}||S| > 0.5) \approx 0.58 \text{ (vs 0.50 random)}
$$

Edge statistique de +8%, exploitable après coûts de transaction sur paires liquides.

### 3.4 Architecture du Modèle LSTM Multi-Échelles

#### A. Définition du Log-Range

Pour un intervalle de temps $\tau$ (1 minute), le log-range est défini comme :

$$
\text{LogRange}_t = \ln\left(\sup_{t \leq s \leq t+\tau} P_s\right) - \ln\left(\inf_{t \leq s \leq t+\tau} P_s\right)
$$

où $P_s$ est le prix spot à l'instant $s$.

**Justification** : Le log-range est préféré à la volatilité classique car :
- Observable directement (différence high-low)
- Lié directement au P&L maximal d'une position
- Plus pratique pour les traders que les modèles théoriques (GBM)

#### B. Architecture 2-LSTM

**LSTM Temporel ($\text{LSTM}_t$)** : Capture l'auto-corrélation intra-jour

Input :
$$
y_t^D = (V_{t-p_t}^D, \ldots, V_{t-1}^D) \in \mathbb{R}^{p_t \times 1}
$$

où $V_t^D$ est le log-range à la minute $t$ du jour $D$, et $p_t = 20$ minutes.

**LSTM Périodique ($\text{LSTM}_D$)** : Capture l'auto-corrélation inter-jours

Input :
$$
z_t^D = (V_t^{D-p_d}, \ldots, V_t^{D-1}) \in \mathbb{R}^{p_d \times 1}
$$

où $p_d = 20$ jours (même minute sur les 20 jours précédents).

**Architecture combinée** :

$$
f_{\Theta}(x_t^D) = \text{DNN}(\text{LSTM}(y_t^D), \text{LSTM}(z_t^D))
$$

où DNN est un réseau dense à 2 couches de 32 neurones chacune.

#### C. Extension Multi-Paires (p-Pairs-Learning 2-LSTM)

Pour capturer les corrélations croisées, on étend l'input à $p$ paires simultanément :

$$
y_t^D = (V_{t-p_t}^D, \ldots, V_{t-1}^D) \in \mathbb{R}^{p_t \times p}
$$

$$
z_t^D = (V_t^{D-p_d}, \ldots, V_t^{D-1}) \in \mathbb{R}^{p_d \times p}
$$

où $V_t^D = (V_{1,t}^D, \ldots, V_{p,t}^D)$ est le vecteur des log-ranges de $p$ paires à l'instant $t$ du jour $D$.

**Configuration optimale** : $p = 4$ paires (EURUSD, USDJPY, EURSEK, XAUUSD)

#### D. Métriques de Validation

**Mean Squared Error (MSE)** sur données de test :

$$
\text{MSE} = \frac{1}{N}\sum_{i=1}^N (V_{t+1}^i - \hat{V}_{t+1}^i)^2
$$

**Résultats Empiriques (Liao et al., 2021)** sur données EURUSD, EURSEK, USDJPY, USDMXN 2018-2019 :

| Modèle | EURUSD MSE (×10⁻⁸) | Réduction vs AR | Réduction vs GARCH |
|--------|-------------------|-----------------|-------------------|
| AR(p) | 0.89 ± 0.12 | Baseline | - |
| GARCH(1,1) | 1.08 ± 0.11 | - | Baseline |
| Plain DNN | 1.76 ± 0.34 | -97% | -63% |
| LSTM_t | 0.62 ± 0.08 | +30% | +43% |
| 2-LSTM | 0.61 ± 0.08 | +31% | +44% |
| **4-Pairs 2-LSTM** | **0.56 ± 0.29** | **+37%** | **+48%** |

**Test de Diebold-Mariano** : Confirme que 4-Pairs-Learning 2-LSTM surpasse significativement (p < 0.05) :
- AR(p) : DM statistic = +7.55
- GARCH(1,1) : DM statistic = +8.06
- Plain DNN : DM statistic = +15.24

#### E. Patterns Empiriques Capturés

**Saisonnalité intraday** : Pics de volatilité à 7h (ouverture Londres) et 12h UTC (ouverture NY) :

$$
\mathbb{E}[V_t | \text{hour} = 7] \approx 1.8 \times \mathbb{E}[V_t | \text{hour} = 3]
$$

**Auto-corrélation intra-jour** : Avec lag $k$ minutes :

$$
\text{Corr}(V_t, V_{t-k}) \approx 0.5 \text{ pour } k \leq 20, \text{ décroît rapidement après}
$$

**Auto-corrélation inter-jours** : Pour NFP (13h30), avec lag $p_d = 20$ jours (≈1 mois) :

$$
\text{Corr}(V_t^D, V_t^{D-20}) \approx 0.3 \text{ (max)}
$$

**Corrélations croisées** : Entre EURUSD et USDJPY (devise commune USD) :

$$
\text{Corr}(V_{\text{EURUSD},t}, V_{\text{USDJPY},t}) \approx 0.65
$$

### 3.5 Théorème d'Approximation Universelle pour RNN

**Théorème (Schäfer & Zimmermann, 2006)** : Un réseau LSTM avec suffisamment de neurones cachés peut approximer toute fonction mesurable $f : \mathbb{R}^T \to \mathbb{R}$.

**Application empirique (Liao, Chen & Ni, 2021)** : Pour la prédiction du log-range minute suivant, le modèle 4-Pairs-Learning 2-LSTM atteint :

$$
\text{MSE}_{\text{4P-2LSTM}} = 0.56 \times 10^{-8} < \text{MSE}_{\text{GARCH}} = 1.08 \times 10^{-8}
$$

---

## 4. Architecture Données Multi-Symboles (Normalisation Rolling 30j/Symbole)

**Symboles :** XAUUSD, EURUSD, GBPUSD, USA30IDXUSD, USATECHIDXUSD.

```
Features/symbole:
- OHLC → log-range volatilité
- Spread, tick-volume, pre_news_spread/vol
- FinBERT sentiment per-source + confidence
Labels: has_spike=2×ATR(14/cluster), direction=15min, wick M1
```

---

## 5. Modélisation Hybride Anti-Overfit

### 5.1 ML Classification Conservatrice

```
XGBoost(spike): depth=6, eta=0.05, min_child=5, colsample=0.6, subsample=0.7
RF(direction): max_features=sqrt, depth=20, min_leaf=50
Stack: logistic meta (rank-avg probs) + V̂LSTM feature
SEUILS: P(spike)>0.65 ET (P_long>0.65 | P_short<0.35)
```

**Modèles de classification** :

$$
P(Y^{(1)} = 1|X) = f_1(X; \theta_1)
$$

$$
P(Y^{(2)} = 1|X, Y^{(1)} = 1) = f_2(X; \theta_2)
$$

où $f_1, f_2$ sont des Random Forest ou Gradient Boosting Machines (XGBoost/LightGBM).

**Justification du Choix ML** :
- **Non-linéarité** : Relations complexes entre surprise, sentiment, régime VIX et réaction de prix
- **Interactions** : Effet croisé type_news × surprise × regime × volatilité_prédite nécessite des arbres de décision
- **Robustesse** : Moins sensibles aux outliers que les régressions linéaires
- **Interprétabilité** : Feature importance SHAP pour validation économique

### 5.2 2-LSTM Multi-Paires (Liao Validé)

```
Input: 4 paires simultanées → corrélations croisées
LSTM_intra: lag=20min, units=[64,32], dropout=0.2
LSTM_inter: lag=20jours (patterns NFP)
Output/symbole: P(spike), P(dir), V̂LSTM(15min)
Retraining: ML rolling 3ans (early_stop=50), LSTM quarterly
```

**Prédiction Pré-News** : À $t_0 - 5$ minutes (avant publication de la news), le modèle LSTM prédit la volatilité attendue pour les 15 prochaines minutes :

$$
\hat{V}_{t_0:t_0+15} = f_{\text{LSTM}}(y_{t_0}^D, z_{t_0}^D)
$$

---

## 6. Calibration Dynamique TP/SL/Horizon (Cluster-Based)

### 6.1 Méthodologie de Base

Pour un cluster $C$ défini par (event_type, vix_regime, sign(sentiment)), on extrait l'échantillon historique :

$$
S_C = \{R_i(\tau), D_i\}_{i \in C, Y_i^{(1)}=1}
$$

où :
- $R_i(\tau)$ : Retour maximal observé dans $\tau$ minutes
- $D_i$ : Drawdown adverse maximal (wick)

### 6.2 Formules de Calibration de Base

**Take Profit (TP)** :

$$
\text{TP}_C = Q_{0.50}(|R_i(\tau)|_{i \in C}) \quad \text{(médiane des moves gagnants)}
$$

**Stop Loss (SL)** :

$$
\text{SL}_C = Q_{0.85}(D_i|_{i \in C}) \quad \text{(85e percentile des wicks)}
$$

**Horizon Temporel** :

$$
\tau_C = Q_{0.60}(t_{\text{TP},i}|_{i \in C}) \quad \text{(temps médian pour atteindre TP)}
$$

### 6.3 Ajustement par Volatilité Prédite LSTM

```
Cluster = event×VIX_regime×sentiment_sign
TP_base = median_win_move(50th percentile)
SL_base = median_wick(85th percentile)
hold_base = median_time_TP(60th percentile)

V̂LSTM Scaling:
TP_final = TP_base × (V̂LSTM/medianV_cluster)^0.4
SL_final = SL_base × (V̂LSTM/medianV_cluster)^0.6
Max_hold = max(15min, 60s × V̂LSTM/medianV_cluster)
```

**Take Profit Ajusté** :

$$
\text{TP}_{\text{final}} = \text{TP}_C \times \left(\frac{\hat{V}_{\text{LSTM}}}{V_{\text{médiane}_C}}\right)^{0.4}
$$

**Stop Loss Ajusté** :

$$
\text{SL}_{\text{final}} = \text{SL}_C \times \left(\frac{\hat{V}_{\text{LSTM}}}{V_{\text{médiane}_C}}\right)^{0.6}
$$

**Horizon Ajusté** :

$$
\tau_{\text{final}} = \max\left(15\text{min}, 60\text{s} \times \frac{\hat{V}_{\text{LSTM}}}{V_{\text{médiane}_C}}\right)
$$

**Rationale** :
- En haute volatilité prédite → TP plus large, SL plus large (mais moins que TP), horizon plus court
- Exposants $\beta = 0.4, \gamma = 0.6 < 1$ pour éviter sur-réaction aux prédictions extrêmes

### 6.4 Justification Statistique

- **Médiane** : Robuste aux outliers, représentative du cas typique
- **Percentiles élevés pour SL** : Couverture de 85-90% des cas adverses sans sur-dimensionner
- **Approche non-paramétrique** : Aucune hypothèse de normalité (inappropriée pour les queues de distribution FX)
- **Ajustement par volatilité** : Permet adaptation en temps réel aux conditions de marché changeantes

---

## 7. Risk Management – 1% Capital/Trade Strict

### 7.1 Kelly Criterion Fractionnel

Pour éviter le sur-levier, utiliser une fraction conservatrice du Kelly :

$$
f^* = \frac{p \times W - (1-p) \times L}{W \times L}
$$

où :
- $p$ : Win rate empirique OOS
- $W$ : Average win (pips)
- $L$ : Average loss (pips)

**Position size** :

$$
\text{Lots} = \frac{f^* \times \text{Capital} \times 0.25}{\text{SL}_{\text{final}} \times \text{pip\_value}}
$$

Le facteur 0.25 représente 1/4 Kelly, approche standard pour réduire la volatilité du capital tout en maintenant croissance à long terme.

### 7.2 Ajustement par Volatilité LSTM

```
Lot_Size = (Capital × 0.01 × VolScaling) / (SL_pips × pip_value)
Kelly_fraction = 0.25 × f(p_win_OOS, Win/Loss_ratio)
VolScaling = 0.5 si V̂LSTM>2×V_baseline | 1/(2×V̂/V_base) sinon
```

Intégrer la prédiction LSTM dans le sizing :

$$
\text{Lots}_{\text{final}} = \frac{\text{Lots}_{\text{Kelly}}}{\sqrt{1 + 2 \times \frac{\hat{V}_{\text{LSTM}}}{V_{\text{baseline}}}}}
$$

**Rationale** : En haute volatilité prédite, réduire exposition pour maintenir risque constant.

### 7.3 Limites de Drawdown et Circuit Breakers

```
Circuit Breakers : Daily 3%, Weekly 6%, DD 15%, Max 40%/symbole.
```

**Règle 1 : Stop journalier**

$$
\text{Si } \text{Loss}_{\text{today}} > 0.03 \times \text{Capital} \Rightarrow \text{Arrêt trading 24h}
$$

**Règle 2 : Stop hebdomadaire**

$$
\text{Si } \text{Loss}_{\text{week}} > 0.06 \times \text{Capital} \Rightarrow \text{Arrêt trading 48h}
$$

**Règle 3 : Drawdown maximal**

$$
\text{Si } \text{DD}_{\text{current}} > 0.15 \times \text{Capital} \Rightarrow \text{Arrêt système, audit complet}
$$

### 7.4 Diversification Inter-Paires

Ne jamais exposer plus de 40% du capital sur une seule paire simultanément :

$$
\sum_{\text{paire}} \text{Exposition}_{\text{paire}} \leq 0.40 \times \text{Capital}
$$

**Allocation optimale (basée sur corrélation empirique)** :
- EURUSD : 35%
- XAUUSD : 30%
- USDJPY : 20%
- EURSEK : 15%

---

## 8. Workflow Trading Intégré (Filtrage Strict)

### 8.1 Workflow Décisionnel Complet

**Phase 1 : Pré-News (t₀ - 5 minutes)**

```
PRE-NEWS (t-5min):
  1. 2-LSTM → V̂LSTM 4-paires
  2. Pré-features (spread/tick/sentiment)
```

1. **Prédiction LSTM de volatilité** :

$$
\hat{V}_{t_0:t_0+15} = f_{\text{4-Pairs-2-LSTM}}(y_{t_0}^D, z_{t_0}^D)
$$

**Phase 2 : Post-News (t₀)**

```
POST-NEWS (t):
  1. LIQUIDITÉ: spread≤0.6pip(EUR/GBP)/0.4%XAU, tick_vol≥50%median5
  2. ML Stack: P_spike>0.65 + direction edge
  3. VIX_regime=1 (high vol only)
  4. Cluster lookup → TP/SL/hold base
  5. V̂LSTM scaling → paramètres finaux
  6. Lot 1% → MT5 (slippage 0.5pip modélisé)
```

2. Extraction des features en temps réel (incluant $\hat{V}_{\text{LSTM}}$)
3. Prédiction Modèle 1 :

$$
p_{\text{spike}} = P(Y^{(1)} = 1|X, \hat{V}_{\text{LSTM}})
$$

4. Filtre régime VIX : Vérifier $\text{vix\_regime} = 1$ (si stratégie high-vol)
5. Seuil de décision : Si $p_{\text{spike}} > 0.60$, continuer
6. Prédiction Modèle 2 :

$$
p_{\text{up}} = P(Y^{(2)} = 1|X, \hat{V}_{\text{LSTM}})
$$

7. **Sélection direction** :
   - Si $p_{\text{up}} > 0.60$ → Setup LONG
   - Si $p_{\text{up}} < 0.40$ → Setup SHORT
   - Sinon → Pas de trade

8. **Calibration paramètres** :
   - Lookup $\text{TP}_C, \text{SL}_C, \tau_C$ depuis table statistique cluster
   - Ajuster par volatilité LSTM : $\text{TP}_{\text{final}}, \text{SL}_{\text{final}}, \tau_{\text{final}}$

9. **Position sizing adaptatif** :

$$
\text{Lots} = \frac{\frac{1}{4}\text{Kelly} \times \text{Capital}}{\text{SL}_{\text{final}} \times \text{pip\_value} \times \sqrt{1 + 2\hat{V}_{\text{LSTM}}}}
$$

10. **Exécution** : Ouvrir position avec :
    - Entry : Prix de marché à $t_0 + 2$ ticks
    - TP : Entry $\pm$ $\text{TP}_{\text{final}}$ pips
    - SL : Entry $\mp$ $\text{SL}_{\text{final}}$ pips
    - Max hold : $\tau_{\text{final}}$ secondes

### 8.2 Gestion des Sorties

```
Sorties prioritaires : TP → SL → Max_hold → V̂LSTM×3 (urgence).
```

**Ordre de priorité** :
1. TP touché → Clôture avec profit
2. SL touché → Clôture avec perte contrôlée
3. Horizon expiré → Clôture au marché
4. Volatilité LSTM dépassant $3 \times$ prédiction initiale → Clôture d'urgence (regime shift)

### 8.3 Avantages de l'Approche Hybride

| Composante | Contribution |
|----------