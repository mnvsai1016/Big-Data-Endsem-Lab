# Preprocessing & Visualization Report
**Course:** CS 5010 - Analytics & Systems of Big Data Practice  
**Student Name:** M Naga Venkata Sai  
**Roll Number:** CS23I1016  
**Role:** Preprocessing & Visualization Engineer  

---

## 1. Executive Summary & Dataset Overview
- **Raw Dataset:** `datasets/fifa21_male2.csv` (17,125 rows, 107 columns)
- **Preprocessed Output:** `datasets/fifa21_preprocessed.csv` (17,125 rows, 53 selected features)
- **Data Completeness:** 100% of rows preserved, nulls imputed using position medians.

---

## 2. Preprocessing & Data Cleaning Operations
1. **Financial Valuation Standardization:**
   - Stripped `€` symbol, parsed multiplier suffixes (`M` -> $10^6$, `K` -> $10^3$).
   - Fields converted: `Value`, `Wage`, `Release_Clause`.
2. **Biometric Normalization:**
   - `Height`: Parsed $ft'in''$ strings to numeric centimeters (`Height_cm`). Dataset mean: **181.23 cm**.
   - `Weight`: Parsed $lbs$ strings to numeric kilograms (`Weight_kg`). Dataset mean: **75.13 kg**.
3. **Reputation & Skill Traits:**
   - `W/F`, `SM`, `IR`: Removed star characters (`★`) and converted to discrete integer ratings (1 to 5).
   - `Hits`: Converted strings including `1.2K` to float.
4. **Contract Temporal Parsing:**
   - Extracted `Joined_Year`, `Contract_Start`, `Contract_End`, and `Contract_Duration_Years`.
5. **Tactical Role Grouping:**
   - Mapped `BP` into 4 high-level tactical archetypes:
     - **Forward** (ST, CF, RW, LW): 3,287 players
     - **Midfielder** (CAM, CM, CDM, RM, LM): 6,691 players
     - **Defender** (CB, LB, RB, LWB, RWB): 5,571 players
     - **Goalkeeper** (GK): 1,576 players
6. **Discretization & Binning:**
   - `OVA_Tier`: Elite (85+), High (75-84), Mid (65-74), Developing (<65).
   - `Age_Group`: U-21, 22-29 Peak, 30+ Veteran.
   - `Value_Tier`: Superstar (€20M+), High Value (€5M-€20M), Mid Value (€1M-€5M), Developing (<€1M).
7. **Skewness & Variance Stabilization Transformation:**
   - Applied $Log\_Value = \ln(1 + \mathrm{Value})$ and $Log\_Wage = \ln(1 + \mathrm{Wage})$.
   - Raw Market Value Skewness: **+6.268** (Kurtosis: **+60.386**).
   - Active Market Value Log Skewness: **+0.289** (Kurtosis: **-0.057**).

---

## 3. High-Resolution Visualizations & Analytical Inferences (All 10 Plots)

| Fig # | File Name | Plot Type | Key Analytical Inference |
|---|---|---|---|
| **Fig 1** | `fig1_doughnut_tiers.png` | Doughnut Chart | Only **0.6% (109 players)** reach the Elite 85+ tier. 50.2% reside in Mid (65-74) and 36.0% in Developing (<65). |
| **Fig 2** | `fig2_pareto_nationalities.png` | Pareto Chart | The Top 5 nations (ESP, ENG, FRA, BRA, GER) capture **47.8%** of global talent market value; top 14 nations account for >75%. |
| **Fig 3** | `fig3_radar_positions.png` | Radar / Spider Chart | Distinct role archetypes: Forwards dominate Pace/Shooting, Midfielders lead in Passing/Dribbling, Defenders in Physicality/Defending. |
| **Fig 4** | `fig4_area_wage_age.png` | Area Chart | Total weekly wage expenditure peaks at **Age 28 (€17.67M/week)**. Ages 23-29 consume >65% of global wage budgets. |
| **Fig 5** | `fig5_correlation_heatmap.png` | Correlation Heatmap | Overall rating is highly correlated with Reactions ($r=0.87$), Composure ($r=0.71$), Value ($r=0.63$), and Short Passing ($r=0.50$). |
| **Fig 6** | `fig6_violin_value_positions.png` | Violin Plot | Attacking positions (CF, ST, LW, RW) command significant market valuation premiums compared to Fullbacks and Goalkeepers. |
| **Fig 7** | `fig7_scatter_age_growth.png` | Polynomial Scatter & Envelope | Peak career rating occurs at **Age ~28.2 (mean OVA ~71.4)**. Development headroom reaches up to +18 OVA for teenagers. |
| **Fig 8** | `fig8_treemap_clubs.png` | Treemap Chart | Elite European club hierarchy led by Liverpool (€852.9M), Real Madrid (€765.7M), Manchester City (€752.8M), and FC Barcelona (€714.1M). |
| **Fig 9** | `fig9_hexbin_speed_agility.png` | Hexbin Density Plot | Sprint Speed and Agility exhibit strong linear correlation ($r=0.76$). Biomechanical clustering separates wingers from central defenders. |
| **Fig 10** | `fig10_hist_kde_skewness.png` | Two-Panel Histogram + KDE | Demonstrates skewness reduction: raw value skewness (**+6.27**) stabilizes under log transformation to **+0.35**, yielding a near-normal Gaussian bell curve. |

---
**Verification:** All 10 plots are rendered at 200 DPI in `report_images/`. Cleaned dataset saved in `datasets/fifa21_preprocessed.csv`.
