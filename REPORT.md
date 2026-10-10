# Assignment 1 Report

- **Name**: Adolfo Gutierrez de Gandarilla Domínguez
- **Student ID**: 19003
- **Email**: fgutierrezde.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

**IBM Telco Customer Churn**, downloaded from IBM's public sample-data repo
([source](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)),
committed as `data/telco_churn.csv` (948 KB).

- **One row = one customer** of a fictional telecom company, at one point in time.
- **7,043 rows × 21 columns**: a customer ID, demographics (gender, senior
  citizen, partner, dependents), account info (tenure in months, contract
  type, payment method, paperless billing, monthly and total charges), the
  services the customer has (phone, internet, security, backup, streaming…)
  and the target, `Churn` (Yes/No: did the customer leave last month).
- **Class balance**: 1,869 churners (26.5%) vs 5,174 non-churners (73.5%).

I picked it because churn is a real decision companies make every month, the
target is clearly binary (→ logistic regression), and the raw columns are
messy enough that feature engineering matters (text numbers with blanks,
redundant "No internet service" categories, nearly collinear charges).

## Business / real-life framing

**Scenario.** Every month the retention team of the telecom picks which
customers get a **retention offer** (a discount worth about €150 over a
year). The model gives each customer a churn probability; the team contacts
everyone above a threshold. The question the model answers is: *"Who should
we spend the retention budget on?"*

**Business costs (my assumptions, per customer):**

| Outcome | Meaning | Cost |
|---|---|---|
| True negative | Not contacted, stays | €0 |
| False positive | Contacted but would have stayed → offer wasted | €150 |
| False negative | Not contacted, leaves → lose ~12 months × ~€75 bill | €900 |
| True positive | Contacted and was going to leave → offer works 50% of the time | €150 + 50% × €900 = €600 |

**What this framing decides in the pipeline:**

1. **Target definition** — `Churn = Yes` → 1, `No` → 0. It's already the
   event we care about, so no re-labelling is needed.
2. **Split strategy** — the data is a single snapshot with **no dates**, so
   there is no "future" that a random split could leak into training. A
   **stratified random split 60% train / 20% validation / 20% test** is
   appropriate (stratified so all three keep the 26.5% churn rate). If the
   company had monthly snapshots I would use a time-based split instead.
3. **Metric for the threshold** — not accuracy: a model that predicts "nobody
   churns" is already 73.5% accurate and saves no money. The threshold is
   chosen to **minimise total business cost on the validation set**, and the
   test set is only used once at the end. In theory, contacting a customer
   pays off when `p × (50% × €900) > €150`, i.e. **p > 0.33**, so I expect
   the best threshold to be well below the default 0.5.
4. **Metric to compare models** — ROC AUC (ranking quality, independent of
   threshold) and log loss (what logistic regression actually minimises, so
   it's the fairest check that the three implementations agree).

## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
