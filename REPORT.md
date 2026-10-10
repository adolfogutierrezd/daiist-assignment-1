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
| False negative | Not contacted, leaves → we lose 12 months of **their own** bill | 12 × bill |
| True positive | Contacted and was going to leave → offer works 50% of the time | €150 + 50% × 12 × bill |

**Why 12 × each customer's own bill and not a flat amount.** Losing a
customer who pays €110/month hurts much more than losing one who pays €20,
so a flat loss would tell the team to chase cheap customers just as hard as
valuable ones. My first version used a flat €900; the average churner pays
€74.4/month, so 12 × €74.4 = €893. The flat number was a fair average, but
making it per-customer is more honest about where the money is. The bill is
kept aside for the cost only; it is not a model feature (see below).

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
   pays off when the revenue the offer is expected to save is larger than
   the offer: `p × 50% × 12 × bill > €150`, i.e. **p > 25 / bill**. For a
   typical €75/month customer that is **p > 0.33**, so I expect the best
   threshold to be well below the default 0.5. It also means a customer
   paying under €25/month (19.7% of customers) is **never** worth a €150
   offer, however likely they are to leave.
4. **Metric to compare models** — ROC AUC (ranking quality, independent of
   threshold) and log loss (what logistic regression actually minimises, so
   it's the fairest check that the three implementations agree).

## Data preparation & feature engineering

All of it lives in `train.py`, section 2.

**Data-quality decisions**

- **`TotalCharges` blanks.** The column is stored as text and 11 rows are
  blank. All 11 have `tenure = 0`: they are brand-new customers who haven't
  been billed yet, so the correct value is **0**, not a missing value to drop
  or fill with the mean.
- **`customerID` dropped**: it identifies a row but says nothing about churn.

**Encoding decisions**

- **Redundant categories.** Service columns have three values: Yes / No /
  "No internet service" (or "No phone service"). The third value only repeats
  what `InternetService` / `PhoneService` already say, so each service is
  turned into a single **1/0 flag** instead of two dummy columns.
- **Remaining categoricals** (`InternetService`, `Contract`, `PaymentMethod`)
  are one-hot encoded with `drop_first` so the columns aren't redundant.
- **Standardisation** (`StandardScaler`) is **fitted on the training set only**
  and then applied to validation/test, so no information leaks from them.
  It also matters for gradient descent: without it, large-valued columns
  dominate the gradient and the PyTorch versions converge much more slowly.

**Engineered features**

| Feature | Why |
|---|---|
| `log_tenure = log(1 + tenure)` | Churn falls fast in the first months then flattens (53% churn in months 0–6, 36% in 7–12, 10% after 4 years). A linear model can only draw a straight line in raw tenure; the log matches that curve. |
| `no_support` | Internet customer with neither online security nor tech support: nobody to call when something breaks. |
| `bill_increase = MonthlyCharges − TotalCharges / tenure` | Today's bill minus the customer's historical average bill. A price increase is a typical churn trigger, and it replaces `TotalCharges`, which is roughly `tenure × MonthlyCharges` and so adds nothing new. |

**Features I removed, and why (found through the three-method comparison)**

- **`num_services`** (number of add-on services). My first version had it, but
  it is *exactly* the sum of 7 columns already in the model. A linear model
  gains nothing from a sum of its own inputs, and the coefficients stop
  being unique (see the Modeling section).
- **`MonthlyCharges`**. A linear regression of `MonthlyCharges` on the service
  flags has **R² = 0.9988** (on all 7,043 customers): the bill is basically a price list (phone ≈ €20,
  fiber ≈ +€25, each streaming service ≈ €10, each add-on ≈ €5). Keeping it
  next to the services gives the model two ways to say the same thing. It
  is still used indirectly, inside `bill_increase`.

**Did it help? (validation set, same model and C)**

| Version | Columns | Val log loss | Val ROC AUC |
|---|---|---|---|
| Naive: one-hot everything, raw numbers | 30 | 0.4013 | 0.860 |
| Final features but raw `tenure` | 23 | 0.4014 | 0.859 |
| Final features without `no_support` and `bill_increase` | 21 | 0.3961 | 0.863 |
| Final + `month-to-month × fiber` interaction (tested, not kept) | 24 | 0.3954 | 0.864 |
| **Final** | **23** | **0.3954** | **0.864** |

Honest reading: **`log_tenure` is the change that matters**. The two new
behavioural features help only a little. The biggest win of the cleaning is
not the score but a **smaller, non-redundant, interpretable model** whose
coefficients are stable (largest coefficient 0.78 instead of 1.58).

**A feature I tested and rejected.** Month-to-month contracts and fiber
internet are the two strongest risk signals, and customers with both churn
at 55% (30% of all customers). I expected an interaction flag for that
group to help, but validation log loss stayed at 0.3954: the model already
gets this group right by adding the two separate effects, so the extra
column only adds complexity. I left it out.

## Modeling: three implementations, one model

**Model: logistic regression**, because the target is binary (churn yes/no)
and the business needs a **probability** to compare with a threshold.

**Same model, three implementations, same split (train 4,225 / val 1,409 / test 1,409):**

1. **scikit-learn** `LogisticRegression` (L2 penalty, lbfgs solver).
2. **Manual PyTorch**: a weight vector `w` and bias `b` as raw tensors,
   `sigmoid(Xw + b)`, binary cross-entropy written by hand, `loss.backward()`,
   then a manual update `w -= lr * w.grad` inside `torch.no_grad()`.
3. **Standard PyTorch**: `nn.Module` with one `nn.Linear` layer,
   `BCEWithLogitsLoss`, `torch.optim.SGD`.

**Making them minimise the same function.** sklearn minimises
`C × Σ log loss + ½‖w‖²`. Dividing by `C × n` gives
`mean log loss + (λ/2)‖w‖²` with **λ = 1 / (C × n)**. I use this λ in both
PyTorch versions (by hand in the manual loop, and as `weight_decay` on the
weights only, not the bias, in `torch.optim`). Without this step the three
would be solving slightly different problems.

**Tuning (sparingly, on validation log loss):**
- `C ∈ {0.01, 0.1, 1, 10}` → **C = 10**, but C = 1 and C = 10 both give a
  val log loss of 0.3954 (only C = 0.01 is clearly worse, 0.4014), so
  regularisation barely matters with 23 features and 4,225 rows.
- learning rate `∈ {0.01, 0.1, 1}`, 3,000 full-batch epochs → **lr = 0.1** for
  both (0.01 had not converged: val log loss 0.4000 manual / 0.4005 standard,
  vs 0.3954 for both at lr = 0.1).

**Test-set results** (threshold 0.38, picked on validation; cost = € on 1,409 customers):

| Model | ROC AUC | Log loss | Accuracy @0.5 | Business cost @0.38 |
|---|---|---|---|---|
| scikit-learn | 0.8402 | 0.4206 | 0.797 | €288,323 |
| Manual PyTorch | 0.8403 | 0.4207 | 0.798 | €288,323 |
| Standard PyTorch | 0.8402 | 0.4206 | 0.797 | €288,323 |
| Naive baseline (everyone = 26.5%) | 0.5000 | 0.5787 | 0.735 | €337,971 (contacts nobody) |

**Do the three agree? Yes.** The largest coefficient difference from
sklearn is **0.004** and the largest probability difference on any test
customer is **0.0013**. The tiny remaining gap comes from gradient descent
stopping after 3,000 steps while lbfgs converges to a tighter tolerance.

**They did *not* agree at first, and that taught me something.** In my first
version the predictions matched (same AUC) but some coefficients were far
apart: `MonthlyCharges` was −1.34 in sklearn vs −0.32 in PyTorch, and
fiber 0.99 vs 0.57. The cause was multicollinearity: `num_services` was an
exact sum of other columns and `MonthlyCharges` was 99.9% explained by the
services. That creates a flat direction in the loss, where many different
weight vectors give the same predictions. lbfgs lands on one point of it;
gradient descent moves along it very slowly and stops elsewhere. Removing
the two redundant features made the solution unique and the coefficients
matched, with no loss in accuracy.

**Business result.** Using the model at threshold 0.38 costs **€288,323**
vs **€337,971** for doing nothing: a saving of **€49,648 (−14.7%)** on 1,409
customers. It contacts 444 customers and catches 248 of the 374 churners.
The default 0.5 threshold would cost €292,421, and contacting everyone
€380,336. The best threshold found on validation (0.38) is close to the
theoretical break-even for a typical €75/month customer (0.33), which is a
good sign that the probabilities are reasonably well calibrated.

## Limitations & next steps

- **The costs are my assumptions** (€150 offer, 12 months of bill lost, 50%
  offer success). The threshold and the € figures depend directly on them.
  *Fix:* get the real offer cost and acceptance rate from a small A/B test
  of the retention campaign, then re-run `train.py`; the threshold is
  recomputed automatically.
- **One threshold for everyone.** The cost already uses each customer's own
  bill, but the decision still uses a single probability threshold, which
  the brief's slider requires. Since the break-even is `p > 25 / bill`, the
  right cut-off is different for each customer. On the test set, contacting
  whoever has `p × 6 × bill > €150` would cost **€280,704**, another €7.6k
  less than the best single threshold. *Fix:* rank customers by expected
  saving (`p × 6 × bill − 150`) instead of by probability alone.
- **One snapshot, no dates.** I can't check whether the model still works
  next month, and a random split may be optimistic. *Fix:* with monthly
  snapshots, train on months 1…k and test on month k+1 (time-based split).
- **Validation vs test gap.** Validation AUC is 0.864 but test AUC is 0.840.
  With only 1,409 customers per set, part of this is noise. *Fix:* use
  5-fold cross-validation to choose C and the threshold so they don't
  depend on one particular validation split.
- **Linear model limits.** It can't learn interactions unless I build them
  by hand. I tried the most obvious one (month-to-month × fiber) and it
  didn't help, which suggests the remaining error isn't about that pair.
  *Fix:* look at the customers the model gets most wrong (high-probability
  stayers, low-probability churners) to find what they have in common.
- **My behavioural features were weak.** `bill_increase` has a coefficient
  close to 0 (−0.012), and removing it alone only moves val log loss from
  0.3954 to 0.3956. Its signal is likely already carried by tenure and
  contract. I would drop it in a production version to keep the model simpler.

## Generative AI use disclosure

I used **Claude (Anthropic, Claude Code)** as a coding assistant to:
read the assignment brief, download the dataset, write the first version
of `train.py` and `app.py` (loading, training loops, saving models, Plotly
charts and the Gradio layout), run the pipeline, and draft this report's
structure and wording.

I reviewed every decision (the business framing and cost assumptions, the
split, each engineered or removed feature, and the explanation of why the
three implementations first disagreed), and I can explain each of them
without the code in front of me.
