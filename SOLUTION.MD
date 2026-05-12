# SMILES-2026 – Self interference cancellation solution

## my result

| Metric                | Baseline result in db | my result in db |
| --------------------- | --------------------- | --------------- |
| **Average reduction** | 4.02                  | **11.66**       |
| Channel 0             | 3.98                  | 12.80           |
| Channel 1             | 4.86                  | 11.32           |
| Channel 2             | 3.49                  | 9.18            |
| Channel 3             | 3.74                  | 13.33           |

---

### Setup

```bash
pip install numpy scipy
#or
pip install -r requirements.txt
then: 
```
```bash
python applicant_solution.py
```

---

## which techniques i used:

| Feature                  | Implementation                                            |
| ------------------------ | --------------------------------------------------------- |
| Statistical Selection    | 99th percentile masking to isolate interference bursts.   |
| Dimensionality Reduction | EVD-based spatial filtering (similar to PCA).             |
| Regularization           | Tikhonov-like diagonal loading for covariance stability.  |
| Iterative Processing     | Two-pass approach to handle non-linearities or residuals. |

## failed attempts:
firstly i tried to make linear model by scipy.linalg, MLP and CNN models, 
but they didnt pass the check or had worse results than baseline model. then 
i started to dive into self interference cancellation and it led me to my result!
At first i tried tikhonov-like regularization then added evd filter and 99th percentile and it gave me 7.3 db.
Finally i added two pass approach and result were 11+ db!
