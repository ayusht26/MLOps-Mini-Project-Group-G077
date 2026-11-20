# MLOps Mini Project: Tiffin Relay Delivery Time (Jaipur)

**Group ID**: G077  
**Problem ID**: MP160  
**Batch**: 64  
**Faculty**: Dhananjay Kumar  

---

## 👥 Team Members

| Enrollment No. | Name | GitHub Account | Role |
|---|---|---|---|
| **S24CSEU2318** | Ayush Tiwari | [@ayusht26](https://github.com/ayusht26) | Data & Pipeline Lead |
| **S24CSEU2329** | Harsh Dhiman | [@HarshDhiman](https://github.com/HarshDhiman) | Model & Tracking Lead |
| **S24CSEU2688** | Deepanshi Pokhriyal | [@DeepanshiPokhriyal](https://github.com/DeepanshiPokhriyal) | Service & Ops Lead |

---

## 📌 Problem Overview: MP160

- **Client**: Jaipur Dabba Relay Union (Jaipur)
- **Problem Statement**: Predict `delivery_min` (minutes from pick-up to office desk) for relay-based tiffin deliveries across Jaipur routes (Central, Suburb, Old City).
- **Task Type**: Regression
- **Primary Metric**: MAE (Mean Absolute Error, in minutes)

---

## 📁 Repository Structure

```
.
├── README.md                           # Project metadata, team details, instructions
├── GROUP.md                            # Group details and member mappings
├── PROBLEM.md                          # Specific problem requirements and rules
├── STUDENT_HANDBOOK.md                 # Project handbook and submission rubric
├── requirements.txt                    # Pinned library dependencies
├── params.yaml                         # Central pipeline configuration parameters
├── findings.yaml                       # Findings and audit form (root)
├── reports/
│   └── findings.yaml                   # Findings report copy
├── .dvc/                               # DVC configuration
├── data/
│   └── raw/
│       ├── train.csv.dvc               # DVC tracked training data
│       ├── later.csv.dvc               # DVC tracked subsequent data
│       └── holdout_features.csv        # Holdout test features (tracked in git)
├── src/
│   └── tiffin/
│       ├── __init__.py                 # Package initializer
│       ├── config.py                   # Configuration loader for params.yaml
│       ├── utils.py                    # Hashing and data helper utilities
│       └── data_audit.py               # Automated audit for all 7 data anomalies
└── tests/
```

---

## 🛠️ Stage 1: Data Versioning & Anomaly Audit

### 1. Data Version Control (DVC)
Data files are tracked with DVC to avoid storing large CSVs in Git:
- Raw data files are placed under `data/raw/`.
- `data/raw/train.csv` and `data/raw/later.csv` are tracked via `.dvc` pointers (`train.csv.dvc`, `later.csv.dvc`).
- The test features file `data/raw/holdout_features.csv` is committed to git directly.

### 2. The Seven Data Issues Audit
As per the handbook, `train.csv` was analyzed for seven potential data anomalies:
1. **`leaking_column`** (Present): `delivered_min_from_receipt` contains post-delivery receipt records ($r \approx 0.9991$ with target, absent from holdout).
2. **`unit_mixup`** (Present): `wait_at_station_min` in the `Central` zone was recorded in seconds instead of minutes ($\text{factor} = 60$).
3. **`sentinel_value`** (Absent): No impossible sentinel values (e.g. -1, -999, 9999) detected.
4. **`duplicate_rows`** (Absent): 0 extra duplicate rows found.
5. **`stuck_sensor`** (Present): `rider_age` is stuck at `33` for 104 consecutive records from `2026-03-08` to `2026-03-29`.
6. **`target_typos`** (Absent): No $10\times$ typo anomalies in `delivery_min`.
7. **`spelling_variants`** (Absent): Text categories in `transport` and `route_zone` match `later.csv` exactly.

To execute the data audit script locally:
```bash
python -m src.tiffin.data_audit
```

---

## 🚀 Setup & Installation

```bash
# Clone the repository
git clone https://github.com/ayusht26/MLOps-Mini-Project-Group-G077.git
cd MLOps-Mini-Project-Group-G077

# Install dependencies
pip install -r requirements.txt

# Run Stage 1 Data Audit
python -m src.tiffin.data_audit
```
