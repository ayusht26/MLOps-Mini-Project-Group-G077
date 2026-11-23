"""Stage 2: clean, train 3+ settings with MLflow, register the best as @champion.

Run:  python -m src.tiffin.train
"""

import joblib
from pathlib import Path
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.tiffin.clean import clean_train
from src.tiffin.config import load_params
from src.tiffin.model import build_pipeline, mae
from src.tiffin.utils import get_file_sha256


def main() -> None:
    p = load_params()
    target = p["audit"]["target_column"]
    num = p["audit"]["numeric_features"]
    cat = p["audit"]["categorical_features"]
    feats = num + cat
    t = p["tracking"]

    raw = pd.read_csv(p["data"]["raw_train_path"])
    sha = get_file_sha256(p["data"]["raw_train_path"])
    train = clean_train(raw, p)
    later = pd.read_csv(p["data"]["raw_later_path"])

    # time-ordered validation: last 20% of train by date
    cut = int(len(train) * (1 - p["training"]["valid_fraction"]))
    fit_df, val_df = train.iloc[:cut], train.iloc[cut:]

    mlflow.set_tracking_uri(t["uri"])
    mlflow.set_experiment(t["experiment_name"])

    results = []
    for cfg in p["training"]["runs"]:
        with mlflow.start_run(run_name=cfg["name"]) as run:
            mlflow.log_param("data_sha256", sha)
            mlflow.log_param("model", cfg["model"])
            mlflow.log_params({f"hp_{k}": v for k, v in cfg["params"].items()})
            mlflow.log_param("n_train_rows", len(fit_df))

            pipe = build_pipeline(cfg["model"], cfg["params"], num, cat)
            pipe.fit(fit_df[feats], fit_df[target])
            mae_valid = mae(val_df[target], pipe.predict(val_df[feats]))
            mae_later = mae(later[target], pipe.predict(later[feats]))
            mlflow.log_metric("mae_valid", mae_valid)
            mlflow.log_metric("mae_later", mae_later)

            # refit on ALL cleaned train rows -> this is the model we keep (v1)
            final = build_pipeline(cfg["model"], cfg["params"], num, cat)
            final.fit(train[feats], train[target])
            info = mlflow.sklearn.log_model(final, name="model",
                                            serialization_format="cloudpickle")
            results.append((mae_valid, run.info.run_id, info.model_uri, final, cfg["name"]))
            print(f"{cfg['name']:10s} mae_valid={mae_valid:.3f} mae_later={mae_later:.3f}")

    best_mae, best_run, best_uri, best_model, best_name = min(results, key=lambda r: r[0])
    mv = mlflow.register_model(best_uri, t["registered_model_name"])
    MlflowClient().set_registered_model_alias(
        t["registered_model_name"], t["champion_alias"], mv.version)
    Path(t["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, t["model_path"])

    print("\n--- paste into findings.yaml -> tracking ---")
    print("experiment_name:", t["experiment_name"])
    print("best_run_id:", best_run)
    print("registered_model_name:", t["registered_model_name"])
    print("champion_version:", mv.version)


if __name__ == "__main__":
    main()
