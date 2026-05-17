"""
GUADALAHACKS 2026
Entrenamiento desde vectores de caracteristicas de audio.
"""

import copy
import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler


OUTPUT_DIR = Path("./best_model")
DATA_FORMAT = "csv"
SEED = 42
BATCH_SIZE = 32
FINAL_EPOCHS = 250
CV_EPOCHS = 60
PATIENCE = 18
STATE_LOSS_WEIGHT = 0.75
RUN_CROSS_VALIDATION = True
CV_FOLDS = 5
THRESHOLDS = np.arange(0.05, 0.96, 0.05)

IGNORE_LABELS = {
    "no_oil_serpentine_belt",
    "power_steering_combined_no_oil",
    "power_steering_combined_no_oil_serpentine_belt",
    "power_steering_combined_serpentine_belt",
}

STATE_ALLOWED_LABELS = {
    "startup": [
        "normal_engine_startup",
        "bad_ignition",
        "dead_battery",
    ],
    "braking": [
        "normal_brakes",
        "worn_out_brakes",
    ],
    "idle": [
        "normal_engine_idle",
        "low_oil",
        "power_steering",
        "serpentine_belt",
    ],
}

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {DEVICE}")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features_csv():
    df = pd.read_csv("data/mfcc_features.csv")
    if "label" not in df.columns:
        raise ValueError("data/mfcc_features.csv necesita una columna llamada 'label'.")

    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    if not feature_cols:
        feature_cols = [col for col in df.columns if col != "label"]

    X = df[feature_cols].values.astype(np.float32)
    y_text = df["label"].values
    state_text = df["state"].values if "state" in df.columns else None
    return X, y_text, feature_cols, state_text


def load_features_npz():
    data = np.load("data/features.npz")
    X = data["X"].astype(np.float32)
    y_text = data["y"]
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    state_text = data["state"] if "state" in data.files else None
    return X, y_text, feature_cols, state_text


def load_features_jsonl():
    features, labels = [], []
    with open("data/features.jsonl", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            features.append(item["features"])
            labels.append(item["label"])
    X = np.array(features, dtype=np.float32)
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    return X, np.array(labels), feature_cols, None


def parse_labels(label_value):
    if isinstance(label_value, str):
        return [label.strip() for label in label_value.split(",") if label.strip()]
    return [str(label_value)]


def fit_label_encoder(y_text):
    y_multi = [parse_labels(y) for y in y_text]
    label_combo = np.array([",".join(labels) for labels in y_multi])
    mlb = MultiLabelBinarizer()
    y_encoded = mlb.fit_transform(y_multi).astype(np.float32)
    return y_encoded, list(mlb.classes_), label_combo


def fit_state_encoder(state_text):
    if state_text is None:
        return None, []
    encoder = LabelEncoder()
    y_state = encoder.fit_transform(state_text).astype(np.int64)
    return y_state, list(encoder.classes_)


def stratify_or_none(label_combo):
    counts = pd.Series(label_combo).value_counts()
    return label_combo if counts.min() >= 2 else None


def state_allowed_mask(state_indices, state_names, label_names):
    masks = []
    for state_idx in state_indices:
        state = state_names[int(state_idx)]
        allowed = set(STATE_ALLOWED_LABELS.get(state, label_names))
        masks.append([label in allowed for label in label_names])
    return np.array(masks, dtype=bool)


def apply_thresholds(probs, thresholds, label_names, state_indices=None, state_names=None):
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    preds = (probs >= thresholds).astype(np.int32)

    allowed_mask = None
    if state_indices is not None and state_names is not None:
        allowed_mask = state_allowed_mask(state_indices, state_names, label_names)
        preds[~allowed_mask] = 0

    empty_rows = preds.sum(axis=1) == 0
    if empty_rows.any():
        fallback_probs = probs.copy()
        if allowed_mask is not None:
            fallback_probs[~allowed_mask] = -np.inf
        preds[empty_rows, fallback_probs[empty_rows].argmax(axis=1)] = 1

    if "normal" in label_names:
        normal_idx = label_names.index("normal")
        fault_indices = [i for i in range(len(label_names)) if i != normal_idx]
        has_fault = preds[:, fault_indices].sum(axis=1) > 0
        preds[has_fault, normal_idx] = 0

    return preds


def tune_global_threshold(probs, y_true, label_names):
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in THRESHOLDS:
        preds = apply_thresholds(probs, np.full(y_true.shape[1], threshold), label_names)
        score = f1_score(y_true.astype(int), preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return np.full(y_true.shape[1], best_threshold, dtype=np.float32), best_f1


def tune_per_class_thresholds(probs, y_true, label_names):
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)

    for class_idx in range(y_true.shape[1]):
        best_threshold = 0.5
        best_f1 = -1.0
        y_class = y_true[:, class_idx].astype(int)

        for threshold in THRESHOLDS:
            pred_class = (probs[:, class_idx] >= threshold).astype(int)
            score = f1_score(y_class, pred_class, zero_division=0)
            if score > best_f1 + 1e-8 or (
                abs(score - best_f1) <= 1e-8 and abs(threshold - 0.5) < abs(best_threshold - 0.5)
            ):
                best_f1 = score
                best_threshold = float(threshold)

        thresholds[class_idx] = best_threshold

    preds = apply_thresholds(probs, thresholds, label_names)
    macro_f1 = f1_score(y_true.astype(int), preds, average="macro", zero_division=0)
    return thresholds, macro_f1


def metrics_dict(y_true, probs, thresholds, label_names, state_indices=None, state_names=None):
    y_true_int = y_true.astype(int)
    preds = apply_thresholds(probs, thresholds, label_names, state_indices, state_names)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_int,
        preds,
        average=None,
        zero_division=0,
    )
    confusion = multilabel_confusion_matrix(y_true_int, preds)

    per_class = {}
    for idx, label in enumerate(label_names):
        tn, fp, fn, tp = confusion[idx].ravel()
        per_class[label] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "threshold": float(thresholds[idx]),
        }

    try:
        ap_micro = float(average_precision_score(y_true_int, probs, average="micro"))
        ap_macro = float(average_precision_score(y_true_int, probs, average="macro"))
    except ValueError:
        ap_micro = None
        ap_macro = None

    return {
        "macro_f1": float(f1_score(y_true_int, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true_int, preds, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_int, preds, average="weighted", zero_division=0)),
        "samples_f1": float(f1_score(y_true_int, preds, average="samples", zero_division=0)),
        "exact_match": float((preds == y_true_int).all(axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true_int, preds)),
        "average_precision_micro": ap_micro,
        "average_precision_macro": ap_macro,
        "per_class": per_class,
    }


class FeatureClassifier(torch.nn.Module):
    def __init__(self, input_size, num_labels, num_states=0):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.LayerNorm(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
        )
        self.fault_head = torch.nn.Linear(32, num_labels)
        self.state_head = torch.nn.Linear(32, num_states) if num_states else None

    def forward(self, x, return_state=False):
        hidden = self.shared(x)
        fault_logits = self.fault_head(hidden)
        if return_state:
            state_logits = self.state_head(hidden) if self.state_head is not None else None
            return fault_logits, state_logits
        return fault_logits


def make_tensors(X_train, y_train, X_val, y_val, state_train=None, state_val=None):
    tensors = (
        torch.tensor(X_train, dtype=torch.float32, device=DEVICE),
        torch.tensor(y_train, dtype=torch.float32, device=DEVICE),
        torch.tensor(X_val, dtype=torch.float32, device=DEVICE),
        torch.tensor(y_val, dtype=torch.float32, device=DEVICE),
    )
    if state_train is None or state_val is None:
        return tensors + (None, None)
    return tensors + (
        torch.tensor(state_train, dtype=torch.long, device=DEVICE),
        torch.tensor(state_val, dtype=torch.long, device=DEVICE),
    )


def predict_probs(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        return torch.sigmoid(model(X_t)).cpu().numpy()


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    label_names,
    max_epochs,
    seed,
    state_train=None,
    state_val=None,
    num_states=0,
):
    set_seed(seed)
    use_state_head = state_train is not None and state_val is not None and num_states > 0
    model = FeatureClassifier(X_train.shape[1], len(label_names), num_states if use_state_head else 0).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-4
    )

    positive_counts = y_train.sum(axis=0)
    negative_counts = len(y_train) - positive_counts
    pos_weight = np.where(positive_counts > 0, negative_counts / np.maximum(positive_counts, 1), 1.0)
    pos_weight = np.power(np.clip(pos_weight, 1.0, 15.0), 0.8)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)
    )
    state_criterion = torch.nn.CrossEntropyLoss() if use_state_head else None

    X_train_t, y_train_t, X_val_t, y_val_t, state_train_t, state_val_t = make_tensors(
        X_train,
        y_train,
        X_val,
        y_val,
        state_train,
        state_val,
    )

    best_state = None
    best_epoch = 0
    best_score = -1.0
    best_val_loss = float("inf")
    best_thresholds = np.full(y_train.shape[1], 0.5, dtype=np.float32)
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        losses = []
        indices = torch.randperm(len(X_train_t), device=DEVICE)

        for start in range(0, len(X_train_t), BATCH_SIZE):
            batch_indices = indices[start : start + BATCH_SIZE]
            batch_X = X_train_t[batch_indices]
            batch_y = y_train_t[batch_indices]

            if np.random.random() < 0.3:
                noise = torch.randn_like(batch_X) * 0.05
                batch_X = batch_X + noise

            optimizer.zero_grad()
            if use_state_head:
                logits, state_logits = model(batch_X, return_state=True)
                state_loss = state_criterion(state_logits, state_train_t[batch_indices])
                loss = criterion(logits, batch_y) + STATE_LOSS_WEIGHT * state_loss
            else:
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            if use_state_head:
                val_logits, val_state_logits = model(X_val_t, return_state=True)
                val_loss = (
                    criterion(val_logits, y_val_t)
                    + STATE_LOSS_WEIGHT * state_criterion(val_state_logits, state_val_t)
                ).item()
                val_state_acc = (val_state_logits.argmax(dim=1) == state_val_t).float().mean().item()
            else:
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
                val_state_acc = None
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            thresholds, val_macro_f1 = tune_per_class_thresholds(val_probs, y_val, label_names)
            val_preds = apply_thresholds(val_probs, thresholds, label_names)
            val_micro_f1 = f1_score(y_val.astype(int), val_preds, average="micro", zero_division=0)

        improved = val_macro_f1 > best_score + 1e-5
        if improved:
            best_score = val_macro_f1
            best_val_loss = val_loss
            best_thresholds = thresholds
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == max_epochs - 1 or improved:
            state_text = f" | State acc: {val_state_acc:.4f}" if val_state_acc is not None else ""
            print(
                f"Epoch {epoch+1}/{max_epochs} | Loss: {np.mean(losses):.4f} | "
                f"Val Loss: {val_loss:.4f} | F1 macro: {val_macro_f1:.4f} | "
                f"F1 micro: {val_micro_f1:.4f}{state_text}"
            )

        scheduler.step()

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping en epoch {epoch+1} (mejor epoch: {best_epoch}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "thresholds": best_thresholds,
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_score),
        "best_val_loss": float(best_val_loss),
    }


def scaled_split(X_train, X_val, X_test):
    scaler = StandardScaler()
    return (
        scaler.fit_transform(X_train).astype(np.float32),
        scaler.transform(X_val).astype(np.float32),
        scaler.transform(X_test).astype(np.float32),
        scaler,
    )


def split_train_val_test(X, y, label_combo, y_state=None):
    if y_state is None:
        X_train_val, X_test, y_train_val, y_test, combo_train_val, combo_test = train_test_split(
            X,
            y,
            label_combo,
            test_size=0.2,
            random_state=SEED,
            stratify=stratify_or_none(label_combo),
        )
        X_train, X_val, y_train, y_val, combo_train, combo_val = train_test_split(
            X_train_val,
            y_train_val,
            combo_train_val,
            test_size=0.25,
            random_state=SEED,
            stratify=stratify_or_none(combo_train_val),
        )
        return X_train, X_val, X_test, y_train, y_val, y_test, combo_train, combo_val, combo_test, None, None, None

    (
        X_train_val,
        X_test,
        y_train_val,
        y_test,
        state_train_val,
        state_test,
        combo_train_val,
        combo_test,
    ) = train_test_split(
        X,
        y,
        y_state,
        label_combo,
        test_size=0.2,
        random_state=SEED,
        stratify=stratify_or_none(label_combo),
    )
    X_train, X_val, y_train, y_val, state_train, state_val, combo_train, combo_val = train_test_split(
        X_train_val,
        y_train_val,
        state_train_val,
        combo_train_val,
        test_size=0.25,
        random_state=SEED,
        stratify=stratify_or_none(combo_train_val),
    )
    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        combo_train,
        combo_val,
        combo_test,
        state_train,
        state_val,
        state_test,
    )


def run_cross_validation(X, y, label_combo, label_names, y_state=None, state_names=None):
    if not RUN_CROSS_VALIDATION:
        return None

    print("\n" + "=" * 50)
    print(f"Cross-validation estratificada ({CV_FOLDS} folds)...")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, label_combo), start=1):
        print(f"\nFold {fold}/{CV_FOLDS}")
        X_train_val, X_fold_test = X[train_val_idx], X[test_idx]
        y_train_val, y_fold_test = y[train_val_idx], y[test_idx]
        combo_train_val = label_combo[train_val_idx]
        state_train_val = y_state[train_val_idx] if y_state is not None else None

        if state_train_val is None:
            X_fold_train, X_fold_val, y_fold_train, y_fold_val = train_test_split(
                X_train_val,
                y_train_val,
                test_size=0.25,
                random_state=SEED + fold,
                stratify=stratify_or_none(combo_train_val),
            )
            state_fold_train = None
            state_fold_val = None
        else:
            (
                X_fold_train,
                X_fold_val,
                y_fold_train,
                y_fold_val,
                state_fold_train,
                state_fold_val,
            ) = train_test_split(
                X_train_val,
                y_train_val,
                state_train_val,
                test_size=0.25,
                random_state=SEED + fold,
                stratify=stratify_or_none(combo_train_val),
            )

        X_fold_train, X_fold_val, X_fold_test, _ = scaled_split(
            X_fold_train, X_fold_val, X_fold_test
        )
        result = train_model(
            X_fold_train,
            y_fold_train,
            X_fold_val,
            y_fold_val,
            label_names,
            max_epochs=CV_EPOCHS,
            seed=SEED + fold,
            state_train=state_fold_train,
            state_val=state_fold_val,
            num_states=len(state_names or []),
        )
        probs = predict_probs(result["model"], X_fold_test)
        metrics = metrics_dict(y_fold_test, probs, result["thresholds"], label_names)
        fold_metrics.append(metrics)
        print(
            f"Fold {fold} test | macro F1: {metrics['macro_f1']:.4f} | "
            f"micro F1: {metrics['micro_f1']:.4f} | exact match: {metrics['exact_match']:.4f}"
        )

    summary = {}
    for key in ["macro_f1", "micro_f1", "weighted_f1", "samples_f1", "exact_match", "hamming_loss"]:
        values = np.array([m[key] for m in fold_metrics], dtype=np.float32)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}

    print("\nResumen CV:")
    print(
        f"  macro F1: {summary['macro_f1']['mean']:.4f} +/- {summary['macro_f1']['std']:.4f}\n"
        f"  micro F1: {summary['micro_f1']['mean']:.4f} +/- {summary['micro_f1']['std']:.4f}\n"
        f"  exact match: {summary['exact_match']['mean']:.4f} +/- {summary['exact_match']['std']:.4f}"
    )

    return {"folds": fold_metrics, "summary": summary}


# ── Carga y filtrado ──────────────────────────────────────────────────────────
loaders = {"csv": load_features_csv, "npz": load_features_npz, "jsonl": load_features_jsonl}
X, y_text, feature_names, state_text = loaders[DATA_FORMAT]()

mask = np.array([label not in IGNORE_LABELS for label in y_text])
X = X[mask]
y_text = y_text[mask]
if state_text is not None:
    state_text = state_text[mask]
print(f"Muestras después de filtrar combinadas: {X.shape[0]}")

y_encoded, label_names, label_combo = fit_label_encoder(y_text)
y_state, state_names = fit_state_encoder(state_text)

print(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} caracteristicas")
print(f"Labels unicos: {np.unique(y_text)}")
print(f"Clases: {label_names}")
for i, label in enumerate(label_names):
    print(f"  {label}: {int(y_encoded[:, i].sum())}")
if y_state is not None:
    print(f"Estados: {state_names}")
    for i, state in enumerate(state_names):
        print(f"  {state}: {int((y_state == i).sum())}")

# ── Cross-validation ──────────────────────────────────────────────────────────
cv_results = run_cross_validation(X, y_encoded, label_combo, label_names, y_state, state_names)

# ── Entrenamiento final ───────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Entrenamiento final con split train/val/test...")

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    _combo_train,
    _combo_val,
    _combo_test,
    state_train,
    state_val,
    state_test,
) = split_train_val_test(X, y_encoded, label_combo, y_state)
X_train, X_val, X_test, scaler = scaled_split(X_train, X_val, X_test)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

final_result = train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    label_names,
    max_epochs=FINAL_EPOCHS,
    seed=SEED,
    state_train=state_train,
    state_val=state_val,
    num_states=len(state_names),
)
model = final_result["model"]
val_probs = predict_probs(model, X_val)
test_probs = predict_probs(model, X_test)

global_thresholds, global_val_f1 = tune_global_threshold(val_probs, y_val, label_names)
per_class_thresholds, per_class_val_f1 = tune_per_class_thresholds(val_probs, y_val, label_names)

default_metrics = metrics_dict(y_test, test_probs, np.full(len(label_names), 0.5), label_names)
global_metrics = metrics_dict(y_test, test_probs, global_thresholds, label_names)
final_metrics = metrics_dict(y_test, test_probs, per_class_thresholds, label_names)
state_guided_metrics = (
    metrics_dict(y_test, test_probs, per_class_thresholds, label_names, state_test, state_names)
    if state_test is not None
    else None
)

print("\nComparacion en test:")
print(
    f"  Threshold 0.50      | macro F1: {default_metrics['macro_f1']:.4f} | "
    f"micro F1: {default_metrics['micro_f1']:.4f} | exact: {default_metrics['exact_match']:.4f}"
)
print(
    f"  Threshold global   | macro F1: {global_metrics['macro_f1']:.4f} | "
    f"micro F1: {global_metrics['micro_f1']:.4f} | exact: {global_metrics['exact_match']:.4f} "
    f"(val macro F1: {global_val_f1:.4f})"
)
print(
    f"  Threshold por clase| macro F1: {final_metrics['macro_f1']:.4f} | "
    f"micro F1: {final_metrics['micro_f1']:.4f} | exact: {final_metrics['exact_match']:.4f} "
    f"(val macro F1: {per_class_val_f1:.4f})"
)
if state_guided_metrics is not None:
    print(
        f"  + Estado conocido   | macro F1: {state_guided_metrics['macro_f1']:.4f} | "
        f"micro F1: {state_guided_metrics['micro_f1']:.4f} | "
        f"exact: {state_guided_metrics['exact_match']:.4f}"
    )

print("\nReporte por clase (threshold por clase):")
print(
    classification_report(
        y_test.astype(int),
        apply_thresholds(test_probs, per_class_thresholds, label_names),
        target_names=label_names,
        zero_division=0,
    )
)

# ── Guardar artefactos ────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)

metrics_payload = {
    "recommended_general_metric": "macro_f1",
    "training_strategy": "multi_task_state_auxiliary" if y_state is not None else "fault_only",
    "state_loss_weight": STATE_LOSS_WEIGHT if y_state is not None else 0.0,
    "ignored_labels": list(IGNORE_LABELS),
    "final_test": final_metrics,
    "test_with_known_state_rules": state_guided_metrics,
    "test_with_global_threshold": global_metrics,
    "test_with_threshold_0_50": default_metrics,
    "validation": {
        "best_epoch": final_result["best_epoch"],
        "best_val_macro_f1_during_training": final_result["best_val_macro_f1"],
        "best_val_loss": final_result["best_val_loss"],
        "global_threshold_macro_f1": float(global_val_f1),
        "per_class_threshold_macro_f1": float(per_class_val_f1),
    },
    "cross_validation": cv_results,
}

with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_payload, f, indent=2)

config = {
    "label_names": label_names,
    "feature_names": feature_names,
    "input_size": int(X.shape[1]),
    "num_labels": len(label_names),
    "task_type": "multi_label",
    "training_strategy": "multi_task_state_auxiliary" if y_state is not None else "fault_only",
    "state_names": state_names,
    "state_allowed_labels": STATE_ALLOWED_LABELS,
    "state_loss_weight": STATE_LOSS_WEIGHT if y_state is not None else 0.0,
    "ignored_labels": list(IGNORE_LABELS),
    "threshold": float(per_class_thresholds.mean()),
    "thresholds": per_class_thresholds.astype(float).tolist(),
    "threshold_strategy": "per_class_validation_f1",
    "feature_mean": scaler.mean_.astype(float).tolist(),
    "feature_scale": scaler.scale_.astype(float).tolist(),
    "output": "probabilities",
    "general_metric": "macro_f1",
}

with open(OUTPUT_DIR / "cardiag_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)


class InferenceModel(torch.nn.Module):
    def __init__(self, classifier, feature_mean, feature_scale):
        super().__init__()
        self.classifier = classifier
        self.register_buffer("feature_mean", torch.tensor(feature_mean, dtype=torch.float32).view(1, -1))
        self.register_buffer("feature_scale", torch.tensor(feature_scale, dtype=torch.float32).view(1, -1))

    def forward(self, features):
        features = (features - self.feature_mean) / self.feature_scale
        logits = self.classifier(features)
        return torch.sigmoid(logits)


print("\nExportando a ONNX...")
model = model.to("cpu").eval()
inference_model = InferenceModel(model, scaler.mean_, scaler.scale_).eval()
dummy_input = torch.randn(1, X.shape[1], dtype=torch.float32)
sidecar = OUTPUT_DIR / "model.onnx.data"
if sidecar.exists():
    sidecar.unlink()

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="You are using the legacy TorchScript-based ONNX export.*",
    )
    torch.onnx.export(
        inference_model,
        (dummy_input,),
        str(OUTPUT_DIR / "model.onnx"),
        input_names=["features"],
        output_names=["probabilities"],
        opset_version=18,
        dynamo=False,
        external_data=False,
        do_constant_folding=True,
        dynamic_axes={
            "features": {0: "batch"},
            "probabilities": {0: "batch"},
        },
    )

gc.collect()

print(f"\nModelo guardado en: {OUTPUT_DIR}")
print("  - model.onnx listo para produccion")
print("  - cardiag_config.json")
print("  - metrics.json")
print(f"\nInput: vector de {X.shape[1]} caracteristicas sin escalar")
print(f"Output: {len(label_names)} probabilidades ({', '.join(label_names)})")