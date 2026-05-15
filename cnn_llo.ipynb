# ── 0. Setup ──────────────────────────────────────────────────────────────────
DRIVE_PATH = '/kaggle/input/datasets/daphndelacou/mont-terri/'

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── GPU ───────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU — go to Settings → Accelerator → GPU T4 x2")

# ── 2. Load Data ──────────────────────────────────────────────────────────────
print("\nLoading data...")
print("Available files:")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(f"  {os.path.join(dirname, filename)}")

df_coord = pd.read_parquet(DRIVE_PATH + 'sensors.parquet', engine='pyarrow')
df_coord.columns = ['id', 'x', 'y', 'z']
df_coord = df_coord.drop_duplicates(subset='id', keep='first')
df_coord = df_coord[df_coord['id'] != 'N2']

df_train = pd.read_parquet(DRIVE_PATH + 'train_cleaned_2.parquet', engine='pyarrow')
df_train.columns = ['id', 'time', 'power', 'temperature', 'run_id']

df_test = pd.read_parquet(DRIVE_PATH + 'test.parquet', engine='pyarrow')
df_test.columns = ['id', 'time', 'power']

print(f"df_coord: {df_coord.shape}")
print(f"df_train: {df_train.shape}")
print(f"df_test:  {df_test.shape}")

# ── 3. Assign Run IDs to Test ─────────────────────────────────────────────────
def assign_run_ids_new(df, min_time=0.0):
    df = df.copy()
    df['run_id'] = 0
    for sensor_id in df['id'].unique():
        mask = (df['id'] == sensor_id) & (df['time'] == min_time)
        zero_indices = sorted(df[mask].index.tolist())
        for run_num, start_idx in enumerate(zero_indices, start=1):
            if run_num < len(zero_indices):
                end_idx = zero_indices[run_num]
                run_mask = (df['id'] == sensor_id) & (df.index >= start_idx) & (df.index < end_idx)
            else:
                run_mask = (df['id'] == sensor_id) & (df.index >= start_idx)
            df.loc[run_mask, 'run_id'] = run_num
    return df

df_test_run = assign_run_ids_new(df_test, min_time=864000)
print(f"\nTest run distribution: {df_test_run['run_id'].value_counts().sort_index().to_dict()}")

# ── 4. Feature Engineering ────────────────────────────────────────────────────
seconds_per_year = 365.25 * 24 * 3600

def distance_from_canister(x, y):
    nearest_x = np.clip(x, 0, 0.5)
    nearest_y = np.clip(y, 0, 2.4)
    return np.sqrt((x - nearest_x)**2 + (y - nearest_y)**2)

def get_zone(x, y):
    return 0 if x <= 1.4 else 1

def build_features(df, df_coord):
    df = df.copy()
    df = df.merge(df_coord[['id', 'x', 'y']], on='id', how='left')
    df['dist_from_canister'] = df.apply(
        lambda row: distance_from_canister(row['x'], row['y']), axis=1)
    df['dist_radial']  = np.sqrt(df['x']**2 + df['y']**2)
    df['zone']         = df.apply(lambda row: get_zone(row['x'], row['y']), axis=1)
    df['time_years']   = df['time'] / seconds_per_year
    df['log_time']     = np.log1p(df['time_years'])
    df['log_dist']     = np.log1p(df['dist_from_canister'])
    df['power_x_dist'] = df['power'] * df['dist_from_canister']
    df['power_x_time'] = df['power'] * df['log_time']
    epsilon = 1e-6
    df['inv_dist_sq']            = 1.0 / (df['dist_from_canister']**2 + epsilon)
    df['power_over_dist_sq']     = df['power'] / (df['dist_from_canister']**2 + epsilon)
    df['log_power_over_dist_sq'] = np.log1p(df['power_over_dist_sq'])
    return df

df_train_feat = build_features(df_train, df_coord)
df_test_feat  = build_features(df_test_run, df_coord)

features = [
    'x', 'dist_from_canister', 'dist_radial', 'log_dist',
    'zone', 'log_time', 'power', 'run_id',
    'power_x_dist', 'power_x_time',
    'inv_dist_sq', 'power_over_dist_sq', 'log_power_over_dist_sq'
]
target     = 'temperature'
weight_map = {0: 1.0, 1: 3.0}
df_train_feat['weight'] = df_train_feat['zone'].map(weight_map)

print(f"\nFeatures ({len(features)}): {features}")
print(f"Train shape: {df_train_feat.shape}")
print(f"Test shape:  {df_test_feat.shape}")

# ── 5. Scaling ────────────────────────────────────────────────────────────────
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_all_raw  = df_train_feat[features].values.astype(np.float32)
y_all_raw  = df_train_feat[target].values.astype(np.float32).reshape(-1, 1)
X_test_raw = df_test_feat[features].values.astype(np.float32)

X_all_scaled  = scaler_X.fit_transform(X_all_raw)
y_all_scaled  = scaler_y.fit_transform(y_all_raw)
X_test_scaled = scaler_X.transform(X_test_raw)

weights_all = df_train_feat['weight'].values.astype(np.float32)
zones_all   = df_train_feat['zone'].values
sensor_ids  = df_train_feat['id'].values

print(f"\nX_all_scaled:  {X_all_scaled.shape}")
print(f"X_test_scaled: {X_test_scaled.shape}")

# ── 6. Model ──────────────────────────────────────────────────────────────────
class ThermalNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

class WeightedMSELoss(nn.Module):
    def forward(self, pred, target, weights):
        return ((pred.squeeze() - target)**2 * weights).mean()

# ── 7. Metrics ────────────────────────────────────────────────────────────────
def weighted_rmse(y_true, y_pred, weights):
    return np.sqrt(np.average((y_true - y_pred)**2, weights=weights))

def evaluate_model(y_true, y_pred, weights, zones, label=""):
    wrmse = weighted_rmse(y_true, y_pred, weights)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for zone_id, zone_name in [(0, 'Canister+Buffer'), (1, 'OPA')]:
        mask = zones == zone_id
        if mask.sum() == 0: continue
        z_wrmse = weighted_rmse(y_true[mask], y_pred[mask], weights[mask])
        print(f"  {zone_name} WRMSE: {z_wrmse:.4f}°C ({mask.sum():,} samples)")
    print(f"  MAE:           {mae:.4f}°C")
    print(f"  RMSE:          {rmse:.4f}°C")
    print(f"  R²:            {r2:.4f}")
    print(f"  Weighted RMSE: {wrmse:.4f}°C  ← Kaggle metric")
    return wrmse

# ── 8. Build Stratified Location Folds ───────────────────────────────────────
def build_stratified_location_folds(df_train_feat, df_coord, n_folds=5):
    """
    Build LLO folds stratified by zone AND distance.
    Each fold gets sensors from all zones and distances.
    With 240 sensors and 5 folds → ~48 sensors per fold.
    """
    np.random.seed(42)

    sensor_info = df_coord[
        df_coord['id'].isin(df_train_feat['id'].unique())
    ].copy()

    sensor_info['zone'] = sensor_info.apply(
        lambda r: get_zone(r['x'], r['y']), axis=1)
    sensor_info['dist'] = sensor_info.apply(
        lambda r: distance_from_canister(r['x'], r['y']), axis=1)

    # Bin distance into 3 groups: near, medium, far
    sensor_info['dist_bin'] = pd.qcut(
        sensor_info['dist'], q=3, labels=[0, 1, 2])

    # Stratum = zone + distance bin
    sensor_info['stratum'] = (sensor_info['zone'].astype(str) + '_' +
                               sensor_info['dist_bin'].astype(str))

    print(f"\nSensors per stratum:")
    print(sensor_info['stratum'].value_counts().sort_index())

    # Assign fold within each stratum (round-robin)
    sensor_info['fold'] = -1
    for stratum in sensor_info['stratum'].unique():
        mask    = sensor_info['stratum'] == stratum
        indices = sensor_info[mask].index.tolist()
        np.random.shuffle(indices)
        for i, idx in enumerate(indices):
            sensor_info.loc[idx, 'fold'] = i % n_folds

    # Build fold lists
    folds = []
    print(f"\nFold composition ({n_folds} folds, ~48 sensors each):")
    for fold_id in range(n_folds):
        fold_mask   = sensor_info['fold'] == fold_id
        val_sensors = sensor_info[fold_mask]['id'].tolist()
        zone_counts = sensor_info[fold_mask]['zone'].value_counts()
        dist_counts = sensor_info[fold_mask]['dist_bin'].value_counts()
        folds.append(val_sensors)
        print(f"  Fold {fold_id+1}: {len(val_sensors)} sensors | "
              f"zones={zone_counts.to_dict()} | "
              f"dist_bins={dist_counts.to_dict()}")

    return folds, sensor_info

N_FOLDS = 5  # 240 sensors / 5 folds = 48 sensors per fold
sensor_folds, sensor_info = build_stratified_location_folds(
    df_train_feat, df_coord, n_folds=N_FOLDS
)

# ── 9. Training Config ────────────────────────────────────────────────────────
EPOCHS        = 30
BATCH_SIZE    = 8192
LEARNING_RATE = 1e-3
PATIENCE      = 5

print(f"\nTraining config:")
print(f"  Folds:         {N_FOLDS} (~48 sensors per fold)")
print(f"  Epochs:        {EPOCHS}")
print(f"  Batch size:    {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Early stop:    patience={PATIENCE}")

# ── 10. LLO Training Loop ─────────────────────────────────────────────────────
criterion         = WeightedMSELoss()
oof_preds_scaled  = np.zeros(len(X_all_scaled))
test_preds_scaled = np.zeros(len(X_test_scaled))
fold_scores       = []
train_histories   = []

for fold_idx, val_sensors in enumerate(sensor_folds):
    print(f"\n{'='*50}")
    print(f"  Fold {fold_idx+1}/{N_FOLDS}")
    print(f"  Holding out {len(val_sensors)} sensors")
    print(f"{'='*50}")

    # ── Split by sensor location ──────────────────────────────────────────
    train_mask = ~np.isin(sensor_ids, val_sensors)
    val_mask   =  np.isin(sensor_ids, val_sensors)

    train_pos  = np.where(train_mask)[0]
    val_pos    = np.where(val_mask)[0]

    X_fit = X_all_scaled[train_pos].astype(np.float32)
    y_fit = y_all_scaled[train_pos].astype(np.float32).flatten()
    w_fit = weights_all[train_pos]

    X_val = X_all_scaled[val_pos].astype(np.float32)
    y_val = y_all_scaled[val_pos].astype(np.float32).flatten()
    w_val = weights_all[val_pos]

    n_train_sensors = len(set(sensor_ids[train_pos]))
    n_val_sensors   = len(set(sensor_ids[val_pos]))
    print(f"  Train: {len(X_fit):,} rows ({n_train_sensors} sensors)")
    print(f"  Val:   {len(X_val):,} rows ({n_val_sensors} sensors)")

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_fit),
            torch.from_numpy(y_fit),
            torch.from_numpy(w_fit)
        ),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
            torch.from_numpy(w_val)
        ),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model     = ThermalNet(len(features)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5, min_lr=1e-5)

    best_val_loss    = float('inf')
    best_model_state = None
    train_losses     = []
    val_losses       = []
    patience_count   = 0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y, batch_w in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_X).squeeze()
            loss = criterion(pred, batch_y, batch_w)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y, batch_w in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_w = batch_w.to(device)
                pred     = model(batch_X).squeeze()
                val_loss += criterion(pred, batch_y, batch_w).item()

        avg_train  = train_loss / len(train_loader)
        avg_val    = val_loss   / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(avg_val)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            best_model_state = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train: {avg_train:.6f} | "
                  f"Val: {avg_val:.6f} | "
                  f"LR: {current_lr:.6f}")

    train_histories.append({'train': train_losses, 'val': val_losses})

    # ── Load best model ───────────────────────────────────────────────────
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    # ── OOF predictions ───────────────────────────────────────────────────
    oof_preds_batch = []
    with torch.no_grad():
        for i in range(0, len(X_val), BATCH_SIZE):
            batch = torch.from_numpy(X_val[i:i+BATCH_SIZE]).to(device)
            oof_preds_batch.append(model(batch).cpu().numpy())
    oof_fold = np.vstack(oof_preds_batch).flatten()
    oof_preds_scaled[val_pos] = oof_fold

    # ── Test predictions ──────────────────────────────────────────────────
    test_preds_batch = []
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(
            X_test_scaled.astype(np.float32))),
        batch_size=BATCH_SIZE, shuffle=False
    )
    with torch.no_grad():
        for batch in test_loader:
            test_preds_batch.append(
                model(batch[0].to(device)).cpu().numpy()
            )
    test_preds_scaled += np.vstack(test_preds_batch).flatten() / N_FOLDS

    # ── Fold score ────────────────────────────────────────────────────────
    oof_fold_orig = scaler_y.inverse_transform(
        oof_fold.reshape(-1, 1)).flatten()
    y_val_orig    = scaler_y.inverse_transform(
        y_val.reshape(-1, 1)).flatten()
    fold_wrmse    = weighted_rmse(y_val_orig, oof_fold_orig, w_val)
    fold_scores.append(fold_wrmse)
    print(f"\n  Fold {fold_idx+1} WRMSE: {fold_wrmse:.4f}°C")

    # ── Save checkpoint ───────────────────────────────────────────────────
    os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
    torch.save(best_model_state,
               f'/kaggle/working/checkpoints/llo_fold{fold_idx+1}.pt')
    np.save('/kaggle/working/checkpoints/llo_oof.npy',  oof_preds_scaled)
    np.save('/kaggle/working/checkpoints/llo_test.npy', test_preds_scaled)
    print(f"  Checkpoint saved ✅")

    del model
    gc.collect()
    torch.cuda.empty_cache()

# ── 11. OOF Evaluation ────────────────────────────────────────────────────────
oof_preds_orig = scaler_y.inverse_transform(
    oof_preds_scaled.reshape(-1, 1)).flatten()
y_all_orig     = scaler_y.inverse_transform(
    y_all_scaled).flatten()

llo_wrmse = evaluate_model(
    y_all_orig, oof_preds_orig,
    weights_all, zones_all,
    "Leave-Location-Out Results"
)

print(f"\n  Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
print(f"  Mean:        {np.mean(fold_scores):.4f}°C")
print(f"  Std:         {np.std(fold_scores):.4f}°C")
print(f"  LLO WRMSE:   {llo_wrmse:.4f}°C")

print(f"\n{'='*50}")
print(f"  CV Strategy Comparison")
print(f"{'='*50}")
print(f"  Time-split k-fold CV:  ~1.29°C  ← optimistic")
print(f"  Leave-Location-Out CV:  {llo_wrmse:.4f}°C  ← realistic")
if llo_wrmse > 1.8:
    print(f"\n  ⚠️  Large gap → model struggles with new locations")
    print(f"  → Consider GCN or adding more spatial features")
elif llo_wrmse < 1.4:
    print(f"\n  ✅ Small gap → model generalizes well to new locations")
    print(f"  → Focus on ensembling for further improvement")
else:
    print(f"\n  ⚠️  Moderate gap → some location generalization issue")
    print(f"  → Physics features help, consider PINN ensemble")

# ── 12. Plot Training History ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, N_FOLDS, figsize=(18, 4))
for i in range(N_FOLDS):
    axes[i].plot(train_histories[i]['train'], label='Train', color='blue')
    axes[i].plot(train_histories[i]['val'],   label='Val',   color='orange')
    axes[i].set_title(f'Fold {i+1}\n({len(sensor_folds[i])} sensors)',
                      fontsize=9)
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel('Loss')
    axes[i].legend(fontsize=7)
    axes[i].grid(True, alpha=0.3)
plt.suptitle('Leave-Location-Out Training History', fontsize=13)
plt.tight_layout()
plt.savefig('/kaggle/working/llo_training_history.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ── 13. Per-sensor score map ──────────────────────────────────────────────────
sensor_scores = []
for sensor_id in df_train_feat['id'].unique():
    mask = sensor_ids == sensor_id
    if mask.sum() == 0: continue
    y_true = y_all_orig[mask]
    y_pred = oof_preds_orig[mask]
    w      = weights_all[mask]
    wrmse  = weighted_rmse(y_true, y_pred, w)
    coord  = df_coord[df_coord['id'] == sensor_id]
    if len(coord) == 0: continue
    coord  = coord.iloc[0]
    sensor_scores.append({
        'sensor': sensor_id,
        'wrmse':  wrmse,
        'x':      coord['x'],
        'y':      coord['y'],
        'zone':   get_zone(coord['x'], coord['y'])
    })

sensor_score_df = pd.DataFrame(sensor_scores).sort_values(
    'wrmse', ascending=False)

print("\nWorst predicted sensors:")
print(sensor_score_df.head(10)[['sensor', 'x', 'y', 'zone', 'wrmse']].to_string())
print("\nBest predicted sensors:")
print(sensor_score_df.tail(10)[['sensor', 'x', 'y', 'zone', 'wrmse']].to_string())

plt.figure(figsize=(22, 5))
scatter = plt.scatter(
    sensor_score_df['x'],
    sensor_score_df['y'],
    c=sensor_score_df['wrmse'],
    cmap='RdYlGn_r',
    s=120, edgecolors='black', linewidths=0.5
)
plt.colorbar(scatter, label='WRMSE (°C)')
for _, row in sensor_score_df.iterrows():
    plt.annotate(row['sensor'], (row['x'], row['y']),
                 fontsize=5, ha='left', va='bottom',
                 xytext=(2, 2), textcoords='offset points')
plt.axvline(x=1.4, color='teal', linewidth=1.5,
            linestyle='--', label='Buffer/OPA boundary')
plt.xlabel('x (m)', fontsize=12)
plt.ylabel('y (m)', fontsize=12)
plt.title('Per-sensor LLO Validation WRMSE\n'
          'Red = model struggles | Green = model predicts well',
          fontsize=13)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('/kaggle/working/llo_per_sensor_map.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ── 14. Generate Final Predictions ───────────────────────────────────────────
test_preds_orig = scaler_y.inverse_transform(
    test_preds_scaled.reshape(-1, 1)
).flatten()

print(f"\nPredicted temperature range: "
      f"{test_preds_orig.min():.2f} - {test_preds_orig.max():.2f}°C")

# ── 15. Save Submission ───────────────────────────────────────────────────────
submission = pd.DataFrame({
    "Id":          np.arange(len(df_test_feat), dtype=int),
    "temperature": test_preds_orig
})

assert list(submission.columns) == ["Id", "temperature"]
assert len(submission) == 2190480, f"Wrong rows: {len(submission)}"
assert np.isfinite(submission["temperature"]).all()
assert submission.isna().sum().sum() == 0

submission.to_csv('/kaggle/working/submission_llo.csv', index=False)

print(f"\n✅ Done!")
print(f"Shape: {submission.shape}")
print(f"Temperature range: "
      f"{test_preds_orig.min():.2f} - {test_preds_orig.max():.2f}°C")
print(f"\nFiles saved to /kaggle/working/:")
print(f"  submission_llo.csv")
print(f"  llo_training_history.png")
print(f"  llo_per_sensor_map.png")
print(f"  checkpoints/llo_fold1-{N_FOLDS}.pt")
print(f"\nDownload from the Output tab!")
print(submission.head(10))
8   8    17.100200
9   9    17.096550
