import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from tqdm import tqdm

# =========================
# Config
# =========================
DATA_PATH = "_Liver_16388_39_features_4_label_20240910.csv"

# 9 個時間閾值（向量欄順序）
T_LIST = [2.0, 5.0, 9.9, 14.9, 22.9, 34.5, 51.0, 63.5, 84.1]

# OOF：100 次 80/20
N_SEEDS = 100
TEST_SIZE = 0.2

# XGBoost 分類器（建 9T 用）
XGB_CLS_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    device="cuda",          # 無 GPU 可移除此鍵或改 "cpu"
    n_estimators=100,
    use_label_encoder=False,
    verbosity=0
)

# XGBoost 回歸器（鏈式回填；訓練只用「一開始就完整」）
XGB_REG_PARAMS = dict(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    device="cuda",
    n_estimators=300,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=0
)

# Base 特徵：BCLC + 9 種治療
BASE_FEATS = [
    "BCLC_stage",
    "Liver_transplantation", "Surgical_resection", "Radiofrequency", "TACE",
    "Target_therapy", "Immunotherapy", "HAIC", "Radiotherapy", "Best_support_care"
]

# =========================
# Step 0: 載入與基本清理
# =========================
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Hospital", "Label"], errors="ignore")

ALL_FEATS = [c for c in df.columns if c not in ["time", "event"]]
print(f"Total samples after basic filter: {len(df)}")
print(f"Num available features: {len(ALL_FEATS)}")

# =========================
# 公用函式：建 9T + coverage（缺者為 NaN）
# =========================
def build_and_fill_9T_seedwise_avg(df_all: pd.DataFrame,
                                   sample_feature_cols: list,
                                   t_list=None,
                                   n_seeds: int = 100,
                                   test_size: float = 0.2):
    """
    每個 seed：
      - 9 個分類器：只把「驗證集」機率寫入 outputs_val_only（train 不寫）
      - 另建 outputs_trainval：train+val 機率皆寫入，僅用於回歸器訓練完整樣本
      - 用 outputs_trainval 決定完整樣本並訓練 9 個回歸器；在 outputs_val_only 的 NaN 處做鏈式回填 → W_complete
      - 統計：
          val_sum/val_cnt：只加驗證集原生機率
          fill_sum/fill_cnt：只加回歸器填補後的機率（僅限原本是 NaN 的 cell）
    最後：
      - 若某 cell val_cnt>0：final = val_sum/val_cnt（真正 OOF）
      - 否則 final = fill_sum/fill_cnt（全部都是補值）
    """
    if t_list is None:
        t_list = T_LIST
    cols_9T = [f"T={t}" for t in t_list]
    n = len(df_all)

    # 全局累積器
    val_sum  = pd.DataFrame(0.0, index=df_all.index, columns=cols_9T)
    val_cnt  = pd.DataFrame(0,   index=df_all.index, columns=cols_9T, dtype=int)
    fill_sum = pd.DataFrame(0.0, index=df_all.index, columns=cols_9T)
    fill_cnt = pd.DataFrame(0,   index=df_all.index, columns=cols_9T, dtype=int)

    seeds_done = 0

    for seed in tqdm(range(n_seeds), desc="Seeds"):
        # 這份只放「驗證集」的機率，用來後續平均（train 一律 NaN）
        outputs_val_only   = pd.DataFrame(np.nan, index=df_all.index, columns=cols_9T, dtype=float)
        # 這份放「train+val」的機率，用於找完整樣本訓練回歸器
        outputs_trainval   = pd.DataFrame(np.nan, index=df_all.index, columns=cols_9T, dtype=float)

        # === 9 個二分類器 ===
        for j, T in enumerate(t_list):
            col = cols_9T[j]

            # 排除 (time <= T & event==0)
            mask_keep = ~((df_all["time"] <= T) & (df_all["event"] == 0))
            valid = df_all[mask_keep].copy()

            # label
            valid["label"] = (valid["time"] >= T).astype(int)
            X_all = valid[sample_feature_cols]
            y_all = valid["label"].astype(int)
            idx_all = valid.index

            X_tr, X_va, y_tr, y_va, idx_tr, idx_va = train_test_split(
                X_all, y_all, idx_all, test_size=test_size, random_state=seed, stratify=y_all
            )

            clf = xgb.XGBClassifier(**XGB_CLS_PARAMS)
            clf.fit(X_tr, y_tr)

            prob_tr = clf.predict_proba(X_tr)[:, 1]
            prob_va = clf.predict_proba(X_va)[:, 1]

            # 只把驗證集寫入「最終平均用」矩陣；train 保持 NaN
            outputs_val_only.loc[idx_va, col] = prob_va

            # 訓練回歸器要用到 train+val，盡量擴大完整樣本
            outputs_trainval.loc[idx_tr, col] = prob_tr
            outputs_trainval.loc[idx_va, col] = prob_va

        # === 回歸器：用 train+val 決定完整樣本；在 val-only 的 NaN 處做回填 ===
        cols = cols_9T
        # 初始完整樣本（本 seed 中 9 欄都有值）——用 train+val 矩陣來判斷
        full_mask = outputs_trainval[cols].notna().all(axis=1)
        full_idx = outputs_trainval.index[full_mask]
     
        train_vec = outputs_trainval.loc[full_idx, cols].copy()  # 真值（本 seed 的 prob）
        train_feats_mat = df_all.loc[full_idx, sample_feature_cols].copy()

        # 工作矩陣從「val-only」開始：驗證集保持其原值，其餘 NaN 等待回填
        W = outputs_val_only.copy()

        for i in range(9):
            tgt_col = cols[i]
            prev_cols = cols[:i]

            # 訓練資料（完整樣本）：樣本特徵 + 前 i 欄真值（train+val）
            if i == 0:
                X_train = train_feats_mat.values
            else:
                X_train = pd.concat([train_feats_mat, train_vec[prev_cols]], axis=1).values
            y_train = train_vec[tgt_col].values

            reg = xgb.XGBRegressor(**XGB_REG_PARAMS)
            reg.fit(X_train, y_train)

            # 需要回填（本 seed 中此欄在 val-only 仍是 NaN）
            need_fill_mask = W[tgt_col].isna()
            if need_fill_mask.sum() == 0:
                continue

            if i == 0:
                X_pred = df_all.loc[need_fill_mask, sample_feature_cols].values
            else:
                F_pred = df_all.loc[need_fill_mask, sample_feature_cols]
                V_pred = W.loc[need_fill_mask, prev_cols]
                if V_pred.isna().any().any():
                    V_pred = V_pred.fillna(0.0)  # 依序回填，理論上不會
                X_pred = pd.concat([F_pred, V_pred], axis=1).values

            y_hat = reg.predict(X_pred)
            W.loc[need_fill_mask, tgt_col] = y_hat

        # === 累積：分開累計驗證值與回填值 ===
        # 驗證值：在 val-only 中非 NaN 的 cell
        val_mask = outputs_val_only.notna()
        val_sum[val_mask] += outputs_val_only[val_mask]
        val_cnt[val_mask] += 1

        # 回填值：原本是 NaN（在 val-only），但在 W 中已被填上
        fill_mask = outputs_val_only.isna() & W.notna()
        fill_sum[fill_mask] += W[fill_mask]
        fill_cnt[fill_mask] += 1

        seeds_done += 1

    if seeds_done == 0:
        raise RuntimeError("所有 seed 都沒有可用的完整樣本，回填流程未能執行。")

    # === 匯總：優先用驗證平均，其次用回填平均 ===
    final = pd.DataFrame(index=df_all.index, columns=cols_9T, dtype=float)

    # 有驗證者：用驗證平均
    has_val = (val_cnt.values > 0)
    final.values[has_val] = (val_sum.values[has_val] / np.maximum(val_cnt.values[has_val], 1))

    # 無驗證者：用回填平均
    no_val = ~has_val
    # 避免除以 0：理論上 fill_cnt 應該 >0，若仍為 0 就設為 0（或 np.nan）
    denom = np.maximum(fill_cnt.values[no_val], 1)
    fill_avg = np.zeros_like(denom, dtype=float)
    fill_avg = fill_sum.values[no_val] / denom
    final.values[no_val] = fill_avg

    return final, seeds_done