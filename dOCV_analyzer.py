#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
저전압 불량 선별 로직 고도화 분석기
dOCV 로직 비교: Baseline / C(DPAT) / A(온도회귀) / B(NNR) / D(K기울기) / E(ML)
"""

import sys
import os
import warnings
import traceback
import numpy as np
import pandas as pd


def _mad_normal(x):
    """scipy 없이 MAD (정규분포 환산, ×1.4826) 계산"""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return np.median(np.abs(x - med)) * 1.4826

warnings.filterwarnings("ignore")

# ── 옵션 의존성 ──────────────────────────────────────────────
try:
    from sklearn.linear_model import HuberRegressor
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve, auc
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MPLOT_OK = True
except Exception:
    MPLOT_OK = False

try:
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
    TK_OK = True
except ImportError:
    TK_OK = False

# ── 기본 파라미터 ────────────────────────────────────────────
K_SIGMA      = 3.5   # MAD 배수 (동적 임계값)
FLOOR_MV     = 0.5   # 바닥값 (mV)
FIXED_OFFSET = 0.8   # 현재 고정 오프셋 (mV)
T_REF        = 25.0  # 기준온도 (°C)
DT1_DAY      = 1.0   # OCV1→OCV2 측정 간격 (일)
DT2_DAY      = 2.0   # OCV2→OCV3 측정 간격 (일)

# ════════════════════════════════════════════════════════════
# 1. 파일 로드 & 컬럼 자동 감지
# ════════════════════════════════════════════════════════════

def load_excel(path: str) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        if len(df) > 5:
            print(f"  시트 '{sheet}' 로드: {len(df)} 행, {len(df.columns)} 열")
            return df
    return pd.read_excel(path, sheet_name=0)


def auto_map(df: pd.DataFrame) -> dict:
    """컬럼명 패턴 자동 매핑"""
    uc = {c.upper().replace(" ", "_").replace("(", "").replace(")", ""): c
          for c in df.columns}
    m = {}

    groups = {
        "ocv1": ["OCV1", "OCV_1", "V1", "VOLT1", "VOLTAGE1", "전압1"],
        "ocv2": ["OCV2", "OCV_2", "V2", "VOLT2", "VOLTAGE2", "전압2"],
        "ocv3": ["OCV3", "OCV_3", "V3", "VOLT3", "VOLTAGE3", "전압3"],
        "t1":   ["T1", "TEMP1", "TEMPERATURE1", "TMP1", "온도1"],
        "t2":   ["T2", "TEMP2", "TEMPERATURE2", "TMP2", "온도2"],
        "t3":   ["T3", "TEMP3", "TEMPERATURE3", "TMP3", "온도3"],
        "row":  ["ROW", "TRAY_ROW", "행", "Y"],
        "col":  ["COL", "COLUMN", "TRAY_COL", "열", "X"],
        "label":["LABEL", "DEFECT", "FAULT", "NG", "불량", "결과", "RESULT"],
    }
    for key, patterns in groups.items():
        for p in patterns:
            if p in uc:
                m[key] = uc[p]
                break

    return m


# ════════════════════════════════════════════════════════════
# 2. 특징량 계산
# ════════════════════════════════════════════════════════════

def compute_features(df: pd.DataFrame, m: dict,
                     dt1: float = DT1_DAY,
                     dt2: float = DT2_DAY) -> pd.DataFrame:
    out = df.copy()

    o1 = df[m["ocv1"]].astype(float).values
    o2 = df[m["ocv2"]].astype(float).values
    o3 = df[m["ocv3"]].astype(float).values

    # ── 기본 ΔOCV ──────────────────────────────────────────
    out["dOCV_raw"]  = o1 - o3        # 현재 로직 기준
    out["dOCV_12"]   = o1 - o2
    out["dOCV_23"]   = o2 - o3
    out["OCV_mean"]  = (o1 + o2 + o3) / 3

    # ── 3점 직선 적합 → K값(mV/day) ─────────────────────────
    t_pts = np.array([0.0, dt1, dt1 + dt2])
    K_list, nonlin_list = [], []
    for i in range(len(out)):
        y = np.array([o1[i], o2[i], o3[i]])
        coeff = np.polyfit(t_pts, y, 1)
        K_list.append(-coeff[0])                         # 양수 = 강하
        y_fit = np.polyval(coeff, t_pts)
        nonlin_list.append(float(np.max(np.abs(y - y_fit))))

    out["K_slope"]       = K_list
    out["nonlin_resid"]  = nonlin_list
    out["consistent"]    = (out["dOCV_12"] > 0) & (out["dOCV_23"] > 0)

    # ── 온도 ────────────────────────────────────────────────
    if all(k in m for k in ["t1", "t2", "t3"]):
        T1 = df[m["t1"]].astype(float).values
        T2 = df[m["t2"]].astype(float).values
        T3 = df[m["t3"]].astype(float).values
        out["T1"]     = T1
        out["T2"]     = T2
        out["T3"]     = T3
        out["T_avg"]  = (T1 + T2 + T3) / 3
        out["T_delta"]= T1 - T3
    elif "t1" in m:
        T1 = df[m["t1"]].astype(float).values
        out["T1"] = out["T_avg"] = T1
        out["T_delta"] = 0.0

    return out


# ════════════════════════════════════════════════════════════
# 3. 판정 로직들
# ════════════════════════════════════════════════════════════

# ── Baseline: 현재 로직 ──────────────────────────────────────
def method_baseline(feat: pd.DataFrame,
                    offset: float = FIXED_OFFSET) -> pd.Series:
    """mode + 고정 오프셋 (현재 운영 로직)"""
    d = feat["dOCV_raw"]
    rounded = (d * 1000).round() / 1000
    mode_v = rounded.mode().iloc[0]
    return d > (mode_v + offset)


# ── Method-C: 로버스트 DPAT ──────────────────────────────────
def method_dpat(feat: pd.DataFrame,
                k: float = K_SIGMA,
                floor: float = FLOOR_MV,
                col: str = "dOCV_raw") -> pd.Series:
    """median + k·1.4826·MAD (DPAT)"""
    d = feat[col]
    med = float(np.median(d))
    mad = _mad_normal(d)
    thr = max(med + k * mad, med + floor)
    return d > thr


# ── Method-A: 온도 회귀 보정 ─────────────────────────────────
def method_temp_regression(feat: pd.DataFrame,
                           k: float = K_SIGMA,
                           floor: float = FLOOR_MV) -> tuple:
    """
    ΔOCV ~ T 로버스트 회귀 → 잔차에 DPAT
    반환: (flags, residuals, expected)
    """
    if "T_avg" not in feat.columns:
        flags = method_dpat(feat, k, floor)
        return flags, feat["dOCV_raw"].copy(), pd.Series(0.0, index=feat.index)

    d = feat["dOCV_raw"].values
    T = feat["T_avg"].values

    if SKLEARN_OK:
        reg = HuberRegressor(epsilon=1.5, max_iter=300)
        reg.fit(T.reshape(-1, 1), d)
        d_exp = reg.predict(T.reshape(-1, 1))
    else:
        # numpy 기반 반복적 이상치 제거 회귀 (IRLS 근사)
        d_exp = _robust_linreg_numpy(T, d)

    resid = pd.Series(d - d_exp, index=feat.index)
    expected = pd.Series(d_exp, index=feat.index)

    med = float(np.median(resid))
    mad = _mad_normal(resid)
    thr = max(med + k * mad, med + floor)
    flags = resid > thr

    return flags, resid, expected


def _robust_linreg_numpy(T: np.ndarray, y: np.ndarray,
                         n_iter: int = 10) -> np.ndarray:
    """반복적 이상치 제거 선형 회귀 (numpy 전용 폴백)"""
    mask = np.ones(len(T), dtype=bool)
    coeff = np.polyfit(T[mask], y[mask], 1)
    for _ in range(n_iter):
        y_fit = np.polyval(coeff, T)
        resid = np.abs(y - y_fit)
        mad = _mad_normal(resid[mask])
        if mad < 1e-12:
            break
        mask = resid <= 2.5 * mad
        if mask.sum() < 4:
            break
        coeff = np.polyfit(T[mask], y[mask], 1)
    return np.polyval(coeff, T)


# ── Method-B: NNR 공간 이웃 잔차 ────────────────────────────
def method_nnr(feat: pd.DataFrame, m: dict,
               kernel: int = 3,
               k: float = K_SIGMA,
               floor: float = FLOOR_MV) -> tuple:
    """
    각 셀 = 주변 kernel×kernel 이웃 중앙값과 비교
    반환: (flags, nnr_residuals)
    """
    if "row" not in m or "col" not in m:
        flags = method_dpat(feat, k, floor)
        return flags, feat["dOCV_raw"].copy()

    rows  = feat[m["row"]].astype(int).values
    cols  = feat[m["col"]].astype(int).values
    d_arr = feat["dOCV_raw"].values
    half  = kernel // 2

    r_min, c_min = rows.min(), cols.min()
    nr = rows.max() - r_min + 1 + 2 * kernel
    nc = cols.max() - c_min + 1 + 2 * kernel
    grid = np.full((nr, nc), np.nan)

    for i in range(len(feat)):
        ri = rows[i] - r_min + kernel
        ci = cols[i] - c_min + kernel
        grid[ri, ci] = d_arr[i]

    nnr = np.full(len(feat), np.nan)
    for i in range(len(feat)):
        ri = rows[i] - r_min + kernel
        ci = cols[i] - c_min + kernel
        patch = grid[ri - half: ri + half + 1,
                     ci - half: ci + half + 1].flatten()
        nb = patch[~np.isnan(patch)]
        # 자신 제외
        self_val = d_arr[i]
        nb_no_self = nb[nb != self_val] if len(nb) > 1 else nb
        if len(nb_no_self) > 0:
            nnr[i] = self_val - np.median(nb_no_self)
        else:
            nnr[i] = 0.0

    resid = pd.Series(nnr, index=feat.index).fillna(0.0)
    med = float(np.median(resid))
    mad = _mad_normal(resid)
    thr = max(med + k * mad, med + floor)
    flags = resid > thr

    return flags, resid


# ── Method-D: 3점 K기울기 판정 ──────────────────────────────
def method_kslope(feat: pd.DataFrame,
                  k: float = K_SIGMA,
                  floor: float = FLOOR_MV,
                  require_consistent: bool = False) -> pd.Series:
    """3점 기울기(K값) DPAT + 일관성 체크"""
    flags = method_dpat(feat, k, floor, col="K_slope")
    if require_consistent:
        flags = flags & feat["consistent"]
    return flags


# ── Method-A+B 결합 ──────────────────────────────────────────
def method_ab_combined(feat: pd.DataFrame, m: dict,
                       k: float = K_SIGMA,
                       floor: float = FLOOR_MV) -> pd.Series:
    """A와 B 중 하나라도 불량이면 불량 (OR)"""
    flag_a, _, _ = method_temp_regression(feat, k, floor)
    flag_b, _    = method_nnr(feat, m, k=k, floor=floor)
    return flag_a | flag_b


# ── Method-E: 다변량 이상탐지 (numpy 전용 구현) ─────────────
def method_ml(feat: pd.DataFrame,
              contamination: float = 0.05) -> tuple:
    """
    로버스트 다변량 이상탐지 (numpy 전용)
    각 특징의 로버스트 Z점수(median/MAD 기반)를 Lp-norm으로 결합
    sklearn 있으면 Isolation Forest + LOF 앙상블 사용
    반환: (flags, anomaly_score)
    """
    fcols = ["dOCV_raw", "K_slope", "nonlin_resid", "dOCV_12", "dOCV_23"]
    if "T_avg"   in feat.columns: fcols.append("T_avg")
    if "T_delta" in feat.columns: fcols.append("T_delta")

    X = feat[fcols].fillna(0).values.astype(float)

    if SKLEARN_OK:
        # sklearn 경로: Isolation Forest + LOF
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        iso = IsolationForest(contamination=contamination,
                              n_estimators=200, random_state=42)
        iso.fit(Xs)
        score_iso = -iso.score_samples(Xs)

        lof = LocalOutlierFactor(n_neighbors=min(20, len(Xs) - 1),
                                 contamination=contamination)
        lof.fit_predict(Xs)
        score_lof = -lof.negative_outlier_factor_

        def norm01(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-12)

        score = (norm01(score_iso) + norm01(score_lof)) / 2

    else:
        # numpy 전용: 로버스트 Mahalanobis 유사 점수
        # 각 특징을 median/MAD로 표준화 → 가중 L1-norm 합산
        med = np.median(X, axis=0)
        mad = np.array([_mad_normal(X[:, j]) for j in range(X.shape[1])])
        mad = np.where(mad < 1e-12, 1.0, mad)
        Xz = np.abs(X - med) / mad

        # dOCV_raw와 K_slope에 가중치 2 부여 (핵심 특징)
        weights = np.ones(len(fcols))
        for i, c in enumerate(fcols):
            if c in ("dOCV_raw", "K_slope"):
                weights[i] = 2.0

        score = (Xz * weights).sum(axis=1)
        score = (score - score.min()) / (score.max() - score.min() + 1e-12)

    thr    = np.percentile(score, 100 * (1 - contamination))
    flags  = pd.Series(score >= thr, index=feat.index)
    scores = pd.Series(score, index=feat.index)

    return flags, scores


# ════════════════════════════════════════════════════════════
# 4. 전체 실행 & 성능 비교
# ════════════════════════════════════════════════════════════

def run_all(feat: pd.DataFrame, m: dict,
            k: float = K_SIGMA,
            floor: float = FLOOR_MV,
            labels: pd.Series = None) -> pd.DataFrame:

    res   = pd.DataFrame(index=feat.index)
    extra = {}   # 잔차/스코어 저장

    print("\n" + "="*55)
    print(" 방법별 불량 판정 실행")
    print("="*55)

    # Baseline
    res["Baseline"]       = method_baseline(feat).astype(int)

    # Method-C
    res["C_DPAT"]         = method_dpat(feat, k, floor).astype(int)

    # Method-A
    fa, ra, ea            = method_temp_regression(feat, k, floor)
    res["A_TempReg"]      = fa.astype(int)
    extra["A_residual"]   = ra
    extra["A_expected"]   = ea

    # Method-D
    res["D_Kslope"]       = method_kslope(feat, k, floor).astype(int)
    res["D_Kslope_strict"]= method_kslope(feat, k, floor,
                                          require_consistent=True).astype(int)

    # Method-B (위치 있을 때만)
    if "row" in m and "col" in m:
        fb, rb            = method_nnr(feat, m, k=k, floor=floor)
        res["B_NNR"]      = fb.astype(int)
        extra["B_residual"] = rb
        res["AB_Combined"]= method_ab_combined(feat, m, k, floor).astype(int)

    # Method-E
    fe, se                = method_ml(feat)
    res["E_ML"]           = fe.astype(int)
    extra["E_score"]      = se

    # ── 출력 ────────────────────────────────────────────────
    print(f"\n{'방법':<25} {'불량수':>6} {'불량율':>7}")
    print("-"*40)
    for col in res.columns:
        n = int(res[col].sum())
        pct = n / len(res) * 100
        print(f"  {col:<23} {n:>6}  {pct:>6.2f}%")

    if labels is not None:
        _eval_all(res, labels)

    return res, extra


def _eval_all(res: pd.DataFrame, labels: pd.Series):
    y = labels.astype(int).values
    print(f"\n{'방법':<25} {'감도':>7} {'특이도':>7} {'미검':>5} {'과검':>5}")
    print("-"*52)
    for col in res.columns:
        yp = res[col].astype(int).values
        tp = int(((yp==1)&(y==1)).sum())
        fp = int(((yp==1)&(y==0)).sum())
        fn = int(((yp==0)&(y==1)).sum())
        tn = int(((yp==0)&(y==0)).sum())
        sens = tp / (tp+fn+1e-9)
        spec = tn / (tn+fp+1e-9)
        print(f"  {col:<23} {sens:>6.3f}  {spec:>6.3f}  {fn:>4}  {fp:>4}")


# ════════════════════════════════════════════════════════════
# 5. 시각화
# ════════════════════════════════════════════════════════════

def plot_dashboard(feat: pd.DataFrame, res: pd.DataFrame,
                   extra: dict, m: dict, save_dir: str = None):
    if not MPLOT_OK:
        print("[경고] matplotlib 없음 → 시각화 생략")
        return

    plt.rcParams["font.family"]        = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle("dOCV 선별 로직 고도화 – 방법 비교 대시보드",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    d     = feat["dOCV_raw"]
    K     = feat["K_slope"]
    n_mth = len(res.columns)

    # ── (0,0) ΔOCV 분포 ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(d, bins=60, color="steelblue", alpha=0.75, edgecolor="white")
    med_d = float(np.median(d))
    mad_d = _mad_normal(d)
    mode_d = float((d * 1000).round().mode().iloc[0] / 1000)
    ax.axvline(med_d, color="orange", lw=2, label=f"Median {med_d:.3f}")
    ax.axvline(mode_d, color="red", lw=2, ls="--",
               label=f"Mode {mode_d:.3f}")
    ax.axvline(med_d + K_SIGMA * mad_d, color="green", lw=1.5, ls=":",
               label=f"DPAT({K_SIGMA}σ)")
    ax.axvline(mode_d + FIXED_OFFSET, color="purple", lw=1.5, ls="-.",
               label=f"Baseline (+{FIXED_OFFSET})")
    ax.set_title("ΔOCV 분포 & 임계값 비교")
    ax.set_xlabel("ΔOCV (mV)")
    ax.legend(fontsize=7)

    # ── (0,1) 온도 vs ΔOCV ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    if "T_avg" in feat.columns:
        T = feat["T_avg"]
        base_flag = res["Baseline"].astype(bool)
        ax.scatter(T[~base_flag], d[~base_flag],
                   c="steelblue", s=8, alpha=0.4, label="정상")
        ax.scatter(T[base_flag], d[base_flag],
                   c="red", s=20, alpha=0.8, label="불량(Baseline)")
        if "A_expected" in extra:
            order = T.argsort()
            ax.plot(T.values[order], extra["A_expected"].values[order],
                    "orange", lw=2, label="로버스트 회귀")
        ax.set_xlabel("T_avg (°C)")
        ax.set_ylabel("ΔOCV (mV)")
        ax.set_title("온도 vs ΔOCV & 회귀선")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "온도 데이터 없음",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title("온도 vs ΔOCV")

    # ── (0,2) K값 분포 ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(K, bins=60, color="mediumpurple", alpha=0.75, edgecolor="white")
    med_K = float(np.median(K))
    mad_K = _mad_normal(K)
    thr_K = med_K + K_SIGMA * mad_K
    ax.axvline(med_K, color="orange", lw=2, label=f"Median {med_K:.4f}")
    ax.axvline(thr_K, color="red", lw=2, ls="--",
               label=f"임계값({K_SIGMA}σ) {thr_K:.4f}")
    ax.set_title("K값(3점 기울기) 분포")
    ax.set_xlabel("K (mV/day)")
    ax.legend(fontsize=7)

    # ── (1,0~1) 방법별 불량 수 막대 ─────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    names  = list(res.columns)
    counts = [int(res[c].sum()) for c in names]
    bars   = ax.bar(names, counts,
                    color=["#e74c3c", "#3498db", "#2ecc71",
                           "#9b59b6", "#f39c12", "#1abc9c",
                           "#e67e22", "#34495e"][:len(names)],
                    edgecolor="white")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                str(cnt), ha="center", va="bottom", fontsize=9)
    ax.set_title("방법별 불량 판정 셀 수")
    ax.set_ylabel("불량 판정 수")
    ax.tick_params(axis="x", rotation=35)

    # ── (1,2) 방법 간 Jaccard 동의율 히트맵 ─────────────────
    ax = fig.add_subplot(gs[1, 2])
    J = np.zeros((n_mth, n_mth))
    for i, mi in enumerate(names):
        for j, mj in enumerate(names):
            inter = (res[mi].astype(bool) & res[mj].astype(bool)).sum()
            union = (res[mi].astype(bool) | res[mj].astype(bool)).sum()
            J[i, j] = inter / (union + 1e-9)
    im = ax.imshow(J, cmap="RdYlGn", vmin=0, vmax=1)
    short = [n.replace("Method_", "")[:8] for n in names]
    ax.set_xticks(range(n_mth)); ax.set_xticklabels(short, rotation=45, fontsize=7)
    ax.set_yticks(range(n_mth)); ax.set_yticklabels(short, fontsize=7)
    ax.set_title("방법 간 Jaccard 동의율")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # ── (2, 0~2) 앙상블 맵 or ΔOCV-K 산점도 ────────────────
    ax = fig.add_subplot(gs[2, :])
    vote = res.sum(axis=1)

    if "row" in m and "col" in m:
        rows_v = feat[m["row"]].astype(int).values
        cols_v = feat[m["col"]].astype(int).values
        nr = rows_v.max() - rows_v.min() + 1
        nc = cols_v.max() - cols_v.min() + 1
        vmap = np.zeros((nr, nc))
        r0, c0 = rows_v.min(), cols_v.min()
        for i in range(len(feat)):
            vmap[rows_v[i]-r0, cols_v[i]-c0] = vote.iloc[i] / n_mth
        im2 = ax.imshow(vmap, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_title("트레이 앙상블 불량 맵 (진할수록 다수 방법 불량 판정)")
        ax.set_xlabel("Col"); ax.set_ylabel("Row")
        plt.colorbar(im2, ax=ax, label="불량 판정 비율")
    else:
        sc = ax.scatter(d, K, c=vote, cmap="RdYlGn_r", s=12, alpha=0.7)
        ax.set_xlabel("ΔOCV (mV)"); ax.set_ylabel("K 기울기 (mV/day)")
        ax.set_title("ΔOCV vs K값 (색상 = 불량 판정 방법 수)")
        plt.colorbar(sc, ax=ax, label="불량 판정 방법 수")

    plt.tight_layout()

    if save_dir:
        out_png = os.path.join(save_dir, "dOCV_dashboard.png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"[저장] 대시보드 이미지: {out_png}")

    plt.show()


# ════════════════════════════════════════════════════════════
# 6. 결과 저장
# ════════════════════════════════════════════════════════════

def save_results(feat: pd.DataFrame, res: pd.DataFrame,
                 extra: dict, path: str):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # 시트1: 원본+특징+판정
        out = feat.copy()
        for key, s in extra.items():
            if hasattr(s, "values"):
                out[key] = s.values
        for col in res.columns:
            out[col] = res[col].values
        out.to_excel(writer, sheet_name="결과_전체", index=False)

        # 시트2: 요약
        summary = pd.DataFrame({
            "방법":      res.columns.tolist(),
            "불량판정수": [int(res[c].sum()) for c in res.columns],
            "불량율(%)": [round(res[c].mean()*100, 3) for c in res.columns],
        })
        summary.to_excel(writer, sheet_name="요약", index=False)

        # 시트3: 앙상블 불량 셀만
        vote = res.sum(axis=1)
        high_risk = out[vote >= 2]
        high_risk.to_excel(writer, sheet_name="다중불량셀(2개이상)", index=False)

    print(f"[저장] Excel: {path}")


# ════════════════════════════════════════════════════════════
# 7. GUI
# ════════════════════════════════════════════════════════════

class MappingDialog:
    def __init__(self, parent, df_cols, auto_m):
        self.result = None
        top = tk.Toplevel(parent)
        top.title("컬럼 매핑 & 분석 설정")
        top.geometry("560x680")
        top.grab_set()
        self._top = top

        ttk.Label(top, text="Excel 컬럼을 데이터 항목에 연결해주세요",
                  font=("Arial", 11, "bold")).pack(pady=10)

        frm = ttk.Frame(top, padding=15)
        frm.pack(fill="both", expand=True)

        fields = [
            ("ocv1",  "OCV1 (첫번째 전압)",  True),
            ("ocv2",  "OCV2 (두번째 전압)",  True),
            ("ocv3",  "OCV3 (세번째 전압)",  True),
            ("t1",    "T1 (OCV1 시점 온도)", False),
            ("t2",    "T2 (OCV2 시점 온도)", False),
            ("t3",    "T3 (OCV3 시점 온도)", False),
            ("row",   "Row (트레이 행)",     False),
            ("col",   "Col (트레이 열)",     False),
            ("label", "Label (불량 정답)",   False),
        ]
        opts = ["(없음)"] + list(df_cols)
        self._vars = {}

        for i, (key, lbl, req) in enumerate(fields):
            fg = "red" if req else "black"
            ttk.Label(frm, text=lbl + (" *" if req else ""),
                      foreground=fg).grid(row=i, column=0, sticky="w",
                                          padx=5, pady=3)
            v = tk.StringVar(value=auto_m.get(key, "(없음)"))
            ttk.Combobox(frm, textvariable=v, values=opts,
                         width=26).grid(row=i, column=1, padx=5, pady=3)
            self._vars[key] = v

        sep_row = len(fields)
        ttk.Separator(frm, orient="horizontal").grid(
            row=sep_row, column=0, columnspan=2, sticky="ew", pady=8)

        params = [
            ("k (MAD 배수, 기본 3.5)",    "k",     str(K_SIGMA)),
            ("바닥값 floor (mV)",          "floor", str(FLOOR_MV)),
            ("고정 오프셋 (mV, Baseline)", "offset",str(FIXED_OFFSET)),
            ("dt1 (OCV1→2 간격, 일)",      "dt1",   str(DT1_DAY)),
            ("dt2 (OCV2→3 간격, 일)",      "dt2",   str(DT2_DAY)),
            ("OCV 단위 (V 또는 mV)",       "unit",  "V"),
        ]
        self._pvars = {}
        for i, (lbl, key, default) in enumerate(params):
            ttk.Label(frm, text=lbl).grid(
                row=sep_row+1+i, column=0, sticky="w", padx=5, pady=3)
            if key == "unit":
                v = tk.StringVar(value=default)
                ttk.Combobox(frm, textvariable=v, values=["V", "mV"],
                             width=10).grid(row=sep_row+1+i, column=1,
                                            sticky="w", padx=5)
            else:
                v = tk.StringVar(value=default)
                ttk.Entry(frm, textvariable=v,
                          width=12).grid(row=sep_row+1+i, column=1,
                                         sticky="w", padx=5)
            self._pvars[key] = v

        ttk.Button(top, text="  분석 시작  ",
                   command=self._ok).pack(pady=12)
        top.wait_window()

    def _ok(self):
        mapping = {}
        for key, v in self._vars.items():
            val = v.get()
            if val and val != "(없음)":
                mapping[key] = val

        req = ["ocv1", "ocv2", "ocv3"]
        missing = [r for r in req if r not in mapping]
        if missing:
            messagebox.showerror("오류", f"필수 컬럼 매핑 없음: {missing}")
            return

        def _float(key, default):
            try:
                return float(self._pvars[key].get())
            except ValueError:
                return default

        self.result = {
            "mapping": mapping,
            "k":      _float("k",      K_SIGMA),
            "floor":  _float("floor",  FLOOR_MV),
            "offset": _float("offset", FIXED_OFFSET),
            "dt1":    _float("dt1",    DT1_DAY),
            "dt2":    _float("dt2",    DT2_DAY),
            "unit":   self._pvars["unit"].get(),
        }
        self._top.destroy()


def run_gui():
    root = tk.Tk()
    root.title("dOCV 선별 로직 분석기  v1.0")
    root.geometry("520x320")
    root.resizable(False, False)

    frm = ttk.Frame(root, padding=30)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="저전압 불량 선별 로직 고도화 분석기",
              font=("Arial", 13, "bold")).pack(pady=8)
    ttk.Label(frm,
              text="OCV 3개 + 온도 3개가 포함된 Excel 파일을 선택하세요",
              font=("Arial", 9)).pack()

    file_var = tk.StringVar(value="파일 미선택")
    ttk.Label(frm, textvariable=file_var,
              foreground="gray", font=("Arial", 9)).pack(pady=6)

    sel = [None]

    def pick():
        p = filedialog.askopenfilename(
            title="Excel 파일 선택",
            filetypes=[("Excel", "*.xlsx *.xls"), ("모든 파일", "*.*")])
        if p:
            sel[0] = p
            file_var.set(os.path.basename(p))

    def analyze():
        if not sel[0]:
            messagebox.showerror("오류", "먼저 Excel 파일을 선택해주세요.")
            return
        try:
            print(f"\n파일 로드: {sel[0]}")
            df = load_excel(sel[0])
            auto_m = auto_map(df)
            print(f"  자동 감지 매핑: {auto_m}")

            dlg = MappingDialog(root, df.columns, auto_m)
            if dlg.result is None:
                return

            cfg = dlg.result
            m   = cfg["mapping"]

            # V → mV 변환
            if cfg["unit"] == "V":
                for key in ["ocv1", "ocv2", "ocv3"]:
                    if key in m:
                        df[m[key]] = df[m[key]].astype(float) * 1000

            root.config(cursor="wait"); root.update()

            feat = compute_features(df, m, dt1=cfg["dt1"], dt2=cfg["dt2"])

            labels = None
            if "label" in m:
                labels = df[m["label"]]

            res, extra = run_all(feat, m, k=cfg["k"],
                                 floor=cfg["floor"], labels=labels)

            save_dir = os.path.dirname(sel[0])
            base     = os.path.splitext(os.path.basename(sel[0]))[0]
            out_xl   = os.path.join(save_dir, f"{base}_dOCV_결과.xlsx")
            save_results(feat, res, extra, out_xl)

            root.config(cursor=""); root.update()
            plot_dashboard(feat, res, extra, m, save_dir)

            messagebox.showinfo("완료",
                f"분석 완료!\n결과 저장: {out_xl}")

        except Exception as e:
            root.config(cursor="")
            messagebox.showerror("오류",
                f"분석 중 오류 발생:\n{e}\n\n{traceback.format_exc()[:600]}")

    ttk.Button(frm, text="  Excel 파일 선택  ", command=pick).pack(pady=8)
    ttk.Button(frm, text="  분석 시작  ",       command=analyze).pack(pady=4)
    ttk.Label(frm,
              text="결과 Excel과 PNG가 원본 파일 폴더에 저장됩니다",
              font=("Arial", 8), foreground="gray").pack(pady=8)

    root.mainloop()


# ════════════════════════════════════════════════════════════
# 8. CLI 모드 (python dOCV_analyzer.py <파일경로>)
# ════════════════════════════════════════════════════════════

def run_cli(path: str):
    print(f"파일: {path}")
    df   = load_excel(path)
    m    = auto_map(df)
    print(f"자동 감지: {m}")

    for key in ["ocv1", "ocv2", "ocv3"]:
        if key not in m:
            raise ValueError(f"필수 컬럼({key}) 자동 감지 실패."
                             f" 컬럼명 확인: {list(df.columns)}")

    # 단위 자동 감지 (중앙값이 5 이하면 V로 판단)
    med_v = float(df[m["ocv1"]].median())
    if med_v < 5.0:
        print(f"  OCV 중앙값={med_v:.4f} → V 단위로 판단, mV로 변환")
        for key in ["ocv1", "ocv2", "ocv3"]:
            df[m[key]] = df[m[key]].astype(float) * 1000

    feat       = compute_features(df, m)
    res, extra = run_all(feat, m)

    save_dir = os.path.dirname(os.path.abspath(path))
    base     = os.path.splitext(os.path.basename(path))[0]
    out_xl   = os.path.join(save_dir, f"{base}_dOCV_결과.xlsx")
    save_results(feat, res, extra, out_xl)
    plot_dashboard(feat, res, extra, m, save_dir)


# ════════════════════════════════════════════════════════════
# ENTRY
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli(sys.argv[1])
    elif TK_OK:
        run_gui()
    else:
        print("사용법: python dOCV_analyzer.py <Excel파일경로>")
