#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
저전압 불량 선별 로직 고도화 분석기  v1.1
dOCV 로직 비교: Baseline / C(DPAT) / A(온도회귀) / B(NNR) / D(K기울기) / E(ML)
- OCV 단위: mV 고정
- 온도 단위: °C 고정
- Label: A=양품(0), E=불량(1)
- Cell No → Row/Col 자동 변환 (1-144, A열-L열 × 1-12행)
- TRAY ID 기준 트레이별 상대판정
"""

import sys
import os
import warnings
import traceback
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _mad_normal(x):
    """MAD (정규분포 환산 ×1.4826), NaN 무시"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 0.0
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med)) * 1.4826


# ── 옵션 의존성 ──────────────────────────────────────────────
try:
    from sklearn.linear_model import HuberRegressor
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
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
K_SIGMA      = 3.5
FLOOR_MV     = 0.5
FIXED_OFFSET = 0.8
DT1_DAY      = 1.0
DT2_DAY      = 2.0

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
        "ocv1":    ["OCV1", "OCV_1", "V1", "VOLT1", "VOLTAGE1", "전압1"],
        "ocv2":    ["OCV2", "OCV_2", "V2", "VOLT2", "VOLTAGE2", "전압2"],
        "ocv3":    ["OCV3", "OCV_3", "V3", "VOLT3", "VOLTAGE3", "전압3"],
        "t1":      ["T1", "TEMP1", "TEMPERATURE1", "TMP1", "온도1"],
        "t2":      ["T2", "TEMP2", "TEMPERATURE2", "TMP2", "온도2"],
        "t3":      ["T3", "TEMP3", "TEMPERATURE3", "TMP3", "온도3"],
        "tray_id": ["TRAY_ID", "TRAY ID", "TRAY", "트레이ID", "트레이", "LOT_ID", "LOT"],
        "cell_no": ["CELL_NO", "CELL NO", "CELLNO", "셀번호", "CELL_NUMBER"],
        "label":   ["LABEL", "DEFECT", "FAULT", "NG", "불량", "결과", "RESULT",
                    "GRADE", "등급"],
    }
    for key, patterns in groups.items():
        for p in patterns:
            if p in uc:
                m[key] = uc[p]
                break
    return m


# ════════════════════════════════════════════════════════════
# 2. Cell No 변환 & Label 파싱
# ════════════════════════════════════════════════════════════

def cell_no_to_rowcol(series: pd.Series) -> tuple:
    """
    Cell No (1~144) → row (1~12), col (1~12)
    A01~A12 = 1~12  (col=1, A열)
    B01~B12 = 13~24 (col=2, B열)
    ...
    L01~L12 = 133~144 (col=12, L열)
    """
    n = series.astype(int)
    col = (n - 1) // 12 + 1   # 1~12 (A~L)
    row = (n - 1) % 12 + 1    # 1~12
    return row, col


def cell_no_to_pos_label(series: pd.Series) -> pd.Series:
    """Cell No → 'A01', 'B03' 형태 위치 문자열"""
    n = series.astype(int)
    col_idx = (n - 1) // 12          # 0-indexed
    row_idx = (n - 1) % 12 + 1       # 1~12
    return pd.Series(
        [f"{chr(65 + c)}{r:02d}" for c, r in zip(col_idx, row_idx)],
        index=series.index
    )


def parse_label(series: pd.Series) -> pd.Series:
    """A(양품)→0, E(불량)→1, 이미 숫자면 그대로"""
    def _conv(v):
        if isinstance(v, str):
            return 1 if v.strip().upper() == "E" else 0
        return int(v)
    return series.map(_conv)


# ════════════════════════════════════════════════════════════
# 3. 특징량 계산
# ════════════════════════════════════════════════════════════

def compute_features(df: pd.DataFrame, m: dict,
                     dt1: float = DT1_DAY,
                     dt2: float = DT2_DAY) -> tuple:
    """
    특징량 계산 후 (feat_df, updated_mapping) 반환
    Cell No → _row, _col 자동 추가
    """
    out = df.copy()
    m = m.copy()

    # Cell No → Row / Col
    if "cell_no" in m:
        row, col = cell_no_to_rowcol(df[m["cell_no"]])
        out["_row"] = row
        out["_col"] = col
        out["position"] = cell_no_to_pos_label(df[m["cell_no"]])
        m["row"] = "_row"
        m["col"] = "_col"

    o1 = df[m["ocv1"]].astype(float).values
    o2 = df[m["ocv2"]].astype(float).values
    o3 = df[m["ocv3"]].astype(float).values

    out["dOCV_raw"] = o1 - o3
    out["dOCV_12"]  = o1 - o2
    out["dOCV_23"]  = o2 - o3
    out["OCV_mean"] = (o1 + o2 + o3) / 3

    # 3점 직선 적합 → K값(mV/day)
    t_pts = np.array([0.0, dt1, dt1 + dt2])
    K_list, nonlin_list = [], []
    for i in range(len(out)):
        y = np.array([o1[i], o2[i], o3[i]])
        coeff = np.polyfit(t_pts, y, 1)
        K_list.append(-coeff[0])
        y_fit = np.polyval(coeff, t_pts)
        nonlin_list.append(float(np.max(np.abs(y - y_fit))))

    out["K_slope"]      = K_list
    out["nonlin_resid"] = nonlin_list
    out["consistent"]   = (out["dOCV_12"] > 0) & (out["dOCV_23"] > 0)

    # 온도
    if all(k in m for k in ["t1", "t2", "t3"]):
        T1 = df[m["t1"]].astype(float).values
        T2 = df[m["t2"]].astype(float).values
        T3 = df[m["t3"]].astype(float).values
        out["T1"]      = T1
        out["T2"]      = T2
        out["T3"]      = T3
        out["T_avg"]   = (T1 + T2 + T3) / 3
        out["T_delta"] = T1 - T3
    elif "t1" in m:
        T1 = df[m["t1"]].astype(float).values
        out["T1"] = out["T_avg"] = T1
        out["T_delta"] = 0.0

    return out, m


# ════════════════════════════════════════════════════════════
# 4. 판정 로직 (단일 트레이 또는 전체 데이터셋)
# ════════════════════════════════════════════════════════════

def method_baseline(feat: pd.DataFrame,
                    offset: float = FIXED_OFFSET) -> pd.Series:
    """mode + 고정 오프셋"""
    d = feat["dOCV_raw"]
    mode_v = float((d * 1000).round().mode().iloc[0] / 1000)
    return d > (mode_v + offset)


def method_dpat(feat: pd.DataFrame,
                k: float = K_SIGMA,
                floor: float = FLOOR_MV,
                col: str = "dOCV_raw") -> pd.Series:
    """median + k·MAD (DPAT)"""
    d = feat[col]
    med = float(np.nanmedian(d))
    mad = _mad_normal(d)
    return d > max(med + k * mad, med + floor)


def method_temp_regression(feat: pd.DataFrame,
                           k: float = K_SIGMA,
                           floor: float = FLOOR_MV) -> tuple:
    """ΔOCV ~ T 로버스트 회귀 → 잔차 DPAT. (flags, residuals, expected)"""
    if "T_avg" not in feat.columns:
        flags = method_dpat(feat, k, floor)
        return flags, feat["dOCV_raw"].copy(), pd.Series(0.0, index=feat.index)

    d = feat["dOCV_raw"].values
    T = feat["T_avg"].values

    # NaN이 있는 행 제외하고 적합, 예측은 NaN 위치에 트레이 평균 온도로 대체
    valid = ~(np.isnan(T) | np.isnan(d))
    if valid.sum() < 4:
        flags = method_dpat(feat, k, floor)
        return flags, feat["dOCV_raw"].copy(), pd.Series(0.0, index=feat.index)

    T_fill = np.where(np.isnan(T), np.nanmean(T), T)   # NaN → 평균 온도

    if SKLEARN_OK:
        reg = HuberRegressor(epsilon=1.5, max_iter=300)
        reg.fit(T_fill[valid].reshape(-1, 1), d[valid])
        d_exp = reg.predict(T_fill.reshape(-1, 1))
    else:
        d_exp = _robust_linreg_numpy(T_fill, d, valid_mask=valid)

    resid    = pd.Series(d - d_exp, index=feat.index)
    expected = pd.Series(d_exp,     index=feat.index)
    med = float(np.median(resid))
    mad = _mad_normal(resid)
    flags = resid > max(med + k * mad, med + floor)
    return flags, resid, expected


def _robust_linreg_numpy(T, y, n_iter=10, valid_mask=None):
    """반복적 이상치 제거 선형 회귀 (numpy 전용). valid_mask로 NaN 행 제외."""
    if valid_mask is None:
        valid_mask = ~(np.isnan(T) | np.isnan(y))
    mask = valid_mask.copy()
    if mask.sum() < 4:
        return np.full(len(T), np.nanmean(y))
    coeff = np.polyfit(T[mask], y[mask], 1)
    for _ in range(n_iter):
        y_fit = np.polyval(coeff, T)
        resid = np.abs(y - y_fit)
        mad = _mad_normal(resid[mask])
        if mad < 1e-12:
            break
        mask = valid_mask & (resid <= 2.5 * mad)
        if mask.sum() < 4:
            break
        coeff = np.polyfit(T[mask], y[mask], 1)
    return np.polyval(coeff, T)


def method_nnr(feat: pd.DataFrame, m: dict,
               kernel: int = 3,
               k: float = K_SIGMA,
               floor: float = FLOOR_MV) -> tuple:
    """NNR 공간 이웃 잔차. (flags, nnr_residuals)"""
    if "row" not in m or "col" not in m:
        return method_dpat(feat, k, floor), feat["dOCV_raw"].copy()

    rows  = feat[m["row"]].astype(int).values
    cols  = feat[m["col"]].astype(int).values
    d_arr = feat["dOCV_raw"].values
    half  = kernel // 2

    r_min, c_min = rows.min(), cols.min()
    nr = rows.max() - r_min + 1 + 2 * kernel
    nc = cols.max() - c_min + 1 + 2 * kernel
    grid = np.full((nr, nc), np.nan)
    for i in range(len(feat)):
        grid[rows[i] - r_min + kernel, cols[i] - c_min + kernel] = d_arr[i]

    nnr = np.full(len(feat), 0.0)
    for i in range(len(feat)):
        ri = rows[i] - r_min + kernel
        ci = cols[i] - c_min + kernel
        patch = grid[ri - half: ri + half + 1,
                     ci - half: ci + half + 1].flatten()
        nb = patch[~np.isnan(patch)]
        nb = nb[nb != d_arr[i]] if len(nb) > 1 else nb
        if len(nb) > 0:
            nnr[i] = d_arr[i] - np.median(nb)

    resid = pd.Series(nnr, index=feat.index)
    med = float(np.median(resid))
    mad = _mad_normal(resid)
    return resid > max(med + k * mad, med + floor), resid


def method_kslope(feat: pd.DataFrame,
                  k: float = K_SIGMA,
                  floor: float = FLOOR_MV,
                  require_consistent: bool = False) -> pd.Series:
    flags = method_dpat(feat, k, floor, col="K_slope")
    if require_consistent:
        flags = flags & feat["consistent"]
    return flags


def method_ml(feat: pd.DataFrame,
              contamination: float = 0.05) -> tuple:
    """다변량 이상탐지. sklearn 있으면 IF+LOF, 없으면 로버스트 Z점수."""
    fcols = ["dOCV_raw", "K_slope", "nonlin_resid", "dOCV_12", "dOCV_23"]
    if "T_avg"   in feat.columns: fcols.append("T_avg")
    if "T_delta" in feat.columns: fcols.append("T_delta")

    X = feat[fcols].fillna(0).values.astype(float)

    if SKLEARN_OK and len(X) >= 10:
        Xs = StandardScaler().fit_transform(X)
        iso = IsolationForest(contamination=contamination,
                              n_estimators=200, random_state=42)
        iso.fit(Xs)
        s1 = -iso.score_samples(Xs)

        lof = LocalOutlierFactor(n_neighbors=min(20, len(Xs) - 1),
                                 contamination=contamination)
        lof.fit_predict(Xs)
        s2 = -lof.negative_outlier_factor_

        norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
        score = (norm(s1) + norm(s2)) / 2
    else:
        med = np.median(X, axis=0)
        mad = np.array([_mad_normal(X[:, j]) for j in range(X.shape[1])])
        mad = np.where(mad < 1e-12, 1.0, mad)
        Xz  = np.abs(X - med) / mad
        w   = np.ones(len(fcols))
        for i, c in enumerate(fcols):
            if c in ("dOCV_raw", "K_slope"):
                w[i] = 2.0
        score = (Xz * w).sum(axis=1)
        score = (score - score.min()) / (score.max() - score.min() + 1e-12)

    thr = np.percentile(score, 100 * (1 - contamination))
    return pd.Series(score >= thr, index=feat.index), pd.Series(score, index=feat.index)


# ════════════════════════════════════════════════════════════
# 5. 트레이별 판정 실행 (핵심)
# ════════════════════════════════════════════════════════════

def _run_one_tray(feat_t: pd.DataFrame, m: dict,
                  k: float, floor: float) -> tuple:
    """단일 트레이(또는 전체) 데이터에 모든 방법 적용"""
    res   = pd.DataFrame(index=feat_t.index)
    extra = {}

    res["Baseline"]        = method_baseline(feat_t).astype(int)
    res["C_DPAT"]          = method_dpat(feat_t, k, floor).astype(int)

    fa, ra, ea             = method_temp_regression(feat_t, k, floor)
    res["A_TempReg"]       = fa.astype(int)
    extra["A_residual"]    = ra
    extra["A_expected"]    = ea

    res["D_Kslope"]        = method_kslope(feat_t, k, floor).astype(int)
    res["D_Kslope_strict"] = method_kslope(feat_t, k, floor,
                                           require_consistent=True).astype(int)

    if "row" in m and "col" in m:
        fb, rb              = method_nnr(feat_t, m, k=k, floor=floor)
        res["B_NNR"]        = fb.astype(int)
        extra["B_residual"] = rb
        # A+B 결합 (OR)
        fa2, _, _           = method_temp_regression(feat_t, k, floor)
        fb2, _              = method_nnr(feat_t, m, k=k, floor=floor)
        res["AB_Combined"]  = (fa2 | fb2).astype(int)

    fe, se                 = method_ml(feat_t)
    res["E_ML"]            = fe.astype(int)
    extra["E_score"]       = se

    return res, extra


def run_all(feat: pd.DataFrame, m: dict,
            k: float = K_SIGMA,
            floor: float = FLOOR_MV,
            labels: pd.Series = None,
            t_min: float = None,
            t_max: float = None) -> tuple:
    """
    TRAY ID 기준 트레이별 상대판정.
    t_min / t_max 설정 시 해당 온도 범위 밖 셀은 분석 제외(-1).
    """
    tray_col = m.get("tray_id")

    # ── 온도 필터 마스크 ────────────────────────────────────
    temp_mask = pd.Series(True, index=feat.index)
    if "T_avg" in feat.columns and (t_min is not None or t_max is not None):
        T_avg = feat["T_avg"]
        if t_min is not None:
            temp_mask &= (T_avg >= t_min)
        if t_max is not None:
            temp_mask &= (T_avg <= t_max)
        n_excl = (~temp_mask).sum()
        lo  = str(t_min) if t_min is not None else "제한없음"
        hi  = str(t_max) if t_max is not None else "제한없음"
        rng = f"{lo} ~ {hi}"
        print(f"\n  [온도 필터] {rng} °C  →  분석 대상 {temp_mask.sum()}셀 / 제외 {n_excl}셀")

    print("\n" + "="*60)
    if tray_col and tray_col in feat.columns:
        n_trays = feat[tray_col].nunique()
        print(f" 트레이별 상대판정  ({n_trays}개 트레이)")
    else:
        print(" 전체 데이터 일괄 판정 (TRAY ID 없음)")
    print("="*60)

    # 결과 초기화: -1 = 분석 제외
    METHOD_COLS = ["Baseline", "C_DPAT", "A_TempReg",
                   "D_Kslope", "D_Kslope_strict",
                   "B_NNR", "AB_Combined", "E_ML"]
    res   = pd.DataFrame(-1, index=feat.index, columns=METHOD_COLS)
    extra = {}

    def _run_group(group_feat, group_mask):
        """온도 필터 적용 후 분석 대상만 추려서 실행"""
        sub = group_feat[group_mask.reindex(group_feat.index, fill_value=False)]
        if len(sub) == 0:
            return pd.DataFrame(-1, index=group_feat.index, columns=METHOD_COLS), {}
        sub_res, sub_extra = _run_one_tray(sub, m, k, floor)
        # 제외 셀은 -1 유지
        full_res = pd.DataFrame(-1, index=group_feat.index, columns=sub_res.columns)
        full_res.update(sub_res)
        return full_res, sub_extra

    if tray_col and tray_col in feat.columns:
        res_parts   = []
        extra_parts = {}
        for tray_id, group in feat.groupby(feat[tray_col], sort=False):
            tray_res, tray_extra = _run_group(group, temp_mask)
            res_parts.append(tray_res)
            for key, val in tray_extra.items():
                extra_parts.setdefault(key, []).append(val)
        res   = pd.concat(res_parts).reindex(feat.index)
        extra = {key: pd.concat(vals).reindex(feat.index)
                 for key, vals in extra_parts.items()}
    else:
        res, extra = _run_group(feat, temp_mask)

    # ── 요약 출력 (제외 셀 제외하고 집계) ────────────────────
    n_analyzed = int(temp_mask.sum())
    print(f"\n  {'방법':<25} {'불량수':>6} {'불량율(분석대상)':>16}")
    print("  " + "-"*50)
    for col in res.columns:
        analyzed = res[col][res[col] >= 0]   # -1 제외
        n   = int((analyzed == 1).sum())
        pct = n / max(n_analyzed, 1) * 100
        print(f"  {col:<25} {n:>6}  {pct:>15.2f}%")

    if labels is not None:
        _eval_all(res, labels, temp_mask)

    return res, extra


def _eval_all(res: pd.DataFrame, labels: pd.Series,
              temp_mask: pd.Series = None):
    """감도/특이도 계산. 분석 제외(-1) 셀은 무시."""
    y = labels.astype(int).values
    print(f"\n  {'방법':<25} {'감도':>7} {'특이도':>7} {'미검':>5} {'과검':>5}")
    print("  " + "-"*52)
    for col in res.columns:
        yp_raw = res[col].values
        # -1(제외) 셀은 평가에서 빼기
        valid  = yp_raw >= 0
        if temp_mask is not None:
            valid &= temp_mask.values
        yp = yp_raw[valid]
        yt = y[valid]
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        print(f"  {col:<25} {sens:>6.3f}  {spec:>6.3f}  {fn:>4}  {fp:>4}")


# ════════════════════════════════════════════════════════════
# 6. 시각화
# ════════════════════════════════════════════════════════════

def plot_dashboard(feat: pd.DataFrame, res: pd.DataFrame,
                   extra: dict, m: dict, save_dir: str = None):
    if not MPLOT_OK:
        print("[경고] matplotlib 없음 → 시각화 생략")
        return

    plt.rcParams["font.family"]        = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle("dOCV 선별 로직 고도화 - 방법 비교 대시보드",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    d     = feat["dOCV_raw"]
    K     = feat["K_slope"]
    n_mth = len(res.columns)

    # (0,0) ΔOCV 분포 & 임계값
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(d, bins=60, color="steelblue", alpha=0.75, edgecolor="white")
    med_d  = float(np.median(d))
    mad_d  = _mad_normal(d)
    mode_d = float((d * 1000).round().mode().iloc[0] / 1000)
    ax.axvline(med_d, color="orange", lw=2, label=f"Median {med_d:.3f}")
    ax.axvline(mode_d, color="red", lw=2, ls="--", label=f"Mode {mode_d:.3f}")
    ax.axvline(med_d + K_SIGMA * mad_d, color="green", lw=1.5, ls=":",
               label=f"DPAT({K_SIGMA}σ)")
    ax.axvline(mode_d + FIXED_OFFSET, color="purple", lw=1.5, ls="-.",
               label=f"Baseline(+{FIXED_OFFSET}mV)")
    ax.set_title("ΔOCV 분포 & 임계값 비교")
    ax.set_xlabel("ΔOCV (mV)")
    ax.legend(fontsize=7)

    # (0,1) 온도 vs ΔOCV
    ax = fig.add_subplot(gs[0, 1])
    if "T_avg" in feat.columns:
        T         = feat["T_avg"]
        base_flag = res["Baseline"].astype(bool)
        ax.scatter(T[~base_flag], d[~base_flag],
                   c="steelblue", s=8, alpha=0.4, label="양품")
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

    # (0,2) K값 분포
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(K, bins=60, color="mediumpurple", alpha=0.75, edgecolor="white")
    med_K = float(np.median(K))
    thr_K = med_K + K_SIGMA * _mad_normal(K)
    ax.axvline(med_K, color="orange", lw=2, label=f"Median {med_K:.4f}")
    ax.axvline(thr_K, color="red", lw=2, ls="--",
               label=f"임계값({K_SIGMA}σ) {thr_K:.4f}")
    ax.set_title("K값(3점 기울기) 분포")
    ax.set_xlabel("K (mV/day)")
    ax.legend(fontsize=7)

    # (1,0~1) 방법별 불량 수
    ax = fig.add_subplot(gs[1, :2])
    names  = list(res.columns)
    counts = [int(res[c].sum()) for c in names]
    palette = ["#e74c3c","#3498db","#2ecc71","#9b59b6",
               "#f39c12","#1abc9c","#e67e22","#34495e","#16a085"]
    bars = ax.bar(names, counts, color=palette[:len(names)], edgecolor="white")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(cnt), ha="center", va="bottom", fontsize=9)
    ax.set_title("방법별 불량 판정 셀 수")
    ax.set_ylabel("불량 판정 수")
    ax.tick_params(axis="x", rotation=35)

    # (1,2) Jaccard 히트맵
    ax = fig.add_subplot(gs[1, 2])
    J = np.zeros((n_mth, n_mth))
    for i, mi in enumerate(names):
        for j, mj in enumerate(names):
            inter = (res[mi].astype(bool) & res[mj].astype(bool)).sum()
            union = (res[mi].astype(bool) | res[mj].astype(bool)).sum()
            J[i, j] = inter / (union + 1e-9)
    im = ax.imshow(J, cmap="RdYlGn", vmin=0, vmax=1)
    short = [n[:8] for n in names]
    ax.set_xticks(range(n_mth)); ax.set_xticklabels(short, rotation=45, fontsize=7)
    ax.set_yticks(range(n_mth)); ax.set_yticklabels(short, fontsize=7)
    ax.set_title("방법 간 Jaccard 동의율")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (2, :) 트레이 맵 or 산점도
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
            vmap[rows_v[i] - r0, cols_v[i] - c0] = vote.iloc[i] / n_mth
        im2 = ax.imshow(vmap, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        # X축 레이블: A~L
        ax.set_xticks(range(nc))
        ax.set_xticklabels([chr(65 + c0 - 1 + i) for i in range(nc)], fontsize=9)
        ax.set_yticks(range(nr))
        ax.set_yticklabels(range(r0, r0 + nr), fontsize=9)
        ax.set_title("트레이 앙상블 불량 맵 (진할수록 다수 방법 불량 판정)")
        ax.set_xlabel("열 (A~L)"); ax.set_ylabel("행 (1~12)")
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
        print(f"[저장] 대시보드: {out_png}")

    plt.show()


# ════════════════════════════════════════════════════════════
# 7. 결과 저장
# ════════════════════════════════════════════════════════════

def save_results(feat: pd.DataFrame, res: pd.DataFrame,
                 extra: dict, path: str):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        out = feat.copy()
        for key, s in extra.items():
            if hasattr(s, "values"):
                out[key] = s.values
        for col in res.columns:
            out[col] = res[col].values
        out.to_excel(writer, sheet_name="결과_전체", index=False)

        summary = pd.DataFrame({
            "방법":      res.columns.tolist(),
            "불량판정수": [int(res[c].sum()) for c in res.columns],
            "불량율(%)": [round(res[c].mean() * 100, 3) for c in res.columns],
        })
        summary.to_excel(writer, sheet_name="요약", index=False)

        vote = res.sum(axis=1)
        out[vote >= 2].to_excel(writer, sheet_name="다중불량셀(2개이상)", index=False)

    print(f"[저장] Excel: {path}")


# ════════════════════════════════════════════════════════════
# 8. GUI
# ════════════════════════════════════════════════════════════

class MappingDialog:
    def __init__(self, parent, df_cols, auto_m):
        self.result = None
        top = tk.Toplevel(parent)
        top.title("컬럼 매핑 & 분석 설정")
        top.geometry("560x700")
        top.grab_set()
        self._top = top

        ttk.Label(top, text="Excel 컬럼을 데이터 항목에 연결해주세요",
                  font=("Arial", 11, "bold")).pack(pady=10)

        frm = ttk.Frame(top, padding=15)
        frm.pack(fill="both", expand=True)

        fields = [
            ("ocv1",    "OCV1 (첫번째 전압, mV)",   True),
            ("ocv2",    "OCV2 (두번째 전압, mV)",   True),
            ("ocv3",    "OCV3 (세번째 전압, mV)",   True),
            ("t1",      "T1 (OCV1 시점 온도, °C)",  False),
            ("t2",      "T2 (OCV2 시점 온도, °C)",  False),
            ("t3",      "T3 (OCV3 시점 온도, °C)",  False),
            ("tray_id", "TRAY ID (트레이 구분)",     False),
            ("cell_no", "Cell No (1~144)",          False),
            ("label",   "Label (A=양품, E=불량)",    False),
        ]
        opts = ["(없음)"] + list(df_cols)
        self._vars = {}

        for i, (key, lbl, req) in enumerate(fields):
            ttk.Label(frm, text=lbl + (" *" if req else ""),
                      foreground="red" if req else "black").grid(
                row=i, column=0, sticky="w", padx=5, pady=3)
            v = tk.StringVar(value=auto_m.get(key, "(없음)"))
            ttk.Combobox(frm, textvariable=v, values=opts,
                         width=26).grid(row=i, column=1, padx=5, pady=3)
            self._vars[key] = v

        sep = len(fields)
        ttk.Separator(frm, orient="horizontal").grid(
            row=sep, column=0, columnspan=2, sticky="ew", pady=8)

        params = [
            ("k (MAD 배수, 기본 3.5)",       "k",     str(K_SIGMA)),
            ("바닥값 floor (mV)",             "floor", str(FLOOR_MV)),
            ("dt1 (OCV1→2 간격, 일)",         "dt1",   str(DT1_DAY)),
            ("dt2 (OCV2→3 간격, 일)",         "dt2",   str(DT2_DAY)),
            ("온도 필터 최솟값 °C (빈칸=없음)", "t_min", ""),
            ("온도 필터 최댓값 °C (빈칸=없음)", "t_max", ""),
        ]
        self._pvars = {}
        for i, (lbl, key, default) in enumerate(params):
            ttk.Label(frm, text=lbl).grid(
                row=sep + 1 + i, column=0, sticky="w", padx=5, pady=3)
            v = tk.StringVar(value=default)
            ttk.Entry(frm, textvariable=v, width=12).grid(
                row=sep + 1 + i, column=1, sticky="w", padx=5)
            self._pvars[key] = v

        ttk.Button(top, text="  분석 시작  ", command=self._ok).pack(pady=12)
        top.wait_window()

    def _ok(self):
        mapping = {}
        for key, v in self._vars.items():
            val = v.get()
            if val and val != "(없음)":
                mapping[key] = val

        for r in ["ocv1", "ocv2", "ocv3"]:
            if r not in mapping:
                messagebox.showerror("오류", f"필수 컬럼 누락: {r}")
                return

        def _f(key, default):
            try:   return float(self._pvars[key].get())
            except: return default

        def _fopt(key):
            """빈 문자열이면 None 반환"""
            v = self._pvars[key].get().strip()
            try:   return float(v) if v else None
            except: return None

        self.result = {
            "mapping": mapping,
            "k":     _f("k",    K_SIGMA),
            "floor": _f("floor",FLOOR_MV),
            "dt1":   _f("dt1",  DT1_DAY),
            "dt2":   _f("dt2",  DT2_DAY),
            "t_min": _fopt("t_min"),
            "t_max": _fopt("t_max"),
        }
        self._top.destroy()


def run_gui():
    root = tk.Tk()
    root.title("dOCV 선별 로직 분석기  v1.1")
    root.geometry("520x330")
    root.resizable(False, False)

    frm = ttk.Frame(root, padding=30)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="저전압 불량 선별 로직 고도화 분석기",
              font=("Arial", 13, "bold")).pack(pady=8)
    ttk.Label(frm, text="OCV(mV) + 온도(°C) + TRAY ID + Cell No 포함 Excel 파일",
              font=("Arial", 9)).pack()
    ttk.Label(frm, text="Label: A=양품, E=불량",
              font=("Arial", 9), foreground="gray").pack()

    file_var = tk.StringVar(value="파일 미선택")
    ttk.Label(frm, textvariable=file_var,
              foreground="gray", font=("Arial", 9)).pack(pady=5)

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
            df     = load_excel(sel[0])
            auto_m = auto_map(df)
            dlg    = MappingDialog(root, df.columns, auto_m)
            if dlg.result is None:
                return

            cfg = dlg.result
            m   = cfg["mapping"]

            root.config(cursor="wait"); root.update()

            feat, m = compute_features(df, m, dt1=cfg["dt1"], dt2=cfg["dt2"])

            labels = None
            if "label" in m:
                labels = parse_label(df[m["label"]])

            res, extra = run_all(feat, m, k=cfg["k"],
                                 floor=cfg["floor"], labels=labels,
                                 t_min=cfg["t_min"], t_max=cfg["t_max"])

            save_dir = os.path.dirname(sel[0])
            base     = os.path.splitext(os.path.basename(sel[0]))[0]
            out_xl   = os.path.join(save_dir, f"{base}_dOCV_결과.xlsx")
            save_results(feat, res, extra, out_xl)

            root.config(cursor=""); root.update()
            plot_dashboard(feat, res, extra, m, save_dir)
            messagebox.showinfo("완료", f"분석 완료!\n결과: {out_xl}")

        except Exception as e:
            root.config(cursor="")
            messagebox.showerror("오류",
                f"{e}\n\n{traceback.format_exc()[:600]}")

    ttk.Button(frm, text="  Excel 파일 선택  ", command=pick).pack(pady=8)
    ttk.Button(frm, text="  분석 시작  ",       command=analyze).pack(pady=4)
    ttk.Label(frm, text="결과 Excel과 PNG가 원본 파일 폴더에 저장됩니다",
              font=("Arial", 8), foreground="gray").pack(pady=6)

    root.mainloop()


# ════════════════════════════════════════════════════════════
# 9. CLI 모드
# ════════════════════════════════════════════════════════════

def run_cli(path: str):
    df   = load_excel(path)
    m    = auto_map(df)
    print(f"자동 감지: {m}")

    for key in ["ocv1", "ocv2", "ocv3"]:
        if key not in m:
            raise ValueError(f"필수 컬럼({key}) 자동 감지 실패. "
                             f"컬럼 목록: {list(df.columns)}")

    feat, m = compute_features(df, m)

    labels = None
    if "label" in m:
        labels = parse_label(df[m["label"]])

    res, extra = run_all(feat, m, labels=labels)

    save_dir = os.path.dirname(os.path.abspath(path))
    base     = os.path.splitext(os.path.basename(path))[0]
    out_xl   = os.path.join(save_dir, f"{base}_dOCV_결과.xlsx")
    save_results(feat, res, extra, out_xl)
    plot_dashboard(feat, res, extra, m, save_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli(sys.argv[1])
    elif TK_OK:
        run_gui()
    else:
        print("사용법: python dOCV_analyzer.py <Excel파일경로>")
