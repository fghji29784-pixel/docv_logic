#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
온도-OCV 진단 스크립트  (dU/dT 보정 필요성 판정용)
한 장의 PNG에 5개 진단 그림 + 자동 해석을 출력.

사용법:
    python dOCV_temp_diagnostic.py <엑셀파일경로>
    또는 아래 CONFIG에서 FILE_PATH 지정 후 그냥 실행

필요 컬럼(자동 감지): OCV1~3(mV), T1~3(°C), Cell No 또는 Row/Col, (선택)TRAY ID
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")   # 폰트/레이아웃 경고 억제

import matplotlib
matplotlib.use("Agg")           # 화면 없이 PNG 저장
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ════════════════════════════════════════════════════════════
# CONFIG — 필요 시 수동 지정 (비워두면 자동 감지)
# ════════════════════════════════════════════════════════════
FILE_PATH = ""          # 예: r"C:\data\lot_2026.xlsx"  (비우면 인자/탐색)
COL_OVERRIDE = {        # 자동 감지가 틀리면 여기서 수동 지정
    # "ocv1": "OCV1", "t1": "T1", "cell_no": "Cell No", "tray_id": "TRAY ID",
}
OCV_UNIT = "mV"         # "mV" 또는 "V" (V면 자동으로 ×1000)


# ════════════════════════════════════════════════════════════
# 컬럼 자동 매핑 (메인 분석기와 동일 규칙)
# ════════════════════════════════════════════════════════════
def auto_map(df):
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
        "row":     ["ROW", "TRAY_ROW", "행", "Y"],
        "col":     ["COL", "COLUMN", "TRAY_COL", "열", "X"],
    }
    for key, patterns in groups.items():
        for p in patterns:
            if p in uc:
                m[key] = uc[p]
                break
    m.update(COL_OVERRIDE)
    return m


def cell_no_to_rowcol(series):
    """Cell No(1~144) → row(1~12), col(1~12). A01~A12=1~12, B01~=13~..."""
    n = series.astype(int)
    col = (n - 1) // 12 + 1
    row = (n - 1) % 12 + 1
    return row, col


def load_excel(path):
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        if len(df) > 5:
            print(f"  시트 '{sheet}' 로드: {len(df)}행 {len(df.columns)}열")
            return df
    return pd.read_excel(path, sheet_name=0)


# ════════════════════════════════════════════════════════════
# 메인 진단
# ════════════════════════════════════════════════════════════
def run(path):
    df = load_excel(path)
    m  = auto_map(df)
    print(f"  컬럼 매핑: {m}")

    need = ["ocv1", "ocv2", "ocv3", "t1", "t2", "t3"]
    miss = [k for k in need if k not in m]
    if miss:
        raise ValueError(f"필수 컬럼 누락: {miss}\n실제 컬럼: {list(df.columns)}\n"
                         f"→ 스크립트 상단 COL_OVERRIDE로 수동 지정하세요.")

    # 값 추출
    o1 = df[m["ocv1"]].astype(float).values
    o3 = df[m["ocv3"]].astype(float).values
    if OCV_UNIT.upper() == "V":
        o1, o3 = o1 * 1000, o3 * 1000
    T1 = df[m["t1"]].astype(float).values
    T2 = df[m["t2"]].astype(float).values
    T3 = df[m["t3"]].astype(float).values

    dOCV  = o1 - o3
    T_avg = (T1 + T2 + T3) / 3
    dT_13 = T1 - T3
    dT_12 = T1 - T2
    dT_23 = T2 - T3

    # Row/Col 확보
    has_pos = False
    if "cell_no" in m:
        row, col = cell_no_to_rowcol(df[m["cell_no"]])
        row, col = row.values, col.values
        has_pos = True
    elif "row" in m and "col" in m:
        row = df[m["row"]].astype(int).values
        col = df[m["col"]].astype(int).values
        has_pos = True

    # ── 그림 ────────────────────────────────────────────────
    plt.rcParams["font.family"]        = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(21, 12))
    fig.suptitle("온도-OCV 진단  (dU/dT 보정 필요성 판정)",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.27)

    # 결측 제거 헬퍼
    def clean(*arrs):
        mask = np.ones(len(arrs[0]), dtype=bool)
        for a in arrs:
            mask &= ~np.isnan(a)
        return [a[mask] for a in arrs], mask

    # (1) T1-T3, T1-T2, T2-T3 분포  ★dU/dT 핵심
    ax = fig.add_subplot(gs[0, 0])
    for arr, lbl, c in [(dT_13, "T1−T3", "crimson"),
                        (dT_12, "T1−T2", "steelblue"),
                        (dT_23, "T2−T3", "seagreen")]:
        a = arr[~np.isnan(arr)]
        ax.hist(a, bins=80, alpha=0.5, label=lbl, color=c)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_title("① 시점 간 온도차 분포  ★dU/dT 판정", fontweight="bold")
    ax.set_xlabel("온도차 (°C)"); ax.set_ylabel("셀 수")
    ax.legend(fontsize=9)

    # (2) 온도 vs ΔOCV, 위치(Col)로 색칠
    ax = fig.add_subplot(gs[0, 1])
    if has_pos:
        (Tc, dc, cc), _ = clean(T_avg, dOCV, col.astype(float))
        sc = ax.scatter(Tc, dc, c=cc, cmap="viridis", s=6, alpha=0.5)
        plt.colorbar(sc, ax=ax, label="열 위치 (Col 1~12)")
    else:
        (Tc, dc), _ = clean(T_avg, dOCV)
        ax.scatter(Tc, dc, s=6, alpha=0.4, color="steelblue")
    ax.set_title("② 온도 vs ΔOCV (색=위치)")
    ax.set_xlabel("T_avg (°C)"); ax.set_ylabel("ΔOCV (mV)")

    # (3) 대표 트레이 온도 히트맵
    ax = fig.add_subplot(gs[0, 2])
    heat_info = ""
    if has_pos:
        # 대표 트레이 선택: TRAY ID 있으면 셀 最多 트레이, 없으면 전체
        if "tray_id" in m:
            tids = df[m["tray_id"]].values
            uniq, counts = np.unique(tids, return_counts=True)
            pick = uniq[np.argmax(counts)]
            sel = (tids == pick)
            heat_info = f"트레이 {pick}"
        else:
            sel = np.ones(len(df), dtype=bool)
            heat_info = "전체"
        nr = int(np.nanmax(row)); nc = int(np.nanmax(col))
        grid = np.full((nr, nc), np.nan)
        for i in np.where(sel)[0]:
            if not np.isnan(T_avg[i]):
                grid[int(row[i]) - 1, int(col[i]) - 1] = T_avg[i]
        im = ax.imshow(grid, cmap="inferno", aspect="auto")
        ax.set_xticks(range(nc))
        ax.set_xticklabels([chr(65 + i) for i in range(nc)], fontsize=8)
        ax.set_yticks(range(nr)); ax.set_yticklabels(range(1, nr + 1), fontsize=8)
        ax.set_xlabel("열 (A~L)"); ax.set_ylabel("행 (1~12)")
        plt.colorbar(im, ax=ax, label="T_avg (°C)")
        # 공간 매끄러움 지표: 이웃 차이 vs 전체 산포
        neigh = []
        for r in range(nr):
            for c in range(nc - 1):
                a, b = grid[r, c], grid[r, c + 1]
                if not (np.isnan(a) or np.isnan(b)):
                    neigh.append(abs(a - b))
        gstd = np.nanstd(grid)
        smooth = (np.mean(neigh) / (gstd + 1e-9)) if neigh else np.nan
    else:
        ax.text(0.5, 0.5, "위치(Cell No/Row·Col)\n컬럼 없음",
                ha="center", va="center", transform=ax.transAxes)
        smooth = np.nan
    ax.set_title(f"③ 트레이 온도 형상  ({heat_info})")

    # (4) T1, T2, T3 분포 겹침
    ax = fig.add_subplot(gs[1, 0])
    for arr, lbl, c in [(T1, "T1", "#e74c3c"), (T2, "T2", "#f39c12"),
                        (T3, "T3", "#3498db")]:
        a = arr[~np.isnan(arr)]
        ax.hist(a, bins=80, alpha=0.45, label=lbl, color=c)
    ax.set_title("④ 시점별 온도 분포 (시간 드리프트 확인)")
    ax.set_xlabel("온도 (°C)"); ax.set_ylabel("셀 수")
    ax.legend(fontsize=9)

    # (5) ΔOCV vs (T1-T3)  + 회귀기울기(=경험적 dU/dT)
    ax = fig.add_subplot(gs[1, 1])
    (xx, yy), _ = clean(dT_13, dOCV)
    ax.scatter(xx, yy, s=6, alpha=0.4, color="purple")
    slope = np.nan
    if len(xx) > 10 and np.std(xx) > 1e-6:
        slope, intc = np.polyfit(xx, yy, 1)
        xs = np.linspace(xx.min(), xx.max(), 50)
        ax.plot(xs, slope * xs + intc, "orange", lw=2,
                label=f"기울기 {slope:.3f} mV/°C")
        ax.legend(fontsize=9)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_title("⑤ ΔOCV vs (T1−T3)  (가짜신호 직접확인)")
    ax.set_xlabel("T1−T3 (°C)"); ax.set_ylabel("ΔOCV (mV)")

    # (6) 자동 해석 텍스트
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    std13 = np.nanstd(dT_13)
    pct_big = np.nanmean(np.abs(dT_13) > 1.0) * 100
    rng = np.nanmax(T_avg) - np.nanmin(T_avg)

    lines = ["[ 자동 해석 ]", ""]
    lines.append(f"• T1−T3 표준편차 : {std13:.3f} °C")
    lines.append(f"• |T1−T3|>1°C 셀 : {pct_big:.1f} %")
    lines.append(f"• T_avg 전체 범위 : {rng:.2f} °C")
    if not np.isnan(slope):
        lines.append(f"• ΔOCV~(T1−T3) 기울기: {slope:.3f} mV/°C")
    if not np.isnan(smooth):
        lines.append(f"• 트레이 공간 매끄러움: {smooth:.2f}")
        lines.append("   (<0.5 매끄러움=구배 / >0.9 무작위=핀노이즈)")
    lines.append("")
    lines.append("[ 판정 ]")
    # dU/dT 판정
    if std13 < 0.2:
        lines.append("▶ T1−T3 매우 좁음 →")
        lines.append("  dU/dT 보정 불필요 (스킵)")
    elif std13 < 0.5:
        lines.append("▶ T1−T3 작음 → dU/dT 효과 제한적")
    else:
        lines.append("▶ T1−T3 큼 → dU/dT 보정 유효")
    # 공간 판정
    if not np.isnan(smooth):
        if smooth < 0.5:
            lines.append("▶ 온도 공간 매끄러움 →")
            lines.append("  B_NNR/GP 공간보정 유효, 핀 신뢰")
        elif smooth > 0.9:
            lines.append("▶ 온도 무작위 → 온도핀 불신,")
            lines.append("  공간보정 제한적")
        else:
            lines.append("▶ 온도 약한 구배 존재")

    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=11, va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#f7f7f7", edgecolor="gray"))

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(path)),
                       "온도진단_dOCV.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n[저장] {out}")

    # 콘솔에도 요약
    print("\n── 요약 ──")
    print(f"  T1−T3 std = {std13:.3f}°C, |T1−T3|>1°C = {pct_big:.1f}%, "
          f"T_avg 범위 = {rng:.2f}°C")
    if not np.isnan(slope):
        print(f"  ΔOCV~(T1−T3) 기울기 = {slope:.3f} mV/°C")
    if not np.isnan(smooth):
        print(f"  공간 매끄러움 = {smooth:.2f}")


if __name__ == "__main__":
    path = FILE_PATH or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not path:
        print("사용법: python dOCV_temp_diagnostic.py <엑셀파일>")
        print("  또는 스크립트 상단 FILE_PATH 지정")
        sys.exit(1)
    run(path)
