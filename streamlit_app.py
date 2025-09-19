# streamlit_app.py
# -*- coding: utf-8 -*-
# =========================================================
# 청소년 정서(불안·우울 등) × 기후(글로벌 온도) 대시보드 — Kaggle Only
#
# - 학생 데이터: Kaggle "Student Mental Health & Resilience Dataset"
#   https://www.kaggle.com/datasets/ziya07/student-mental-health-and-resilience-dataset
#   (예상 컬럼)
#   Student_ID,Age,Gender,GPA,Stress_Level,Anxiety_Score,Depression_Score,
#   Daily_Reflections,Sleep_Hours,Steps_Per_Day,Mood_Description,
#   Sentiment_Score,Mental_Health_Status
#
# - 기후 데이터: Kaggle "Climate Change: Earth Surface Temperature Data" (Berkeley Earth)
#   https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data
#   사용 파일: GlobalTemperatures.csv (전 세계 월별/연도별 평균 기온)
#
# ※ 두 데이터는 '국가/연도' 공통 키가 없어 직접 결합하지 않고,
#    좌: 학생 데이터 내부 상관/분포, 우: 같은 시기 글로벌 기온 추세를 병렬 시각화합니다.
#    Kaggle API 인증 필요(Secrets에 [kaggle] username/key). 레포에는 secrets.toml을 커밋하지 마세요.
# =========================================================

import os
import io
import json
import datetime as dt
from base64 import b64encode

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# ----------------- 페이지/폰트 -----------------
st.set_page_config(page_title="청소년 정서 × 기후 대시보드 (Kaggle)", layout="wide")

def inject_font_css():
    """ /fonts/Pretendard-Bold.ttf 존재 시 UI 전역에 적용 """
    font_path = "/fonts/Pretendard-Bold.ttf"
    if os.path.exists(font_path):
        with open(font_path, "rb") as f:
            font_data = b64encode(f.read()).decode("utf-8")
        st.markdown(
            f"""
            <style>
            @font-face {{
              font-family: 'Pretendard';
              src: url(data:font/ttf;base64,{font_data}) format('truetype');
              font-weight: 700; font-style: normal; font-display: swap;
            }}
            html, body, [class*="css"] {{
              font-family: 'Pretendard', system-ui, -apple-system, Segoe UI, Roboto, Arial, 'Noto Sans KR', sans-serif !important;
            }}
            .plotly, .js-plotly-plot * {{ font-family: 'Pretendard', sans-serif !important; }}
            </style>
            """,
            unsafe_allow_html=True,
        )
inject_font_css()

TODAY = dt.date.today()
THIS_YEAR = TODAY.year

st.title("🌿 청소년 정서(불안·우울) × 지구 기온 변화 (Kaggle)")
st.caption("좌: 학생 정신건강 수치 데이터(상관·분포) / 우: 글로벌 평균기온 추세(연도별). 오늘 이후 데이터는 표시하지 않습니다.")

# ----------------- Kaggle 인증 헬퍼 -----------------
def _have_kaggle_env() -> bool:
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))

def _ensure_kaggle_from_secrets():
    """Streamlit Secrets → env + ~/.kaggle/kaggle.json 생성"""
    try:
        u = st.secrets["kaggle"]["username"]
        k = st.secrets["kaggle"]["key"]
        os.environ["KAGGLE_USERNAME"] = u
        os.environ["KAGGLE_KEY"] = k
        kag_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kag_dir, exist_ok=True)
        kag_path = os.path.join(kag_dir, "kaggle.json")
        with open(kag_path, "w") as f:
            json.dump({"username": u, "key": k}, f)
        os.chmod(kag_path, 0o600)
    except Exception:
        pass

def _auth_kaggle():
    """Kaggle API 인증 객체 반환"""
    _ensure_kaggle_from_secrets()
    if not _have_kaggle_env():
        raise RuntimeError("Kaggle 인증 정보가 없습니다. (배포 설정의 Secrets에 [kaggle] username/key 입력 필요)")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # lazy import
    except Exception as e:
        raise RuntimeError("kaggle 패키지가 필요합니다. requirements.txt에 kaggle 추가") from e
    api = KaggleApi()
    api.authenticate()
    return api

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

# ----------------- 데이터 다운로드 (캐시) -----------------
@st.cache_data(ttl=24*3600, show_spinner=True)
def _download_kaggle_artifacts() -> dict:
    """
    Kaggle에서 필요한 파일들을 내려받아 data/raw 에 저장.
    반환: {"raw_dir": <path>}
    """
    api = _auth_kaggle()
    raw_dir = _ensure_dir("data/raw")

    # 학생 정신건강 데이터 (압축 안에 여러 CSV가 있을 수 있음)
    api.dataset_download_files(
        "ziya07/student-mental-health-and-resilience-dataset",
        path=raw_dir, unzip=True
    )

    # 글로벌 온도 데이터
    api.dataset_download_files(
        "berkeleyearth/climate-change-earth-surface-temperature-data",
        path=raw_dir, unzip=True
    )

    return {"raw_dir": raw_dir}

# ----------------- 학생 CSV 선택 & 컬럼 표준화 -----------------
def _normalize_cols(cols):
    # 공백/하이픈/대소문자 차이를 흡수하기 위한 표준 키 변환
    out = []
    for c in cols:
        key = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(c).strip())
        key = key.lower().replace("__", "_")
        while "__" in key:
            key = key.replace("__", "_")
        out.append(key.strip("_"))
    return out

EXPECTED_NUMERIC = [
    "stress_level", "anxiety_score", "depression_score", "sleep_hours",
    "steps_per_day", "gpa", "age", "sentiment_score"
]
EXPECTED_ANY = set(EXPECTED_NUMERIC + [
    "gender", "mental_health_status", "student_id", "daily_reflections", "mood_description"
])

def _find_best_student_csv(raw_dir: str) -> str:
    """압축 해제된 CSV들 중 예상 컬럼과 겹치는 수가 가장 많은 파일 선택"""
    cands = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.lower().endswith(".csv")]
    best_path, best_score = None, -1
    for p in cands:
        try:
            df_head = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        norm = _normalize_cols(df_head.columns)
        score = len(set(norm) & EXPECTED_ANY)
        if score > best_score:
            best_path, best_score = p, score
    if not best_path:
        raise FileNotFoundError("학생 데이터 CSV를 찾지 못했습니다.")
    return best_path

@st.cache_data(ttl=24*3600, show_spinner=False)
def load_student_df(paths: dict) -> pd.DataFrame:
    raw_dir = paths["raw_dir"]
    stu_csv = _find_best_student_csv(raw_dir)
    df = pd.read_csv(stu_csv)

    # 컬럼 표준화(리네이밍)
    original_cols = list(df.columns)
    norm_map = {}
    for c in original_cols:
        key = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(c).strip())
        key = key.lower().replace("__", "_")
        while "__" in key:
            key = key.replace("__", "_")
        key = key.strip("_")
        norm_map[c] = key
    df = df.rename(columns=norm_map)

    # 동의어(알리아스) 흡수
    alias = {
        "stresslevel": "stress_level",
        "anxietyscore": "anxiety_score",
        "depressionscore": "depression_score",
        "sleephours": "sleep_hours",
        "steps": "steps_per_day",
        "stepsperday": "steps_per_day",
        "sentimentscore": "sentiment_score",
        "mentalhealthstatus": "mental_health_status",
        "studentid": "student_id",
    }
    for src, dst in alias.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    # 수치형 변환
    for col in ["age", "gpa", "stress_level", "anxiety_score", "depression_score",
                "sleep_hours", "steps_per_day", "sentiment_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 범주형 정리
    for col in ["gender", "mental_health_status"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df

# ----------------- 글로벌 온도 로드 -----------------
@st.cache_data(ttl=24*3600, show_spinner=False)
def load_global_temp(paths: dict) -> pd.DataFrame:
    """
    GlobalTemperatures.csv (Berkeley Earth)
    주요 컬럼: dt(날짜), LandAverageTemperature(°C), ...
    월별 → 연평균 산출, 미래(오늘 이후 연도)는 제거
    """
    raw_dir = paths["raw_dir"]
    target = None
    # 파일명 탐색: GlobalTemperatures.csv
    for f in os.listdir(raw_dir):
        if f.lower() == "globaltemperatures.csv":
            target = os.path.join(raw_dir, f)
            break
    if target is None:
        # 혹시 다른 폴더 구조로 풀렸을 경우를 위한 탐색
        cands = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.lower().endswith(".csv")]
        for p in cands:
            if "globaltemperatures" in os.path.basename(p).lower():
                target = p
                break
    if target is None:
        raise FileNotFoundError("GlobalTemperatures.csv 를 찾지 못했습니다.")

    gt = pd.read_csv(target, parse_dates=["dt"])
    # 대표로 LandAverageTemperature 사용 (월별)
    value_col = "LandAverageTemperature"
    if value_col not in gt.columns:
        # 두 번째 컬럼을 대체 사용 (보수적)
        value_col = gt.columns[1]

    gt = gt[["dt", value_col]].dropna()
    gt["year"] = gt["dt"].dt.year
    gt = gt[gt["year"] <= THIS_YEAR]
    annual = gt.groupby("year", as_index=False)[value_col].mean().rename(columns={value_col: "global_temp_C"})
    return annual.sort_values("year")

def fit_ols(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    slope, intercept, r, p, se = stats.linregress(x, y)
    return slope, intercept, r, p

# ----------------- 데이터 로드 -----------------
try:
    paths = _download_kaggle_artifacts()
except Exception as e:
    st.error("Kaggle API 처리 중 오류가 발생했습니다. (인증/네트워크/데이터셋 접근 확인)")
    st.exception(e)
    st.stop()

try:
    stu = load_student_df(paths)
except Exception as e:
    st.error("학생 데이터 로딩 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

try:
    glb = load_global_temp(paths)
except Exception as e:
    st.error("글로벌 온도 데이터 로딩 중 오류가 발생했습니다.")
    st.exception(e)
    glb = pd.DataFrame(columns=["year", "global_temp_C"])

# ----------------- 사이드바 (실존 컬럼만 노출) -----------------
st.sidebar.header("⚙️ 보기 설정")
NUMERIC_CANDIDATES = ["stress_level", "anxiety_score", "depression_score", "sleep_hours",
                      "steps_per_day", "gpa", "age", "sentiment_score"]
available = [c for c in NUMERIC_CANDIDATES if c in stu.columns]

if not available:
    st.error(f"수치형 컬럼을 찾지 못했습니다. 실제 컬럼들: {list(stu.columns)}")
    st.stop()

selected_metrics = st.sidebar.multiselect(
    "내부 상관/분포에 사용할 수치 변수",
    options=available,
    default=available[: min(5, len(available))]
)

# ----------------- 레이아웃 -----------------
left, right = st.columns([1.1, 0.9], gap="large")

# === Left: 학생 데이터 분석 ===
with left:
    st.subheader("① 학생 정신건강 데이터 분석 (수치 상관/분포)")

    if not selected_metrics:
        st.info("좌측 사이드바에서 분석할 수치 변수를 선택하세요.")
    else:
        df_sel = stu[selected_metrics].dropna()
        st.write(f"표본 크기: **{len(df_sel)}**")

        # 1) 상관 히트맵
        if len(df_sel) >= 2:
            corr = df_sel.corr(numeric_only=True).round(2)
            fig_corr = px.imshow(
                corr, text_auto=True, aspect="auto",
                title="수치 변수 상관 히트맵",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=420)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("상관분석을 수행하기에 표본이 부족합니다.")

        # 2) 대표 산점도 + OLS (예: 수면 vs 불안 / 걸음수 vs 우울)
        c1, c2 = st.columns(2)

        with c1:
            xcol, ycol = None, None
            if "sleep_hours" in stu.columns and "anxiety_score" in stu.columns:
                xcol, ycol = "sleep_hours", "anxiety_score"
            elif len(selected_metrics) >= 2:
                xcol, ycol = selected_metrics[0], selected_metrics[1]
            if xcol and ycol:
                d = stu[[xcol, ycol]].dropna()
                if len(d) >= 3:
                    s, b, r, p = fit_ols(d[xcol], d[ycol])
                    xg = np.linspace(d[xcol].min(), d[xcol].max(), 200)
                    yhat = s * xg + b
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=d[xcol], y=d[ycol], mode="markers", name="관측치"))
                    fig.add_trace(go.Scatter(x=xg, y=yhat, mode="lines", name="OLS"))
                    fig.update_layout(
                        title=f"{xcol} vs {ycol} — r={r:.3f}, p={p:.3g}, slope={s:.3f}",
                        xaxis_title=xcol, yaxis_title=ycol, height=360
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("산점도를 그리기 충분한 표본이 없습니다.")

        with c2:
            xcol, ycol = None, None
            if "steps_per_day" in stu.columns and "depression_score" in stu.columns:
                xcol, ycol = "steps_per_day", "depression_score"
            elif len(selected_metrics) >= 2:
                xcol, ycol = selected_metrics[-1], selected_metrics[0]
            if xcol and ycol:
                d = stu[[xcol, ycol]].dropna()
                if len(d) >= 3:
                    s, b, r, p = fit_ols(d[xcol], d[ycol])
                    xg = np.linspace(d[xcol].min(), d[xcol].max(), 200)
                    yhat = s * xg + b
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=d[xcol], y=d[ycol], mode="markers", name="관측치"))
                    fig.add_trace(go.Scatter(x=xg, y=yhat, mode="lines", name="OLS"))
                    fig.update_layout(
                        title=f"{xcol} vs {ycol} — r={r:.3f}, p={p:.3g}, slope={s:.3f}",
                        xaxis_title=xcol, yaxis_title=ycol, height=360
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("산점도를 그리기 충분한 표본이 없습니다.")

        # 3) 성별/연령대 집단 비교(존재할 때)
        st.markdown("#### 집단 비교")
        if "gender" in stu.columns and "depression_score" in stu.columns:
            fig_box = px.box(
                stu.dropna(subset=["gender", "depression_score"]),
                x="gender", y="depression_score",
                title="성별별 우울 점수 분포",
                points="outliers"
            )
            fig_box.update_layout(height=360)
            st.plotly_chart(fig_box, use_container_width=True)

        if "age" in stu.columns and "anxiety_score" in stu.columns:
            df_age = stu[["age", "anxiety_score"]].dropna().copy()
            # 간단한 연령대 bin
            try:
                df_age["age_bin"] = pd.cut(
                    df_age["age"],
                    bins=[0, 15, 18, 22, 30, 200],
                    labels=["≤15", "16–18", "19–22", "23–30", "31+"]
                )
                fig_bar = px.box(df_age, x="age_bin", y="anxiety_score", title="연령대별 불안 점수 분포")
                fig_bar.update_layout(height=360)
                st.plotly_chart(fig_bar, use_container_width=True)
            except Exception:
                pass

        # 전처리된 표 다운로드 (선택 변수만)
        st.markdown("##### 전처리된 표 내려받기")
        export_df = stu[selected_metrics].dropna()
        st.download_button(
            "CSV 다운로드",
            export_df.to_csv(index=False).encode("utf-8"),
            file_name="student_metrics_processed.csv",
            mime="text/csv"
        )

# === Right: 글로벌 기온 추세(병렬 비교) ===
with right:
    st.subheader("② 글로벌 평균 기온 추세 (Berkeley Earth, 연평균)")
    if glb.empty:
        st.warning("글로벌 온도 데이터가 비어 있습니다.")
    else:
        y_min, y_max = int(glb["year"].min()), int(glb["year"].max())
        default_start = max(y_min, y_max - 40)  # 최근 40년 기본
        y1, y2 = st.slider("연도 범위(글로벌 기온)", y_min, y_max, (default_start, y_max), key="glb_years")
        gm = glb[(glb["year"] >= y1) & (glb["year"] <= y2)]

        fig_g = px.line(
            gm, x="year", y="global_temp_C", markers=True,
            labels={"year": "연도", "global_temp_C": "글로벌 평균기온(°C)"},
            title="전 세계 연평균 기온(°C)"
        )
        fig_g.update_layout(height=420)
        st.plotly_chart(fig_g, use_container_width=True)
        st.caption("출처: Kaggle · Berkeley Earth — GlobalTemperatures.csv (월평균 → 연평균)")

st.markdown("---")
st.caption("주의: 본 앱은 학생 설문(단면) 데이터와 글로벌 기온(연도별)을 병렬 비교합니다. 직접적 인과/상관을 의미하지 않으며, 시점·지역이 일치하는 패널 데이터가 필요합니다.")
