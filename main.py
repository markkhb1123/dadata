# app.py
import streamlit as st
import pandas as pd
import math

st.set_page_config(page_title="봄·가을은 짧아지고 있는가?", layout="wide")

FILE_NAME = "ta_20260601093156.csv"

st.title("🍂 봄·가을은 정말 짧아지고 있는가?")
st.caption("일평균기온 9일 이동평균 기반 계절 정의로 분석하는 통계 탐구 보고서")


# ---------------------------------------------------------------
# 1. 데이터 불러오기
# ---------------------------------------------------------------
@st.cache_data
def load_data(path):
    # 컬럼명: 날짜, 지점, 평균기온(℃), 최저기온(℃), 최고기온(℃)
    df = pd.read_csv(path)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    # 날짜 앞에 탭/공백이 섞여 있으므로 정리
    date_col = df.columns[0]
    df[date_col] = df[date_col].astype(str).str.strip()
    df["날짜"] = pd.to_datetime(df[date_col], errors="coerce")

    # 평균기온 컬럼 찾기
    avg_col = [c for c in df.columns if "평균" in c][0]
    df["평균기온"] = pd.to_numeric(df[avg_col], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"]).sort_values("날짜").reset_index(drop=True)
    df["연도"] = df["날짜"].dt.year
    return df


# ---------------------------------------------------------------
# 2. 계절 길이 계산
# ---------------------------------------------------------------
def find_season_lengths(df):
    """
    기상학적 계절 정의:
      - 9일 이동평균 기온 사용
      - 봄 시작 : 5℃ 이상 처음 (상반기)
      - 여름 시작 : 20℃ 이상 처음
      - 가을 시작 : 20℃ 미만 처음 (하반기)
      - 겨울 시작 : 5℃ 미만 처음 (하반기)
    """
    results = []
    for year, g in df.groupby("연도"):
        g = g.sort_values("날짜").reset_index(drop=True)
        # 1년치 데이터가 충분치 않으면 제외
        if len(g) < 300:
            continue

        ma = g["평균기온"].rolling(window=9, center=True, min_periods=5).mean()
        g = g.assign(ma=ma).dropna(subset=["ma"]).reset_index(drop=True)

        mid = pd.Timestamp(year=year, month=7, day=1)

        def first_day(cond, after=None, before=None):
            sub = g
            if after is not None:
                sub = sub[sub["날짜"] >= after]
            if before is not None:
                sub = sub[sub["날짜"] <= before]
            hit = sub[cond(sub["ma"])]
            if len(hit) == 0:
                return None
            return hit.iloc[0]["날짜"]

        # 상반기: 기온이 올라가는 구간
        spring_start = first_day(lambda x: x >= 5, before=mid)
        summer_start = first_day(lambda x: x >= 20, before=mid)
        # 하반기: 기온이 내려가는 구간
        autumn_start = first_day(lambda x: x < 20, after=mid)
        winter_start = first_day(lambda x: x < 5, after=mid)

        spring_len = None
        autumn_len = None
        if spring_start is not None and summer_start is not None:
            spring_len = (summer_start - spring_start).days
        if autumn_start is not None and winter_start is not None:
            autumn_len = (winter_start - autumn_start).days

        results.append({
            "연도": year,
            "봄시작": spring_start,
            "여름시작": summer_start,
            "가을시작": autumn_start,
            "겨울시작": winter_start,
            "봄길이": spring_len if (spring_len and spring_len > 0) else None,
            "가을길이": autumn_len if (autumn_len and autumn_len > 0) else None,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------
# 3. 선형 회귀 (외부 라이브러리 없이 직접 구현)
# ---------------------------------------------------------------
def linear_regression(xs, ys):
    """최소제곱법으로 기울기, 절편, 상관계수 r, p값(근사) 계산"""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None

    slope = sxy / sxx
    intercept = my - slope * mx
    r = sxy / math.sqrt(sxx * syy)

    # t 통계량과 p값(양측, 정규근사)
    if abs(r) >= 1:
        p = 0.0
    else:
        t = r * math.sqrt((n - 2) / (1 - r ** 2))
        # 큰 표본 정규근사로 p값 추정
        z = abs(t)
        p = 2 * (1 - normal_cdf(z))
    return {
        "slope": slope, "intercept": intercept,
        "r": r, "r2": r ** 2, "p": p, "n": n
    }


def normal_cdf(z):
    """표준정규분포 누적분포함수 (오차함수 근사)"""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ---------------------------------------------------------------
# 메인
# ---------------------------------------------------------------
try:
    df = load_data(FILE_NAME)
except FileNotFoundError:
    st.error(f"'{FILE_NAME}' 파일을 app.py와 같은 폴더에 두세요.")
    st.stop()

st.success(f"데이터 로드 완료: {len(df):,}행 / "
           f"{df['연도'].min()}년 ~ {df['연도'].max()}년")

seasons = find_season_lengths(df)

# 분석 기간 선택
min_y, max_y = int(seasons["연도"].min()), int(seasons["연도"].max())
yr_range = st.slider("분석 기간 선택", min_y, max_y, (min_y, max_y))
mask = (seasons["연도"] >= yr_range[0]) & (seasons["연도"] <= yr_range[1])
data = seasons[mask].copy()

st.divider()

# ---------------------------------------------------------------
# 봄/가을 길이 그래프 + 회귀
# ---------------------------------------------------------------
for season in ["봄길이", "가을길이"]:
    st.subheader(f"📊 {season[:1]}의 길이 변화")

    sub = data.dropna(subset=[season])[["연도", season]].copy()
    if len(sub) < 3:
        st.warning("분석할 데이터가 부족합니다.")
        continue

    xs = sub["연도"].tolist()
    ys = sub[season].tolist()
    reg = linear_regression(xs, ys)

    # 추세선 값 추가
    sub["추세선"] = [reg["slope"] * x + reg["intercept"] for x in xs]
    chart_df = sub.set_index("연도")[[season, "추세선"]]
    st.line_chart(chart_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("연간 변화량", f"{reg['slope']:.3f} 일/년")
    c2.metric("100년 환산", f"{reg['slope']*100:.1f} 일")
    c3.metric("결정계수 R²", f"{reg['r2']:.3f}")
    c4.metric("p값", f"{reg['p']:.4f}")

    # 통계적 해석
    direction = "짧아지고" if reg["slope"] < 0 else "길어지고"
    significant = reg["p"] < 0.05
    if significant:
        st.info(
            f"➡️ {season[:1]}은 매년 약 **{abs(reg['slope']):.3f}일씩 {direction}** 있습니다. "
            f"p값이 {reg['p']:.4f} < 0.05 이므로 **통계적으로 유의**합니다."
        )
    else:
        st.warning(
            f"➡️ 추세는 {direction} 있으나, p값 {reg['p']:.4f} ≥ 0.05 로 "
            f"**통계적으로 유의하다고 보기 어렵습니다.**"
        )
    st.divider()

# ---------------------------------------------------------------
# 원자료 보기
# ---------------------------------------------------------------
with st.expander("📋 연도별 계절 길이 데이터 보기"):
    st.dataframe(
        data[["연도", "봄시작", "여름시작", "가을시작", "겨울시작",
              "봄길이", "가을길이"]],
        use_container_width=True
    )

st.caption("계절 정의: 9일 이동평균 일평균기온 기준 / 회귀: 최소제곱법, p값은 정규근사")
