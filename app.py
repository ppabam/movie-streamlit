import streamlit as st
import pyarrow.dataset as ds
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

# 페이지 제목
st.title("2024 KOBIS Daily Box Office 데이터 분석")

# 데이터 로드
@st.cache_data  # 데이터 로딩 캐싱으로 성능 최적화
def load_data():
    # path = "/home/tom/data/movie_after/dailyboxoffice"
    path = "data/movie_after_2024.parquet"
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    df = dataset.to_table().to_pandas()
     # dt를 datetime으로 변환 후 YYYY-MM-DD 문자열로 포맷
    df['dt'] = pd.to_datetime(df['dt'], format='%Y%m%d', errors='coerce')
    return df

df = load_data()

# 데이터 미리보기
st.subheader("데이터 미리보기")

# 기본 컬럼 설정 (디폴트로 선택될 컬럼)
default_columns = ['dt', 'movieCd', 'audiAcc', 'multiMovieYn', 'repNationCd', 'rank', 'movieNm']

# 전체 컬럼 리스트
all_columns = df.columns.tolist()

# 사용자가 표시할 컬럼 선택 (기본값으로 default_columns 설정)
selected_columns = st.multiselect(
    "표시할 컬럼 선택",
    options=all_columns,
    default=default_columns  # 기본적으로 dt, movieCd, audiAcc, multiMovieYn, repNationCd, rank 선택
)

# 선택된 컬럼이 없으면 빈 테이블 방지
if not selected_columns:
    st.write("컬럼을 선택해주세요.")
    display_columns = default_columns  # 기본 컬럼으로 fallback
else:
    display_columns = selected_columns
    

# 인덱스 제외하고 데이터프레임 표시
st.dataframe(df[display_columns].head(), use_container_width=True, hide_index=True)


# multiMovieYn과 repNationCd의 NaN 비율 계산
st.subheader("multiMovieYn 및 repNationCd의 NaN 비율")
multiMovieYn_null_ratio = df['multiMovieYn'].isna().mean()
repNationCd_null_ratio = df['repNationCd'].isna().mean()
st.write(f"multiMovieYn NaN 비율: {multiMovieYn_null_ratio:.2%}")
st.write(f"repNationCd NaN 비율: {repNationCd_null_ratio:.2%}")

# 일자별 NaN 비율 추이
st.subheader("일자별 NaN 비율 추이")
null_ratios = df.groupby('dt').agg(
    multiMovieYn_null_ratio=('multiMovieYn', lambda x: x.isna().mean()),
    repNationCd_null_ratio=('repNationCd', lambda x: x.isna().mean())
).reset_index()

# 그래프 그리기
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(null_ratios['dt'], null_ratios['multiMovieYn_null_ratio'], label='multiMovieYn NaN Ratio', marker='o', color='blue')
ax.plot(null_ratios['dt'], null_ratios['repNationCd_null_ratio'], label='repNationCd NaN Ratio', marker='x', color='orange')
ax.set_title('Daily NaN Ratio Trend')
ax.set_xlabel('Date')
ax.set_ylabel('NaN Ratio')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig)

# 매출액과 관객 수 상위 10개 영화 (Altair)
st.subheader("매출액 상위 10개 영화")
top_sales = df.groupby('movieNm')['salesAmt'].sum().sort_values(ascending=False).head(10).reset_index()

chart_sales = alt.Chart(top_sales).mark_bar().encode(
    x=alt.X('salesAmt', title='총 매출액'),
    y=alt.Y('movieNm', sort='-x', title='영화'), # -x로 내림차순 정렬
    tooltip=['movieNm', 'salesAmt']
).properties(
    title='매출액 상위 10개 영화'
)
st.altair_chart(chart_sales, use_container_width=True)


st.subheader("관객 수 상위 10개 영화")
top_audience = df.groupby('movieNm')['audiCnt'].sum().sort_values(ascending=False).head(10).reset_index()

chart_audi = alt.Chart(top_audience).mark_bar().encode(
    x=alt.X('audiCnt', title='관객 수'),
    y=alt.Y('movieNm', sort='-x', title='영화'), # -x로 내림차순 정렬
    tooltip=['audiCnt']
).properties(
    title='관객 수 상위 10개 영화'
)
st.altair_chart(chart_audi, use_container_width=True)

# 날짜별 데이터 필터링 및 그래프
st.subheader("날짜별 multiMovieYn 및 repNationCd 기준 관객 수 분석")

# 날짜 범위 선택
date_range = st.date_input("날짜 범위 선택", [df['dt'].min(), df['dt'].max()])

if len(date_range) == 2:
    start_date, end_date = date_range

    # 날짜 범위 필터링
    filtered_df = df[(df['dt'] >= pd.to_datetime(start_date)) & (df['dt'] <= pd.to_datetime(end_date))]

    # 데이터 변환: 그룹화
    audience_data = filtered_df.groupby(['dt', 'multiMovieYn', 'repNationCd'])['audiCnt'].sum().reset_index()

    # multiMovieYn 기준 선 그래프
    chart_multi = alt.Chart(audience_data).mark_line(point=True).encode(
        x=alt.X('dt:T', title='날짜'),
        y=alt.Y('audiCnt:Q', title='관객 수'),
        color=alt.Color('multiMovieYn:N', scale=alt.Scale(domain=['Y', 'N'], range=['blue', 'orange']),
                        title='멀티영화 여부'),
        tooltip=['dt:T', 'multiMovieYn:N', 'audiCnt:Q']
    ).properties(
        title="멀티영화 여부 (multiMovieYn) 기준 관객 수 변화"
    )

    # repNationCd 기준 선 그래프
    chart_repNation = alt.Chart(audience_data).mark_line(point=True).encode(
        x=alt.X('dt:T', title='날짜'),
        y=alt.Y('audiCnt:Q', title='관객 수'),
        color=alt.Color('repNationCd:N', title='국적 코드'),
        tooltip=['dt:T', 'repNationCd:N', 'audiCnt:Q']
    ).properties(
        title="국적 코드 (repNationCd) 기준 관객 수 변화"
    )

    # 그래프 출력
    st.altair_chart(chart_multi, use_container_width=True)
    st.altair_chart(chart_repNation, use_container_width=True)


# 추가: 영화별 상세 검색
st.subheader("영화별 상세 검색")
movie_name = st.selectbox("영화 선택", df['movieNm'].unique())

if movie_name:
    # 선택한 영화 데이터 필터링
    movie_data = df[df['movieNm'] == movie_name][['dt', 'audiCnt', 'salesAmt']].copy()

    # Altair 차트 생성
    base = alt.Chart(movie_data).encode(
        x=alt.X('dt:T', title='날짜')
    )

    line_audi = base.mark_line(color='blue').encode(
        y=alt.Y('audiCnt:Q', title='관객 수'),
        tooltip=['dt', 'audiCnt']
    )

    line_sales = base.mark_line(color='red').encode(
        y=alt.Y('salesAmt:Q', title='매출액'),
        tooltip=['dt', 'salesAmt']
    )

    # 두 개의 그래프를 겹쳐서 표시
    chart = alt.layer(line_audi, line_sales).resolve_scale(y='independent').properties(
        title=f"'{movie_name}' 관객 수 및 매출액 추이"
    )

    st.altair_chart(chart, use_container_width=True)