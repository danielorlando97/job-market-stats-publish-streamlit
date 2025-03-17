import streamlit as st
# from src.persistence.repositories import JobRepository
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from mongo_job import get_df

import plotly.express as px


st.set_page_config(
    page_title="Trabajando",
    page_icon='📊',
    layout="wide"
)

df = get_df("Trabajando")

df["extract_num"] = df["experience"].str.extract(r"(\d+)")
df["extract_num"] = df["extract_num"].apply(int)
df = df.sort_values(by='extract_num')
df["experience"] = df["extract_num"].apply(str) + ' years'
# fig = px.bar(
#     df,
#     x="sepal_width",
#     y="sepal_length",
# )

st.header("Experience Years Dist")

fig = px.histogram(df, x='experience', color="experience")
event = st.plotly_chart(fig, key="experience", on_select="rerun")

st.divider()

st.header("Modality Dist")

fig = px.histogram(df, x='modality', color="modality")
event = st.plotly_chart(fig, key="modality", on_select="rerun")

st.divider()

st.header("Country Dist")
st.info("Trabajando is job market website witch have only offer from Chile")

st.divider()

st.header("Titles WordCloud")

# Create some sample text
text = ', '.join(str(s) for s in df['name'].to_list())
# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)
fig = plt.figure(figsize=(10, 4))
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
st.pyplot(fig)

st.divider()

st.header("Salary vs Seniority Dist")

df = df[~df['max_salary'].isna()]

max_salary = df['max_salary'].max()
min_salary = df['max_salary'].min()

min_salary, max_salary = st.slider(
    "How old are you?",
    min_salary,
    max_salary,
    (min_salary, max_salary),
    label_visibility='hidden'
)

df = df[(df['max_salary'] >= min_salary) & (df['max_salary'] <= max_salary)]

df['experience'] = df['experience'].str.findall(r'\d+').apply(lambda x: x[0])
df_filtered = df[df['max_salary'] != 0].sort_values('experience')

fig = px.histogram(df_filtered, x='max_salary', color='experience',
                   facet_col='experience', facet_col_wrap=3)
event = st.plotly_chart(fig, key="max_salary_hist", on_select="rerun")

fig = px.box(df_filtered, y="max_salary", color='experience',
             facet_col='experience', facet_col_wrap=3)

# fig.update_yaxes(matches=None)
fig.update_xaxes(matches=None)
event = st.plotly_chart(fig, key="max_salary",
                        on_select="rerun", use_container_width=True)
