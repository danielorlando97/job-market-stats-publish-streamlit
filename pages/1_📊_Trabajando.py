import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
# from src.persistence.repositories import JobRepository
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from mongo_job import get_df

import plotly.express as px
from collections import Counter


st.set_page_config(
    page_title="Trabajando",
    page_icon='📊',
    layout="wide"
)

df = get_df("Trabajando")

# fig = px.bar(
#     df,
#     x="sepal_width",
#     y="sepal_length",
# )

fig = px.histogram(df, x='experience', color="experience")
event = st.plotly_chart(fig, key="experience", on_select="rerun")

fig = px.histogram(df, x='modality', color="modality")
event = st.plotly_chart(fig, key="modality", on_select="rerun")


df_countries = df[['id', 'countries', 'max_salary', 'min_salary']]


def f(countries):
    if not isinstance(countries, str) or countries == '':
        return ['Not Say']

    return [
        c.strip()
        for countries_or_tuple in countries.split(',')
        for c in countries_or_tuple.split(' or ')
    ]


df_countries['countries'] = df_countries['countries'].apply(f)
df_countries = df_countries.explode('countries')
df_countries.fillna("Not Say", inplace=True)

# fig = px.histogram(df_countries, y='countries', color="countries")
# event = st.plotly_chart(fig, key="countries", on_select="rerun")

df_group = df_countries.groupby('countries').agg({'id': 'count'})

# Filter groups where count is >= 10
df_filtered = df_group[df_group['id'] >= 10]

# Sum of dropped rows
others_sum = df_group[df_group['id'] < 10]['id'].sum()

# Append the sum of dropped rows as "Others" if there are any
if others_sum > 0:
    df_filtered.loc['Others'] = others_sum

# Reset index to get 'countries' as a column again
df_filtered = df_filtered.reset_index()

# Plot
fig = px.bar(df_filtered, x='countries', y='id', color="countries")
event = st.plotly_chart(fig, key="countries", on_select="rerun")


# Create some sample text
text = ', '.join(str(s) for s in df['name'].to_list())
# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)
fig = plt.figure(figsize=(10, 4))
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
st.pyplot(fig)


# fig = plt.figure(figsize=(10, 4))
# plot = sns.boxplot(data=df, y="max_salary", hue='seniority')
# st.pyplot(fig)


df['experience'] = df['experience'].str.findall(r'\d+').apply(lambda x: x[0])
df_filtered = df[df['max_salary'] != 0].sort_values('experience')

fig = px.histogram(df_filtered, x='max_salary', color='experience',
                   facet_col='experience', facet_col_wrap=4)
event = st.plotly_chart(fig, key="max_salary_hist", on_select="rerun")

fig = px.box(df_filtered, y="max_salary", color='experience',
             facet_col='experience', facet_col_wrap=4)
# fig.update_yaxes(matches=None)
fig.update_xaxes(matches=None)
event = st.plotly_chart(fig, key="max_salary",
                        on_select="rerun", use_container_width=True)
