import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import pandas as pd

from wordcloud import WordCloud
from mongo_job import get_df
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from collections import defaultdict

from utils.set_tools import compute_distance_matrix, dice_distance
from utils.graphs import subgraph, subgraph_df
from utils.submit_task import submit_task, wait_task_result
from utils.clustering import *

executor = ProcessPoolExecutor()

st.set_page_config(
    page_title="GetOnBoard",
    page_icon='📊',
    layout="wide"
)

DATA = get_df("GetOnBoard")
df = DATA
# Compute Sets
df_set = df[~df['name_tokens'].isna()]['name_tokens'].apply(frozenset)

# submit_task(
#     'GetOnBoard_set_matrix',
#     compute_distance_matrix,
#     df_set.tolist(),
#     dice_distance
# )


df['seniority_num'] = df['seniority'].map({"Expert": 5, 'Senior': 4, "Semi Senior": 3,
                                           "Junior": 2, "No experience required": 1})

df = df.sort_values(by='seniority_num')

st.header("Seniority Dist")

fig = px.histogram(df, x='seniority', color="seniority")
event = st.plotly_chart(fig, key="seniority", on_select="rerun")

st.divider()

st.header("Modality Dist")

fig = px.histogram(df, x='modality', color="modality")
event = st.plotly_chart(fig, key="modality", on_select="rerun")

st.divider()

st.header("Country Dist")

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

st.divider()


st.header("Salary vs Seniority Dist")

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
fig = px.histogram(df, x='max_salary', color='seniority',
                   facet_col='seniority', facet_col_wrap=3)
event = st.plotly_chart(fig, key="max_salary_hist", on_select="rerun")

fig = px.box(df, x='seniority', y="max_salary", color='seniority')
event = st.plotly_chart(fig, key="max_salary", on_select="rerun")

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

st.header("Titles Word Graph")

counter = defaultdict(int)
for title in df_set.to_list():
    for word in title:
        counter[word] += 1

G = nx.Graph()
for title in df_set.to_list():
    for word1, word2 in combinations(title, 2):  # Connect co-occurring words
        if G.has_edge(word1, word2):
            G[word1][word2]["weight"] += 1
        else:
            G.add_edge(word1, word2, weight=1)

start_word = st.selectbox("Choose Title Word", G.nodes)
# Build the graph

if start_word:

    data = subgraph_df(G, start_word=start_word, depth=1)

    st.markdown(f"### `{start_word}` appear in {counter[start_word]} titles")

    fig = go.Figure(go.Treemap(
        ids=data.id,
        labels=data.label,
        parents=data.parent,
        values=data['count'],
        sort=True,
    ))

    fig.update_layout(
        uniformtext=dict(minsize=15, mode='hide'),
        margin=dict(t=0, l=5, r=5, b=5)
    )

    event = st.plotly_chart(fig, key="word_graph", on_select="rerun")

    st.divider()

if "GetOnBoard_data" not in st.session_state:
    st.session_state.GetOnBoard_data = None

if st.session_state.GetOnBoard_data is None:
    with st.spinner("Compute Distances"):

        df = DATA[~DATA['name_tokens'].isna()]
        matrix = compute_distance_matrix(df_set.tolist(), dice_distance)
        # matrix = wait_task_result("GetOnBoard_set_matrix")
        NODES = [
            Node(index, tokens=TokenSet(row['name_tokens']), text=row['name'])
            for index, row in df.iterrows()
        ]

        cluster = build_cluster(matrix)
        root = build_agglomerative_tree(cluster, NODES)
        root = count_visit(root)
        root = compute_tokens_visit(root)
        root = balance_visit(root)
        root = count_visit(root)
        st.session_state.GetOnBoard_data = \
            pd.DataFrame(build_list(root, whole_tree=True))

data = st.session_state.GetOnBoard_data
fig = go.Figure(go.Treemap(
    ids=data.id,
    labels=data.text,
    parents=data.parent,
    values=data.children_count,
    branchvalues="total",
    sort=True,
    hovertext=data.words,
    marker_colorscale='Blues',
    # color='day'
    # hoverinfo="text",
    # root_color="Blues"
))

fig.update_layout(
    uniformtext=dict(minsize=10, mode='hide'),
    margin=dict(t=0, l=0, r=0, b=0)
)

event = st.plotly_chart(fig, key="cluster", on_select="rerun")
