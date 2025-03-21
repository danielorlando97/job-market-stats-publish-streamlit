import streamlit as st
# from src.persistence.repositories import JobRepository
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from mongo_job import get_df
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from utils.set_tools import compute_distance_matrix, dice_distance
from utils.graphs import subgraph, subgraph_df
from utils.clustering import *
from itertools import combinations
from collections import defaultdict
from utils.submit_task import submit_task, wait_task_result

st.set_page_config(
    page_title="Trabajando",
    page_icon='📊',
    layout="wide"
)

DATA = get_df("Trabajando")

df = DATA

# Compute Sets
df_set = df[~df['name_tokens'].isna()]['name_tokens'].apply(frozenset)

# submit_task(
#     'Trabajando_set_matrix',
#     compute_distance_matrix,
#     df_set.tolist(),
#     dice_distance
# )

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

if "Trabajando_data" not in st.session_state:
    st.session_state.Trabajando_data = None

if st.session_state.Trabajando_data is None:
    with st.spinner("Compute Distances"):

        df = DATA[~DATA['name_tokens'].isna()]
        matrix = compute_distance_matrix(df_set.tolist(), dice_distance)
        # matrix = wait_task_result("Trabajando_set_matrix")
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
        st.session_state.Trabajando_data = pd.DataFrame(
            build_list(root, whole_tree=True))


data = st.session_state.Trabajando_data
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
