import streamlit as st
# from src.persistence.repositories import JobRepository
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


st.set_page_config(
    page_title="Job Search",
    page_icon='🔎',
    layout="wide"
)

# repo = JobRepository(workers=0)

# job_market_stats_publish_streamlit = []

# with st.spinner("Saving Jobs"):
#     df = repo.get_df_of_jobs()

# st.header(f"From {len(df)}")

# st.dataframe(df)

# fig = plt.figure(figsize=(10, 4))
# plot = sns.countplot(data=df, x="seniority", orient='v', hue='seniority')
# st.pyplot(fig)

# fig = plt.figure(figsize=(10, 4))
# plot = sns.countplot(data=df, x="modality", orient='v', hue='modality')
# st.pyplot(fig)

# df_countries = df[['id', 'countries', 'max_salary', 'min_salary']]


# def f(countries):
#     if not isinstance(countries, str) or countries == '':
#         return ['Not Say']

#     return [
#         c.strip()
#         for countries_or_tuple in countries.split(',')
#         for c in countries_or_tuple.split(' or ')
#     ]


# df_countries['countries'] = df_countries['countries'].apply(f)
# df_countries = df_countries.explode('countries')
# df_countries.fillna("Not Say")

# df_group = df_countries.groupby('countries').agg({'id': 'count'})

# df_filtered = df_group[df_group['id'] >= 10]

# # Sum of dropped rows
# others_sum = df_group[df_group['id'] < 10]['id'].sum()

# # Append the sum of dropped rows as "Others" if there are any
# if others_sum > 0:
#     df_filtered.loc['Others'] = others_sum

# # st.dataframe(df_countries)

# fig = plt.figure(figsize=(10, 4))
# plot = sns.barplot(data=df_filtered, y="countries", x='id', hue='countries')
# st.pyplot(fig)


# # Create some sample text
# text = ', '.join(str(s) for s in df['name'].to_list())
# # Create and generate a word cloud image:
# wordcloud = WordCloud().generate(text)
# fig = plt.figure(figsize=(10, 4))
# # Display the generated image:
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# st.pyplot(fig)


# fig = plt.figure(figsize=(10, 4))
# plot = sns.boxplot(data=df, y="max_salary", hue='seniority')
# st.pyplot(fig)

# fig = plt.figure(figsize=(10, 4))
# plot = sns.boxplot(data=df, y="min_salary", hue='seniority')
# st.pyplot(fig)


# st.dataframe(
#     df[['company', 'id']]
#     .groupby('company')
#     .agg({'id': 'count'})
#     .sort_values(by='id', ascending=False),
#     use_container_width=True
# )
