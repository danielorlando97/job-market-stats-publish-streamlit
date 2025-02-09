import streamlit as st
# from src.persistence.repositories import JobRepository

# repo = JobRepository(workers=0)

st.set_page_config(
    page_title="Job Search",
    page_icon='🔎',
    layout="wide"
)

# left, right = st.columns([9, 1])

# if right.button("Load Jobs"):
#     find_job_bar = st.progress(0, text="Search Jobs")
#     total = 2
#     jobs = []
#     for index, job_list in enumerate(repo._load_jobs(total)):
#         jobs.extend(job_list)
#         find_job_bar.progress(index/total, text="Search Jobs")
#     find_job_bar.empty()

#     find_job_bar = st.progress(0, text="Search Job Details")
#     for index, job in enumerate(repo._load_job_details(jobs)):
#         jobs[index] = job
#         find_job_bar.progress(index/len(jobs), text="Search Job Details")
#     find_job_bar.empty()

#     with st.spinner("Saving Jobs"):
#         repo._save_jobs(jobs)

# job_name = left.text_input("", label_visibility='collapsed')

# if job_name:
#     with st.spinner("Reading Jobs"):

#         filters = {"name": {"$regex": job_name, "$options": "i"}}
#         jobs = repo.get_jobs(filters=filters, page=0, page_size=60)

#     if jobs:
#         for job_index in range(0, len(jobs), 3):
#             col = st.columns(3)
#             for i in range(3):
#                 try:
#                     job = jobs[job_index + i]
#                 except IndexError:
#                     break
#                 with col[i].container():
#                     col[i].markdown(f"#### [{job.name}]({job.url})")
#                     col[i].markdown(
#                         f"**Salary:** {job.min_salary} - {job.max_salary}")
#                     col[i].markdown(f"**Valid Until:** {job.published_at}")
#             st.divider()
