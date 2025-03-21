import streamlit as st
import concurrent.futures
import time


def submit_task(task_id, func, *args, **kwargs):

    if task_id in st.session_state:
        return

    st.session_state[task_id] = None
    st.session_state[f"{task_id}-result"] = None

    with concurrent.futures.ProcessPoolExecutor() as executor:
        st.session_state[task_id] = executor.submit(func, *args, **kwargs)


def get_task_result(task_id):
    if (
        st.session_state[f"{task_id}-result"] is None
        and st.session_state[task_id].done()
    ):
        st.session_state[f"{task_id}-result"] = st.session_state[task_id].result()

    return st.session_state[f"{task_id}-result"]


def wait_task_result(task_id):
    while get_task_result(task_id) is None:
        time.sleep(5)

    return get_task_result(task_id)
