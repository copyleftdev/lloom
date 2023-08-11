import streamlit as st
import pandas as pd
import openai
import os
import pystache

from lloom import Lloom

openai.api_key = os.environ.get("OPENAI_API_KEY")

lloom_app = Lloom(
    file_path="tests/sales-playbook.yml",
    perform_migration=False,
)

col1, col2 = st.columns([0.3, 0.7])

with col1: 
    st.image("tests/lloom.png")
with col2:
    st.title(lloom_app.title)
    st.write(lloom_app.description)

sales_playbook_collection = lloom_app.stores["sales_playbook_db"]
user_input = st.text_input("Query the Sales Playbook")

if user_input:
    results = sales_playbook_collection.query(
        query_texts=user_input,
        include=["documents"],
        n_results=15,
    )

    agent = lloom_app.agents["helpful_assistant"]
    template = agent["prompt"]
    
    values = {
        "context": "\n".join(results["documents"][0]),
        "query": user_input,
    }

    augemented_query = pystache.render(template, values)

    output = openai.ChatCompletion.create(
        model=agent["model"].replace("openai/", ""),
        messages=[
            {"role": "system", "content": agent["system_statement"]},
            {
                "role": "user", "content": augemented_query
            }
        ],
    )
    st.write(output["choices"][0]["message"]["content"].replace("$", ""))

