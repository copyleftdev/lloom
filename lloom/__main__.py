import os
import random
import string

import typer
from jinja2 import Template

TEMPLATE_STRING = """
import streamlit as st

from lloom import Lloom

lloom_app = Lloom(
    file_path="{{ file_path }}",
    perform_migration=True,
)

st.title(lloom_app.title)

user_input = st.text_input(lloom_app.description)

if user_input:
    output = lloom_app.run(user_input)
    st.write(output)
"""


def main(path: str):
    template = Template(TEMPLATE_STRING)

    rendered_code = template.render(file_path=path)

    random_chars = "".join(random.choice(string.ascii_letters) for i in range(10))

    file_name = f"main-{random_chars}.py"

    with open(file_name, "w") as file:
        file.write(rendered_code)

    os.system(f"streamlit run {file_name}")


if __name__ == "__main__":
    typer.run(main)
