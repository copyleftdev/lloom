import os

import typer


def main(path: str):
    os.system(f"streamlit run main-sp.py")


if __name__ == "__main__":
    typer.run(main)
