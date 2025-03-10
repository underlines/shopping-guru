# shopping-guru

## setup

1. install uv:

- install [uv](https://docs.astral.sh/uv/getting-started/installation/)
- in project folder execute `uv sync`

1. set secrets

- `cp .env-example .env
- add your API keys

1. activate venv and run:

either natively:

- win: `source .venv\Scripts\activate` or linux/osx: `source .venv/bin/activate`
- `streamlit run main.py`

or run directly via uv (which also uses the venv):

- `uv run -- streamlit run main.py`

## file cleanup

Playwright:

- C:\Users\username\AppData\Local\ms-playwright\winldd-1007
