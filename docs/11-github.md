# Chapter 11: Publish on GitHub

**Goal:** put the project on GitHub safely, with automatic tests and a clean README.

## 11.1 Never commit secrets

Your `.env` may contain an API key. Anyone who finds a key on GitHub can spend your money, and bots scan
GitHub for keys within minutes of a push. The `.gitignore` file tells Git which files to leave out:

````gitignore
# Secrets: never commit your API keys
.env

# Virtual environments
venv/
.venv/
env/

# Python cache
__pycache__/
*.py[cod]
.pytest_cache/

# Demo data the app writes (recreated automatically from data/orders_seed.csv)
data/orders.csv
data/tickets.csv

# Gradio / audio files
.gradio/
flagged/
*.wav
*.mp3
*.m4a
*.webm

# Editors and OS
.vscode/
.idea/
.DS_Store
Thumbs.db
````

`data/orders.csv` and `data/tickets.csv` are ignored because the app changes them while running. The clean
original, `data/orders_seed.csv`, is committed.

## 11.2 Create the repository on GitHub

1. Sign in at [github.com](https://github.com) and click **New repository** (the **+** at the top right).
2. Name it `customer_support_app`. Add a description, e.g. *AI customer support agent with LangChain:
   RAG, ReAct agent, DAG routing and Whisper voice input*.
3. Choose **Public**. Leave **"Add a README"**, **.gitignore** and **license** *unticked*, because the project
   already has them.
4. Click **Create repository** and copy the URL shown, e.g.
   `https://github.com/Ramakrishna2006/customer_support_app.git`.

## 11.3 First-time Git setup (once per computer)

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Use the email address of your GitHub account.

## 11.4 Commit and push

From the project folder:

```powershell
git init
git add .
git status
```

**Check the `git status` list before committing.** It must **not** include `.env`, `venv/`, `orders.csv` or
`tickets.csv`. If `.env` is listed, stop and fix `.gitignore` first.

```powershell
git commit -m "AI customer support assistant with LangChain"
git branch -M main
git remote add origin https://github.com/Ramakrishna2006/customer_support_app.git
git push -u origin main
```

The first push opens a browser window to sign in to GitHub. (If it asks for a password in the terminal, use a
**personal access token** from *GitHub → Settings → Developer settings → Personal access tokens*, not your
account password.)

## 11.5 Finish the repository page

1. **README links:** replace `Ramakrishna2006` in `README.md` (badge and clone URL) with your GitHub username.
2. **License:** replace `YOUR NAME` in `LICENSE`.
3. **Screenshot:** run the app, take a screenshot, save it as `docs/images/screenshot.png`, and add this line
   under the title in `README.md`:
   ```markdown
   ![App screenshot](docs/images/screenshot.png)
   ```
4. **Topics:** on the repository page, click the ⚙️ next to *About* and add topics such as `langchain`,
   `rag`, `react-agent`, `ollama`, `gradio`, `whisper`, `customer-support`.

Commit and push those changes:

```powershell
git add .
git commit -m "Add screenshot and links"
git push
```

## 11.6 Automatic tests (GitHub Actions)

`.github/workflows/tests.yml` tells GitHub to run the tests on every push:

````yaml
name: tests

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - name: Install test dependencies
        run: pip install -r requirements-dev.txt
      - name: Run tests (no LLM calls, no API keys needed)
        run: pytest -v
````

Open the **Actions** tab on your repository to watch it run. A green tick means all tests passed; the badge in
the README shows the latest result.

## 11.7 Everyday Git

```powershell
git status                     # what changed?
git add .                      # stage everything
git commit -m "Describe the change"
git push                       # upload
git log --oneline              # history
```

## Checkpoint

Your repository page shows the README with the architecture diagram, the **tests** badge is green, and `.env`
is nowhere in the file list. 🎉 You've built and published the whole project.

Back to the **[guide index](README.md)** · See also **[Troubleshooting](troubleshooting.md)**
