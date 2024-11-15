format:
    ruff format 

lint: format
    ruff check --fix

check:
    ruff format --check
    ruff check

train:
    python3 src/train.py /workspaces/SmartGB/src/dumpped.yml

commit msg:
    git add .
    git commit -m "{{msg}}"
    git push