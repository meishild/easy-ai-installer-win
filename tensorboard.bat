@echo off
call venv\Scripts\activate
pip install tensorboard
python -m tensorboard.main --logdir=./logs/
pause