import os
import sys
from pathlib import Path

import streamlit as st

# Добавление корневой директории в PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Запуск дашборда
if __name__ == "__main__":
    os.chdir(root_dir)
    os.system("streamlit run dashboard/app.py")
