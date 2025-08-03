import os
import subprocess
import sys
from pathlib import Path

# Добавление корневой директории в PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Запуск дашборда
if __name__ == "__main__":
    os.chdir(root_dir)
    try:
        # Безопасный запуск с полным путем
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"], check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running dashboard: {e}")
        sys.exit(1)
