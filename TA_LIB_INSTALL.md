# Инструкция по установке TA-Lib для Windows

## Проблема
TA-Lib требует установки C-библиотеки TA-Lib перед установкой Python-пакета.

## Решение 1: Использование готового wheel файла
1. Скачайте соответствующий wheel файл для Python 3.10 и Windows:
   - https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
   - Или используйте: TA_Lib-0.4.24-cp310-cp310-win_amd64.whl

2. Установите wheel файл:
   ```bash
   pip install TA_Lib-0.4.24-cp310-cp310-win_amd64.whl
   ```

## Решение 2: Установка через conda
```bash
conda install -c conda-forge ta-lib
```

## Решение 3: Ручная установка C-библиотеки
1. Скачайте C-библиотеку TA-Lib: http://ta-lib.org/hdr_dw.html
2. Распакуйте в C:\ta-lib
3. Установите Python-пакет: `pip install TA-Lib`

## Альтернатива
Используйте библиотеку `ta` (уже включена в requirements.txt) вместо TA-Lib:
```python
import ta
# Вместо import talib
```

## После установки
Раскомментируйте строку в requirements.txt:
```
TA-Lib>=0.4.24
``` 