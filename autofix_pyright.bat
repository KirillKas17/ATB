@echo off
REM --- Автоматический фикс типовых ошибок Pyright ---

REM 1. Актуализируем pip и инструменты
python -m pip install --upgrade pip
pip install --upgrade black isort autoflake mypy pyright

REM 2. Сортировка импортов (безопасно)
isort .

REM 3. Форматирование кода (безопасно)
black .

REM 4. Удаление неиспользуемых импортов и переменных (безопасно)
autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r .

REM 5. Установка типов для сторонних библиотек (если поддерживается)
mypy --install-types --non-interactive

REM 6. Повторная проверка Pyright (отчёт сохраняется)
pyright > reports/pyright_autofix.txt

REM 7. Сообщение об окончании
echo.
echo ==========================
echo Автоматические исправления завершены!
echo Новый отчёт: reports\pyright_autofix.txt
echo Проверьте бизнес-логику вручную для сложных ошибок.
echo ==========================
pause