import asyncio
import os
import platform
import shutil
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import psutil
from loguru import logger


@dataclass
class BuildConfig:
    """Конфигурация сборки"""

    onefile: bool = True
    windowed: bool = True
    debug: bool = False
    upx: bool = True
    clean: bool = True
    log_level: str = "INFO"
    name: str = "TradingBot"
    icon: Optional[str] = None
    version: str = "1.0.0"
    company: str = "TradingBot"
    copyright: str = "Copyright © 2024"
    description: str = "Advanced Trading Bot"


@dataclass
class BuildMetrics:
    """Метрики сборки"""

    start_time: datetime
    end_time: datetime
    build_size: int
    build_time: float
    memory_usage: float
    cpu_usage: float
    success: bool
    error: Optional[str] = None


class ExecutableBuilder:
    """Сборщик исполняемого файла"""

    def __init__(self, config: Optional[BuildConfig] = None):
        """Инициализация сборщика"""
        self.config = config or BuildConfig()
        self.metrics_history = []
        self._build_lock = asyncio.Lock()

        # Пути
        self.root_dir = Path(__file__).parent.parent
        self.main_script = self.root_dir / "main.py"
        self.dashboard_script = self.root_dir / "dashboard" / "run_dashboard.py"
        self.dashboard_url = "http://localhost:7860"

        # Директории и файлы для включения
        self.include_dirs = [
            "dashboard",
            "config",
            "models",
            "templates",
            "static",
            "ml",
            "core",
            "utils",
            "exchange",
        ]
        self.include_files = [".env", "requirements.txt", "README.md"]

        # Скрытые импорты
        self.hidden_imports = [
            "uvicorn.logging",
            "uvicorn.loops",
            "uvicorn.loops.auto",
            "uvicorn.protocols",
            "uvicorn.protocols.http",
            "uvicorn.protocols.http.auto",
            "uvicorn.protocols.websockets",
            "uvicorn.protocols.websockets.auto",
            "uvicorn.lifespan",
            "uvicorn.lifespan.on",
            "sklearn",
            "sklearn.ensemble",
            "sklearn.tree",
            "sklearn.metrics",
            "sklearn.model_selection",
            "pandas",
            "numpy",
            "talib",
            "optuna",
            "joblib",
            "aiofiles",
            "fastapi",
            "uvicorn",
            "websockets",
            "python-dotenv",
            "loguru",
        ]

    async def clean_build_dirs(self):
        """Очистка директорий сборки"""
        try:
            dirs_to_clean = ["build", "dist"]
            for dir_name in dirs_to_clean:
                if os.path.exists(dir_name):
                    shutil.rmtree(dir_name)

            # Удаление .spec файлов
            for spec_file in Path(".").glob("*.spec"):
                spec_file.unlink()

            logger.info("Директории сборки очищены")

        except Exception as e:
            logger.error(f"Ошибка при очистке директорий: {str(e)}")
            raise

    def _get_system_metrics(self) -> Dict:
        """Получение системных метрик"""
        return {
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            "cpu_usage": psutil.Process().cpu_percent(),
            "disk_usage": psutil.disk_usage("/").percent,
        }

    def _create_spec_file(self):
        """Создание .spec файла"""
        try:
            datas = []
            for d in self.include_dirs:
                if Path(d).exists():
                    datas.append(f"        ('{d}', '{d}'),")
            for f in self.include_files:
                if Path(f).exists():
                    datas.append(f"        ('{f}', '{f}'),")

            spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

a = Analysis([
    'main.py'],
    pathex=[],
    binaries=[],
    datas=[
{chr(10).join(datas)}
    ],
    hiddenimports={self.hidden_imports},
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='{self.config.name}',
          debug={str(self.config.debug).lower()},
          bootloader_ignore_signals=False,
          strip=False,
          upx={str(self.config.upx).lower()},
          console={str(not self.config.windowed).lower()},
          icon={f"'{self.config.icon}'" if self.config.icon else None},
          version='file_version_info.txt'
)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx={str(self.config.upx).lower()},
               upx_exclude=[],
               name='{self.config.name}')
"""

            with open(f"{self.config.name}.spec", "w", encoding="utf-8") as f:
                f.write(spec_content)

            logger.info(f"Создан файл {self.config.name}.spec")

        except Exception as e:
            logger.error(f"Ошибка при создании .spec файла: {str(e)}")
            raise

    def _create_version_info(self):
        """Создание информации о версии"""
        try:
            version_info = f"""# UTF-8
#
# For more details about fixed file info 'ffi' see:
# http://msdn.microsoft.com/en-us/library/ms646997.aspx
VSVersionInfo(
  ffi=FixedFileInfo(
    # filevers and prodvers should be always a tuple with four items: (1, 2, 3, 4)
    # Set not needed items to zero 0.
    filevers=({",".join(self.config.version.split("."))}),
    prodvers=({",".join(self.config.version.split("."))}),
    # Contains a bitmask that specifies the valid bits 'flags'r
    mask=0x3f,
    # Contains a bitmask that specifies the Boolean attributes of the file.
    flags=0x0,
    # The operating system for which this file was designed.
    # 0x4 - NT and there is no need to change it.
    OS=0x40004,
    # The general type of file.
    # 0x1 - the file is an application.
    fileType=0x1,
    # The function of the file.
    # 0x0 - the function is not defined for this fileType
    subtype=0x0,
    # Creation date and time stamp.
    date=(0, 0)
    ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'{self.config.company}'),
        StringStruct(u'FileDescription', u'{self.config.description}'),
        StringStruct(u'FileVersion', u'{self.config.version}'),
        StringStruct(u'InternalName', u'{self.config.name}'),
        StringStruct(u'LegalCopyright', u'{self.config.copyright}'),
        StringStruct(u'OriginalFilename', u'{self.config.name}.exe'),
        StringStruct(u'ProductName', u'{self.config.name}'),
        StringStruct(u'ProductVersion', u'{self.config.version}')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)"""

            with open("file_version_info.txt", "w", encoding="utf-8") as f:
                f.write(version_info)

            logger.info("Создан файл информации о версии")

        except Exception as e:
            logger.error(f"Ошибка при создании информации о версии: {str(e)}")
            raise

    async def build_executable(self) -> BuildMetrics:
        """Сборка исполняемого файла"""
        try:
            start_time = datetime.now()
            start_metrics = self._get_system_metrics()

            async with self._build_lock:
                # Очистка предыдущей сборки
                if self.config.clean:
                    await self.clean_build_dirs()

                # Создание .spec файла
                self._create_spec_file()

                # Создание информации о версии
                self._create_version_info()

                # Базовые параметры PyInstaller
                cmd = [
                    "pyinstaller",
                    "--clean",
                    "--noconfirm",
                    "--log-level",
                    self.config.log_level,
                ]

                # Добавление флагов
                if self.config.onefile:
                    cmd.append("--onefile")
                if self.config.windowed:
                    cmd.append("--windowed")
                if self.config.icon:
                    cmd.extend(["--icon", self.config.icon])

                # Добавление данных
                for src, dst in [(d, d) for d in self.include_dirs] + [
                    (f, ".") for f in self.include_files
                ]:
                    if os.path.exists(src):
                        cmd.extend(["--add-data", f"{src}{os.pathsep}{dst}"])

                # Добавление скрытых импортов
                for imp in self.hidden_imports:
                    cmd.extend(["--hidden-import", imp])

                # Добавление точки входа
                cmd.append(str(self.main_script))

                # Запуск сборки
                logger.info("Начало процесса сборки...")
                subprocess.run(cmd, check=True)

                # Расчет метрик
                end_time = datetime.now()
                end_metrics = self._get_system_metrics()
                build_time = (end_time - start_time).total_seconds()

                # Получение размера сборки
                build_size = 0
                for path in Path("dist").rglob("*"):
                    if path.is_file():
                        build_size += path.stat().st_size

                metrics = BuildMetrics(
                    start_time=start_time,
                    end_time=end_time,
                    build_size=build_size,
                    build_time=build_time,
                    memory_usage=end_metrics["memory_usage"]
                    - start_metrics["memory_usage"],
                    cpu_usage=end_metrics["cpu_usage"],
                    success=True,
                )

                self.metrics_history.append(metrics)

                logger.info(f"Сборка завершена успешно за {build_time:.2f} сек")
                logger.info(f"Размер сборки: {build_size / 1024 / 1024:.2f} MB")

                return metrics

        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка сборки: {str(e)}")
            metrics = BuildMetrics(
                start_time=start_time,
                end_time=datetime.now(),
                build_size=0,
                build_time=(datetime.now() - start_time).total_seconds(),
                memory_usage=0,
                cpu_usage=0,
                success=False,
                error=str(e),
            )
            self.metrics_history.append(metrics)
            raise

        except Exception as e:
            logger.error(f"Ошибка при сборке: {str(e)}")
            metrics = BuildMetrics(
                start_time=start_time,
                end_time=datetime.now(),
                build_size=0,
                build_time=(datetime.now() - start_time).total_seconds(),
                memory_usage=0,
                cpu_usage=0,
                success=False,
                error=str(e),
            )
            self.metrics_history.append(metrics)
            raise

    async def run_application(self):
        """Запуск приложения"""
        try:
            # Запуск trading engine
            engine_proc = subprocess.Popen(
                [sys.executable, str(self.main_script)],
                creationflags=(
                    subprocess.CREATE_NEW_CONSOLE
                    if platform.system() == "Windows"
                    else 0
                ),
            )

            # Запуск dashboard
            dash_proc = subprocess.Popen(
                [sys.executable, str(self.dashboard_script)],
                creationflags=(
                    subprocess.CREATE_NEW_CONSOLE
                    if platform.system() == "Windows"
                    else 0
                ),
            )

            # Открытие браузера
            webbrowser.open(self.dashboard_url)

            # Ожидание завершения
            engine_proc.wait()
            dash_proc.terminate()

        except Exception as e:
            logger.error(f"Ошибка при запуске приложения: {str(e)}")
            raise


async def main():
    """Основная функция"""
    try:
        # Инициализация сборщика
        builder = ExecutableBuilder()

        if len(sys.argv) > 1 and sys.argv[1] == "build":
            # Сборка исполняемого файла
            metrics = await builder.build_executable()
            logger.info(f"Метрики сборки: {metrics}")
        else:
            # Запуск приложения
            await builder.run_application()

    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Настройка логирования
    logger.add(
        "logs/build_{time}.log", rotation="1 day", retention="7 days", level="INFO"
    )

    # Запуск асинхронного main
    asyncio.run(main())
