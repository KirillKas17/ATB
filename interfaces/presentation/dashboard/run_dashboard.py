#!/usr/bin/env python3
"""
Простой HTTP сервер для запуска дашборда ATB
"""

import http.server
import os
import socketserver
from pathlib import Path


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Добавляем CORS заголовки для разработки
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def main():
    # Переходим в папку dashboard
    dashboard_dir = Path(__file__).parent
    os.chdir(dashboard_dir)

    # Настройки сервера
    PORT = 8080
    HOST = "localhost"

    print(f"🚀 Запуск дашборда ATB...")
    print(f"📊 Откройте браузер и перейдите по адресу: http://{HOST}:{PORT}")
    print(f"📁 Папка дашборда: {dashboard_dir.absolute()}")
    print(f"⏹️  Для остановки нажмите Ctrl+C")
    print("-" * 50)

    try:
        with socketserver.TCPServer((HOST, PORT), DashboardHandler) as httpd:
            print(f"✅ Сервер запущен на http://{HOST}:{PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️  Сервер остановлен")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Порт {PORT} уже занят. Попробуйте другой порт:")
            print(f"   python run_dashboard.py --port 8081")
        else:
            print(f"❌ Ошибка запуска сервера: {e}")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")


if __name__ == "__main__":
    main()
