const CACHE_NAME = 'trading-bot-v1';
const OFFLINE_URL = '/offline.html';

const STATIC_RESOURCES = [
    '/',
    '/index.html',
    '/style.css',
    '/dashboard.js',
    '/manifest.json',
    '/offline.html',
    'https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js',
    'https://cdn.plot.ly/plotly-latest.min.js'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                return cache.addAll(STATIC_RESOURCES);
            })
            .then(() => {
                return self.skipWaiting();
            })
    );
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME) {
                        return caches.delete(cacheName);
                    }
                })
            );
        }).then(() => {
            return self.clients.claim();
        })
    );
});

self.addEventListener('fetch', (event) => {
    if (event.request.mode === 'navigate') {
        event.respondWith(
            fetch(event.request)
                .catch(() => {
                    return caches.match(OFFLINE_URL);
                })
        );
        return;
    }

    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                if (response) {
                    return response;
                }

                return fetch(event.request)
                    .then((response) => {
                        if (!response || response.status !== 200 || response.type !== 'basic') {
                            return response;
                        }

                        const responseToCache = response.clone();
                        caches.open(CACHE_NAME)
                            .then((cache) => {
                                cache.put(event.request, responseToCache);
                            });

                        return response;
                    });
            })
    );
});

self.addEventListener('sync', (event) => {
    if (event.tag === 'sync-trades') {
        event.waitUntil(syncTrades());
    }
});

async function syncTrades() {
    try {
        const db = await openDB();
        const trades = await db.getAll('trades');
        
        for (const trade of trades) {
            if (!trade.synced) {
                try {
                    await fetch('/api/trades', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(trade)
                    });
                    
                    trade.synced = true;
                    await db.put('trades', trade);
                } catch (error) {
                    console.error('Ошибка синхронизации:', error);
                }
            }
        }
    } catch (error) {
        console.error('Ошибка синхронизации:', error);
    }
}

async function openDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('TradingBotDB', 1);
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);
        
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains('trades')) {
                db.createObjectStore('trades', { keyPath: 'id' });
            }
        };
    });
} 