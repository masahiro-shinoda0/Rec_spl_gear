import sqlite3
import requests

# 保存先のデータベース名
DB_NAME = "splatoon3_master.db"

# stat.ink APIの一覧
ENDPOINTS = {
    "abilities": "https://stat.ink/api-info/ability3",
    "stages": "https://stat.ink/api-info/stage3",
    "weapons": "https://stat.ink/api-info/weapon3",
    "rules": "https://stat.ink/api-info/rule3"
}

def setup_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # テーブル作成 (トークンを主キーにする)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS master_data (
            category TEXT,
            key TEXT PRIMARY KEY,
            name_ja TEXT,
            name_en TEXT
        )
    ''')
    conn.commit()
    return conn

def fetch_and_save(conn):
    cursor = conn.cursor()
    
    for category, url in ENDPOINTS.items():
        print(f"Fetching {category} from stat.ink...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            for item in data:
                # トークン(key)、日本語名、英語名を取得
                key = item.get('key')
                name_ja = item.get('name', {}).get('ja_JP', key)
                name_en = item.get('name', {}).get('en_US', key)

                if key:
                    cursor.execute('''
                        INSERT OR REPLACE INTO master_data (category, key, name_ja, name_en)
                        VALUES (?, ?, ?, ?)
                    ''', (category, key, name_ja, name_en))
            
            print(f"Successfully saved {len(data)} items to {category}.")
        except Exception as e:
            print(f"Error fetching {category}: {e}")

    conn.commit()

if __name__ == "__main__":
    connection = setup_db()
    fetch_and_save(connection)
    connection.close()
    print(f"\nCompleted! Database saved as {DB_NAME}")