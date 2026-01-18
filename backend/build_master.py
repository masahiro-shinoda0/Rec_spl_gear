import sqlite3
import requests

DB_NAME = "splatoon3_master.db"

# JSONデータを直接返してくれるエンドポイントに修正
ENDPOINTS = {
    "abilities": "https://stat.ink/api/v3/ability",
    "stages": "https://stat.ink/api/v3/stage",
    "weapons": "https://stat.ink/api/v3/weapon",
    "rules": "https://stat.ink/api/v3/rule"
}

def build():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS master_data")
    cursor.execute('''
        CREATE TABLE master_data (
            category TEXT,
            key TEXT PRIMARY KEY,
            name_ja TEXT,
            name_en TEXT
        )
    ''')

    for category, url in ENDPOINTS.items():
        print(f"Fetching {category}...")
        try:
            # stat.inkのAPIはUser-Agentがないと拒否される場合があるため追加
            headers = {"User-Agent": "Splatoon3-Rec-System-Agent"}
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            data = res.json()

            count = 0
            for item in data:
                key = item.get('key')
                # 名前データ（ja_JP, en_US）を取得
                name_ja = item.get('name', {}).get('ja_JP', key)
                name_en = item.get('name', {}).get('en_US', key)

                if key:
                    cursor.execute(
                        "INSERT OR REPLACE INTO master_data VALUES (?, ?, ?, ?)",
                        (category, key, name_ja, name_en)
                    )
                    count += 1
            
            print(f"  -> Successfully saved {count} items.")
            
        except Exception as e:
            print(f"  -> Error fetching {category}: {e}")

    conn.commit()
    conn.close()
    print("\nFinish! Database is updated.")

if __name__ == "__main__":
    build()