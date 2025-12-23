import pandas as pd
import json
import os
import glob

input_path = './data/*.csv' # CSVが置いてあるフォルダ
output_file = 'splatoon3.inter'

all_data = []

for file in glob.glob(input_path):
    df = pd.read_csv(file)
    
    # 各行（各試合）ごとにループ
    for index, row in df.iterrows():
        # 試合ごとの勝敗結果を取得 ('alpha' または 'bravo')
        match_winner = row['win']
        
        # A1-A4 (Alphaチーム), B1-B4 (Bravoチーム) の8人分を処理
        for team_prefix in ['A', 'B']:
            # このチームが勝ったかどうかを判定
            current_team_name = 'alpha' if team_prefix == 'A' else 'bravo'
            label = 1.0 if match_winner == current_team_name else 0.0
            
            for i in range(1, 5):
                p = f"{team_prefix}{i}-"
                weapon_col = f"{p}weapon"
                ability_col = f"{p}abilities"
                
                # プレイヤーデータが空（切断など）の場合はスキップ
                if pd.isna(row[weapon_col]) or pd.isna(row[ability_col]):
                    continue
                
                try:
                    # JSON文字列を辞書に変換
                    abilities = json.loads(row[ability_col])
                    
                    for ab_name, value in abilities.items():
                        all_data.append({
                            'weapon_id:token': row[weapon_col],
                            'ability_id:token': ab_name,
                            'mode:token': row['mode'],
                            'stage:token': row['stage'],
                            'label:float': label
                        })
                except:
                    continue

# DataFrameに変換して保存
inter_df = pd.DataFrame(all_data)
inter_df.to_csv(output_file, sep='\t', index=False)

# 確認用：ラベルの分布を表示
print(inter_df['label:float'].value_counts())
print(f"変換完了！ {len(inter_df)} 件のインタラクションを作成しました。")
