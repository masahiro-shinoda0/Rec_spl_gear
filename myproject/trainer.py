import pandas as pd
import json
import glob
import os
import random

# 全ギアパワーの定義
ALL_ABILITIES = [
    'ink_saver_main', 'ink_saver_sub', 'ink_recovery_up', 'run_speed_up', 
    'swim_speed_up', 'special_charge_up', 'special_saver', 'special_power_up', 
    'quick_respawn', 'sub_power_up', 'ink_resistance_up', 'sub_resistance_up', 
    'intensify_action', 'opening_gambit', 'last_ditch_effort', 'tenacity', 
    'comeback', 'ninja_squid', 'haunt', 'thermal_ink', 'respawn_punisher', 
    'ability_doubler', 'stealth_jump', 'object_shredder', 'drop_roller'
]

input_path = './data/*.csv' # CSVが格納されているパス
output_file = 'splatoon3_xmatch.inter'
all_data = []

for file in glob.glob(input_path):
    print(f"Processing: {file}")
    df = pd.read_csv(file)
    
    # 【重要】Xマッチのみを抽出
    if 'lobby' in df.columns:
        df = df[df['lobby'] == 'xmatch']
    else:
        continue

    for _, row in df.iterrows():
        # Aチーム・Bチームそれぞれのプレイヤー（計8人）を走査
        for team in ['A', 'B']:
            # チームの勝敗判定
            is_win = 1.0 if (team == 'A' and row['win'] == 'alpha') or (team == 'B' and row['win'] == 'bravo') else 0.0
            
            for i in range(1, 5):
                prefix = f"{team}{i}-"
                weapon = row.get(f"{prefix}weapon")
                abilities_json = row.get(f"{prefix}abilities")
                
                if pd.isna(weapon) or pd.isna(abilities_json):
                    continue
                
                try:
                    equipped_dict = json.loads(abilities_json)
                    equipped_names = []
                    
                    # 1. 装備しているギア（正例）
                    for name, value in equipped_dict.items():
                        # AP値(float)またはメイン専用(bool)を数値化
                        weight = float(value) if not isinstance(value, bool) else (1.0 if value else 0.0)
                        
                        if weight > 0:
                            equipped_names.append(name)
                            all_data.append({
                                'weapon_id:token': weapon,
                                'ability_id:token': name,
                                'mode:token': row['mode'],
                                'stage:token': row['stage'],
                                'weight:float': weight,
                                'label:float': is_win
                            })
                    
                    # 2. 装備していないギア（負例サンプル：案B）
                    # 1プレイヤーにつき5つ、選ばなかったギアを「勝率0/重み0」として追加
                    unused = list(set(ALL_ABILITIES) - set(equipped_names))
                    neg_samples = random.sample(unused, min(len(unused), 12))
                    for neg_name in neg_samples:
                        all_data.append({
                            'weapon_id:token': weapon,
                            'ability_id:token': neg_name,
                            'mode:token': row['mode'],
                            'stage:token': row['stage'],
                            'weight:float': 0.0,
                            'label:float': 0.0
                        })
                except:
                    continue

# DataFrame化して保存
inter_df = pd.DataFrame(all_data)
# RecBole用のヘッダーを付けて保存
inter_df.to_csv(output_file, sep='\t', index=False)
print(f"完了！ Xマッチ限定データ件数: {len(inter_df)}")
