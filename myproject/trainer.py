import pandas as pd
import json
import glob
import os

# 1. スプラトゥーン3の全ギアパワーリスト（内部名）
# これにより「装備しなかったギア」を特定できます
ALL_ABILITIES = [
    'ink_saver_main', 'ink_saver_sub', 'ink_recovery_up', 'run_speed_up', 
    'swim_speed_up', 'special_charge_up', 'special_saver', 'special_power_up', 
    'quick_respawn', 'sub_power_up', 'ink_resistance_up', 'sub_resistance_up', 
    'intensify_action', 'opening_gambit', 'last_ditch_effort', 'tenacity', 
    'comeback', 'ninja_squid', 'haunt', 'thermal_ink', 'respawn_punisher', 
    'ability_doubler', 'stealth_jump', 'object_shredder', 'drop_roller'
]

input_path = './data/*.csv'
output_file = 'splatoon3.inter'
all_data = []

for file in glob.glob(input_path):
    print(f"Reading: {file}")
    df = pd.read_csv(file)
    
    for _, row in df.iterrows():
        # 基本情報
        mode = row['mode']
        stage = row['stage']
        win_team = row['win'] # 'alpha' or 'bravo'
        
        for team in ['A', 'B']:
            team_win_status = 1.0 if (team == 'A' and win_team == 'alpha') or (team == 'B' and win_team == 'bravo') else 0.0
            
            for i in range(1, 5):
                prefix = f"{team}{i}-"
                weapon_col = f"{prefix}weapon"
                ability_col = f"{prefix}abilities"
                
                if pd.isna(row[weapon_col]) or pd.isna(row[ability_col]):
                    continue
                
                try:
                    # 実際に装備しているギアを抽出
                    equipped_dict = json.loads(row[ability_col])
                    equipped_names = []
                    
                    # 装備したギアのポジティブサンプル
                    for name, value in equipped_dict.items():
                        if isinstance(value, bool):
                            weight = 1.0 if value else 0.0
                        else:
                            weight = float(value)
                        
                        if weight > 0:
                            equipped_names.append(name)
                            all_data.append({
                                'weapon_id:token': row[weapon_col],
                                'ability_id:token': name,
                                'mode:token': mode,
                                'stage:token': stage,
                                'weight:float': weight,
                                'label:float': team_win_status # 勝敗を反映
                            })
                    
                    # 【案B】装備しなかったギアをネガティブサンプルとして追加
                    # 1プレイヤーにつき数個、選ばなかったギアを「重み0、勝率0」として教える
                    unused_abilities = list(set(ALL_ABILITIES) - set(equipped_names))
                    import random
                    # 全て入れるとデータが巨大化するため、ランダムに3つ選ぶ（調整可能）
                    neg_samples = random.sample(unused_abilities, min(len(unused_abilities), 3))
                    
                    for neg_name in neg_samples:
                        all_data.append({
                            'weapon_id:token': row[weapon_col],
                            'ability_id:token': neg_name,
                            'mode:token': mode,
                            'stage:token': stage,
                            'weight:float': 0.0,  # 重みなし
                            'label:float': 0.0    # 負け扱い（選ばれなかったため）
                        })
                        
                except Exception as e:
                    continue

# 保存
inter_df = pd.DataFrame(all_data)
inter_df.to_csv(output_file, sep='\t', index=False)
print(f"変換完了: {len(inter_df)}行。負例サンプリングを適用しました。")
