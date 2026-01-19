import numpy as np
import torch
# NumPy 2.0互換性パッチ
np.long = np.int64

import pandas as pd
import glob
import os
from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import get_model
from recbole.data.interaction import Interaction

# 引数に target_mode='area' を追加
def recommend_gear_sets(model_path, target_weapon='liter4k', target_mode='yagura', top_n=10):
    # 1. チェックポイントの読み込み
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # 2. データセットとモデルの再構成
    dataset = create_dataset(config)
    model = get_model(config['model'])(config, dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    device = config['device']
    gear_cols = ['m1', 'm2', 'm3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9']
    
    # ユニークなギア構成の抽出
    all_inter = dataset.inter_feat
    unique_builds = all_inter[gear_cols].drop_duplicates().copy()
    
    print(f"Checking {len(unique_builds)} unique builds for {target_weapon} in {target_mode}...")

    # 3. ターゲットブキとターゲットルールのID取得
    try:
        weapon_id = dataset.token2id('weapon_id', target_weapon)
        # ここでルールのIDを取得
        mode_id = dataset.token2id('mode', target_mode)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 4. 推論用データの作成
    input_dict = {
        'weapon_id': torch.tensor([weapon_id] * len(unique_builds)).to(device),
        'mode': torch.tensor([mode_id] * len(unique_builds)).to(device), # 指定ルールをセット
    }
    
    # ステージが学習に含まれている場合、一旦 0 (デフォルト) をセット
    if 'stage' in all_inter.columns:
        input_dict['stage'] = torch.tensor([0] * len(unique_builds)).to(device)

    for col in gear_cols:
        input_dict[col] = torch.tensor(unique_builds[col].values).to(device)

    inter = Interaction(input_dict)

    # 5. スコア予測
    with torch.no_grad():
        scores = model.predict(inter).cpu().numpy()

    # 6. 結果の集計と表示
    unique_builds['score'] = scores
    top_results = unique_builds.sort_values(by='score', ascending=False).head(top_n)

    print(f"\n--- Top {top_n} Gear Sets for {target_weapon} ({target_mode}) ---")
    for i, (idx, row) in enumerate(top_results.iterrows()):
        m_list = [dataset.id2token(f'm{j+1}', int(row[f'm{j+1}'])) for j in range(3)]
        s_list = [dataset.id2token(f's{j+1}', int(row[f's{j+1}'])) for j in range(9)]
        
        from collections import Counter
        s_counts = Counter(s_list)
        s_str = ", ".join([f"{k}:{v*0.3:.1f}" for k, v in s_counts.items() if k != 'none'])

        print(f"Rank {i+1} (Score: {row['score']:.4f})")
        print(f"  Main: {', '.join(m_list)}")
        print(f"  Sub : {s_str}")
        print("-" * 30)

if __name__ == '__main__':
    model_files = glob.glob('saved/DeepFM-Jan-19-2026_03-23-29.pth')
    if not model_files:
        print("Error: No .pth files found.")
    else:
        latest_model = max(model_files, key=os.path.getctime)
        print(f"Loading model: {latest_model}")
        # ルールを指定して実行
        recommend_gear_sets(latest_model, target_weapon='liter4k', target_mode='yagura')
