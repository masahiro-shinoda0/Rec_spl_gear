import torch
import numpy as np
from recbole.utils import get_model
from recbole.data import create_dataset
from recbole.data.interaction import Interaction

# NumPy互換性
np.long = np.int64

def recommend_with_lift(weapon_name):
    # 【最新のモデルファイル名に書き換えてください】
    model_file = 'saved/FM-Jan-05-2026_03-17-27.pth' 
    
    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    dataset = create_dataset(config)
    model = get_model(config['model'])(config, dataset).to('cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    weapon_id = dataset.token2id('weapon_id', weapon_name)
    ability_tokens = dataset.field2id_token['ability_id'][1:]
    num_items = len(ability_tokens)

    # 全ブキの平均的なスコアを計算するための準備
    all_weapon_ids = torch.arange(1, dataset.num('weapon_id'))
    
    with torch.no_grad():
        # 1. ターゲットブキのスコアを計算 (重み1.0固定)
        input_dict = {
            'weapon_id': torch.full((num_items,), weapon_id, dtype=torch.int64),
            'ability_id': torch.arange(1, num_items + 1, dtype=torch.int64),
            'mode': torch.zeros(num_items, dtype=torch.int64),
            'stage': torch.zeros(num_items, dtype=torch.int64),
        }
        val = torch.full((num_items, 1), 1.0, dtype=torch.float)
        idx = torch.zeros((num_items, 1), dtype=torch.float)
        input_dict['weight'] = torch.cat([val, idx], dim=-1)
        target_scores = model.predict(Interaction(input_dict))

        # 2. 代表的な30種類のブキで「平均的なギアの強さ」を計算
        avg_scores = torch.zeros(num_items)
        sample_size = min(30, len(all_weapon_ids))
        for w_id in all_weapon_ids[:sample_size]:
            input_dict['weapon_id'] = torch.full((num_items,), w_id, dtype=torch.int64)
            avg_scores += model.predict(Interaction(input_dict))
        avg_scores /= sample_size

        # 3. 特化度（偏差）の計算
        lift_scores = target_scores - avg_scores

    results = []
    for i, token in enumerate(ability_tokens):
        results.append((token, target_scores[i].item(), lift_scores[i].item()))
    
    # 特化度(Lift)が大きい順にソート
    results.sort(key=lambda x: x[2], reverse=True)

    print(f"\n✨ 【{weapon_name}】の特化度（リフト値）ランキング")
    print("-" * 75)
    print(f"{'順位':<4} | {'ギアパワー名':<25} | {'予測スコア':<10} | {'特化度(偏差)'}")
    print("-" * 75)
    for i, (name, raw_score, lift) in enumerate(results[:15]):
        print(f"{i+1:>4} | {name:<25} | {raw_score:.4f} | {lift:+.4f}")

if __name__ == '__main__':
    # 52ガロンで検証！
    recommend_with_lift('52gal')
