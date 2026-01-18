import torch
import functools
import os
import sqlite3
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from recbole.data.interaction import Interaction

# --- 1. デバイス・パッチ設定 ---
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
torch.load = functools.partial(torch.load, weights_only=False, map_location=device)

from recbole.data import create_dataset
from recbole.utils import get_model

ml_models = {}
DB_PATH = "splatoon3_master.db"

def get_ja_name(key: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name_ja FROM master_data WHERE key = ?", (key,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else key
    except:
        return key

@asynccontextmanager
async def lifespan(app: FastAPI):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(os.path.dirname(BASE_DIR), "saved", "FM-Jan-08-2026_03-48-50.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
    else:
        print(f"Loading FM Model onto {device}...")
        checkpoint = torch.load(model_path)
        config = checkpoint['config']
        config['device'] = device
        
        # 学習時の数値特徴量を確実に登録する
        config['numerical_features'] = ['weight', 'label']
        
        dataset = create_dataset(config)
        model = get_model(config['model'])(config, dataset).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        ml_models["model"] = model
        ml_models["dataset"] = dataset
        print(f"Ready. Numerical features detected: {dataset.num_values_field}")
    yield
    ml_models.clear()

app = FastAPI(title="Splatoon 3 Gear Recommendation API", lifespan=lifespan)

@app.get("/recommend/{weapon_id}")
async def recommend(weapon_id: str, top_k: int = 5):
    model = ml_models.get("model")
    dataset = ml_models.get("dataset")

    if not model or not dataset:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if weapon_id not in dataset.field2id_token[dataset.uid_field]:
        raise HTTPException(status_code=404, detail=f"Weapon '{weapon_id}' not found.")

    try:
        w_id = dataset.token2id(dataset.uid_field, [weapon_id])[0]
        num_items = dataset.item_num
        item_ids = torch.arange(num_items).to(device)
        
        # モデルが期待するフィールドを構築
        interaction_dict = {
            dataset.uid_field: torch.LongTensor([w_id] * num_items).to(device),
            dataset.iid_field: item_ids,
            'mode': torch.LongTensor([1] * num_items).to(device),
            'stage': torch.LongTensor([1] * num_items).to(device),
        }

        # 数値特徴量の補完 (weight と label の両方を明示)
        # 内部で [1, 1] に分割されるため、float型の入力を2種類用意する
        interaction_dict['weight'] = torch.FloatTensor([1.0] * num_items).to(device)
        interaction_dict['label'] = torch.FloatTensor([0.0] * num_items).to(device)

        inter = Interaction(interaction_dict)
        
        # predictの前に interaction の中身を同期させる (重要)
        # これにより dataset の定義と入力 tensor が一致します
        with torch.no_grad():
            scores = model.predict(inter) 
            
            topk_scores, topk_indices = torch.topk(scores, min(top_k + 2, num_items))
            topk_iid_list = topk_indices.cpu().numpy().tolist()

        recommend_tokens = dataset.id2token(dataset.iid_field, topk_iid_list)
        
        results = []
        for token in recommend_tokens:
            if token == "[PAD]" or len(results) >= top_k:
                continue
            results.append({
                "token": token,
                "name": get_ja_name(token)
            })

        return {
            "input_weapon": {"token": weapon_id, "name": get_ja_name(weapon_id)},
            "recommendations": results
        }

    except Exception:
        error_msg = traceback.format_exc()
        print(f"Full Error Traceback:\n{error_msg}")
        return {"error": "Internal inference error", "details": error_msg.splitlines()[-1]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)