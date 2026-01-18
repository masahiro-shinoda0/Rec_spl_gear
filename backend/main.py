import torch
import functools
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

# PyTorch 2.6+ 互換性 & デバイスマッピングパッチ
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
torch.load = functools.partial(torch.load, weights_only=False, map_location=device)

from recbole.quick_start import load_data_and_model

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(os.path.dirname(BASE_DIR), "saved", "FM-Jan-08-2026_03-48-50.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
    else:
        print(f"Loading Model onto {device}...")
        # データセットとモデルのロード
        config, model, dataset, _, _, _ = load_data_and_model(model_file=model_path)
        model.to(device)
        model.eval()
        
        ml_models["model"] = model
        ml_models["dataset"] = dataset
        print("Splatoon 3 Gear Recommendation System is Ready.")
    yield
    ml_models.clear()

app = FastAPI(title="Splatoon 3 Weapon-to-Gear API", lifespan=lifespan)

@app.get("/recommend/{weapon_id}")
async def get_gear_recommendation(weapon_id: str, top_k: int = 5):
    model = ml_models.get("model")
    dataset = ml_models.get("dataset")

    if not model or not dataset:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 武器名(weapon_id)を内部IDに変換
        # 注意: 学習時の設定により、dataset.uid_field が 'weapon_id' になっています
        w_id = dataset.token2id(dataset.uid_field, weapon_id)
        
        with torch.no_grad():
            w_tensor = torch.LongTensor([w_id]).to(device)
            # 全ギアパワー(ability_id)に対するスコアを計算
            scores = model.full_sort_predict(w_tensor)
            
            # スコア上位k個を取得
            _, topk_iid_list = torch.topk(scores, top_k)
            topk_iid_list = topk_iid_list.cpu().numpy().tolist()[0]

        # 内部IDをギアパワー名(ability_id)に変換
        recommend_gears = dataset.id2token(dataset.iid_field, topk_iid_list)
        
        return {
            "input_weapon": weapon_id,
            "recommended_gears": recommend_gears.tolist(),
            "status": "success"
        }

    except Exception:
        raise HTTPException(status_code=404, detail=f"Weapon '{weapon_id}' not found in dataset.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)