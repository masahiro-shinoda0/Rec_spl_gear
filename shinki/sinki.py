from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed
import torch
from recbole.data import Interaction
import numpy as np

# --- 1. 設定とデータの読み込み ---
# run_recbole の代わりに以下のように記述します
config = Config(
    model='SASRec',
    dataset='ml-100k',
    config_file_list=['sinki.yaml']
)
init_seed(config['seed'], config['reproducibility'])

# データセットの作成
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

# --- 2. モデルとトレーナーの作成 ---
# モデルの初期化
model = get_model(config['model'])(config, train_data.dataset).to(config['device'])

# トレーナーの初期化
trainer = get_trainer(config['model'], config['model'])(config, model)

# --- 3. 学習の実行 ---
# ここで学習が行われます
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

# --- 4. 推論用データの作成（あなたのコードの続き） ---
# dataset は上で作成済みなので、そのまま使えます

# Generate input (item sequence of an active user)


item_sequence = ['12', '13'] # specify token
item_length = len(item_sequence)
pad_length = 50  # pre-defined by recbole

padded_item_sequence = torch.nn.functional.pad(
    torch.tensor(dataset.token2id(dataset.iid_field, item_sequence)), (0, pad_length - item_length), "constant", 0,
)

input_interaction = Interaction(
    {
        "item_id_list": padded_item_sequence.reshape(1, -1),
        "item_length": torch.tensor([item_length]),
    }
)

# Make recommendation: top-5 recommendation

scores = model.full_sort_predict(input_interaction.to(model.device))

top_k = torch.topk(scores, k=5, dim=1).indices
tk_cpu = top_k.to("cpu")
tk = tk_cpu.numpy()
recom_list = dataset.id2token(dataset.iid_field, tk[0])

print("Recommended Tokens: " + recom_list) # Tokens of top 5 items
