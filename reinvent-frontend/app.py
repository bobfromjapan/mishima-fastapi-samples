import gradio as gr
import requests
from rdkit import Chem
from rdkit.Chem import Draw
import os

gen_type_model = {
    "denovo": "priors/reinvent.prior",
    "scaffold": "priors/libinvent.prior",
    "linker": "priors/linkinvent.prior",
    "mol2mol": "priors/mol2mol_medium_similarity.prior",
}

URL = os.getenv("REINVENT_API_URL")  # "http://localhost:8080/sampling"


def generate_compounds(
    gen_type,
    seed,
    temperature,
    num_smiles,
    smiles_input,
):

    # SamplingConfオブジェクトを構築
    conf = {
        "run_type": "sampling",
        "use_cuda": False,
        "json_out_config": "_sampling.json",
        "seed": seed,
        "parameters": {
            "model_file": gen_type_model[gen_type],
            "output_file": "sampling.csv",
            "sample_strategy": "multinomial",
            "temperature": temperature,
            "tb_logdir": None,
            "num_smiles": int(num_smiles),
            "unique_molecules": True,
            "randomize_smiles": True,
        },
    }

    # MultipleSmilesオブジェクトを準備
    smiles = None
    if smiles_input:
        smiles_list = []
        for line in smiles_input.strip().splitlines():
            smiles_list.append({"smiles": line.strip()})
        smiles = {"row": smiles_list}

    # FastAPIエンドポイントにPOSTリクエストを送信
    data = {"conf": conf, "smiles": smiles}

    response = requests.post(URL, json=data)
    print(response.json())
    if response.status_code == 200:
        result = response.json()
        if "results" in result and all(
            "SMILES" in item and "NLL" in item for item in result["results"]
        ):
            smiles_nll_list = [
                (item["SMILES"], item["NLL"]) for item in result["results"]
            ]
            images = []

            # RDKitを使用して画像を生成
            for smi, nll in smiles_nll_list:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    img = Draw.MolToImage(mol, size=(300, 300))
                    # ラベルとしてSMILES文字列とNLL値を設定
                    label = f"SMILES: {smi}\nNLL: {nll}"
                    images.append((img, label))
                else:
                    images.append((None, f"無効なSMILES: {smi}"))

            return images
        else:
            return "APIのレスポンスにエラーがあります。'smiles'または'NLL'が見つかりません。"
    else:
        return f"エラー: {response.status_code}, {response.text}"


# Gradioインターフェースを定義
iface = gr.Interface(
    fn=generate_compounds,
    inputs=[
        gr.Dropdown(
            ["denovo", "scaffold", "linker", "mol2mol"],
            label="generation type",
            value="denovo",
        ),
        gr.Number(label="Seed", value=0),
        gr.Number(label="Temperature", value=0.5),
        gr.Number(label="Number of SMILES", value=10),
        gr.TextArea(
            label="SMILES Input",
            placeholder="SMILESを1行ずつ入力してください（Generation typeがdenovo以外の時に必要）",
        ),
    ],
    outputs=gr.Gallery(label="生成された化合物"),
    title="化合物生成ツール",
    description="Reinvent4を利用して化合物を生成し、RDKitでレンダリングします。画像のラベルにはSMILES文字列とNLL値が表示されます。",
)

iface.launch(server_name="0.0.0.0", server_port=8080)
