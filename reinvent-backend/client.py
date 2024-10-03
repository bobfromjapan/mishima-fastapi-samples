import requests

num_smiles = 10

url = "https://XXXXXXX"

data = {
    "conf": {
        "run_type": "sampling",
        "use_cuda": False,
        "json_out_config": "_sampling.json",
        "seed": 0,
        "parameters": {
            "model_file": "priors/reinvent.prior",
            "output_file": "sampling.csv",
            "num_smiles": num_smiles,
            "unique_molecules": True,
            "randomize_smiles": True,
            "temperature": 0.7,
        },
    },
    "smiles": None,
}

response = requests.post(url + "/sampling", json=data)

print(response.json())
