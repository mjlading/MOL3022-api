from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from model.model import CNNModel, OUTPUT_SIZE

idx_to_structure = {
    0: "_",
    1: "H",
    2: "?",
    3: "E",
    4: "G",
    5: "S",
}

unique_residues = ['I', 'E', 'N', 'F', 'S', 'D', 'G', 'W', 'T', 'L', 'Q', 'C', 'B', 'M', 'V', 'K', 'X', 'H', 'P', 'Z', 'A', 'Y', 'R']
residue_to_idx = {res: i for i, res in enumerate(unique_residues)}

model = CNNModel(num_residues=len(unique_residues), embedding_dim=50)
model.load_state_dict(torch.load("best_model.ptrom", map_location="cpu"))
model.eval()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Predict the structure of the central residue given a sliding window of 17 residues
@app.get("/predict")
def predict(residues: str):
    indices = [residue_to_idx.get(r, residue_to_idx["X"]) for r in residues]
    if len(indices) != 17:
        return {"error": "Expected 17 residues in sliding window."}
    
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

        sorted_preds = sorted(
            [{"structure": idx_to_structure[i], "confidence": round(probs[i].item(), 4)} for i in range(len(probs))],
            key=lambda x: x["confidence"],
            reverse=True
        )

    return {"predictions": sorted_preds}
