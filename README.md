# Protein Structure Predictor API

A simple FastAPI app for predicting the secondary structure of a protein's central residue given a 17-residue sliding window.

## Usage

Start the API and send a GET request to:

```
/predict?residues=ACDEFGHIKLMNPQRST
```

- Input must be exactly 17 amino acid characters.
- Returns a list of predicted structures with confidence scores.

## Project Structure

- `model/` – Contains the CNN model definition.
- `training/` – Code for training the model.
- `main.py` – FastAPI server.

## Requirements

- FastAPI
- Torch
