from BartDataset import BartDataset
from CustomBart_Atomic_Tokenizer import CustomBart_Atomic_Tokenizer
from CustomBart_FG_Tokenizer import CustomBart_FG_Tokenizer
from DualBartModel import DualBartModel
import torch
from utils import *
from pretrained_utils import *


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    smiles = input('Enter the SMILES of the compound: ')
    virus_choice = input('Predict activity against HBV or HCV? (Enter HBV or HCV): ').lower()

    print('Analysis in progress ...')

    if virus_choice == 'hbv':
        model_path = 'model/hbv_model.pt'
        max_pact = max_pact_hbv
        min_pact = min_pact_hbv
    elif virus_choice == 'hcv':
        model_path = 'model/hcv_model.pt'
        max_pact = max_pact_hcv
        min_pact = min_pact_hcv
    else:
        raise ValueError("Invalid input. Please enter either 'HBV' or 'HCV'.")

    max_length = 250
    model = DualBartModel(config1, config2, reg_mod)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    smiles = remove_salt(smiles)

    enc1 = tokenizer1.encode_plus(smiles, truncation=True, max_length=max_length,
                                   padding='max_length', return_tensors='pt')
    enc2 = tokenizer2.encode_plus(smiles, truncation=True, max_length=max_length,
                                   padding='max_length', return_tensors='pt')

    with torch.no_grad():
        output = model(
            input_ids1=enc1['input_ids'].to(device),
            attention_mask1=enc1['attention_mask'].to(device),
            input_ids2=enc2['input_ids'].to(device),
            attention_mask2=enc2['attention_mask'].to(device),
        )

    raw = output.cpu().numpy()[0]
    pact = raw * (max_pact - min_pact) + min_pact
    ec50 = 10 ** (-pact) * 1e9

    print('SMILES:          ', smiles)
    print('Predicted pACT:  ', round(float(pact), 4))
    print('Predicted EC50:  ', round(float(ec50), 4), 'nM')
