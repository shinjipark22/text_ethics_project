import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data_loader import load_data
from processor import process_data
from dataset import EthicsDataset
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def evaluate_saved_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. 저장된 모델 경로 
    model_path = "./models/text_ethics_beomi-kcbert-base"
    
    print(f"저장된 모델 로드 : {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2. 테스트 데이터 로드 (전체 테스트셋 사용)
    import glob
    test_files = glob.glob('data/test/*.json')
    test_df = load_data(test_files[0])
    
    # 3. 전처리
    _, tokenized_test, _ = process_data(test_df, test_df, "beomi/kcbert-base")
    test_dataset = EthicsDataset(tokenized_test, test_df['label'].to_list())
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=32)

    # 4. 평가 시작
    model.eval()
    preds, labels = [], []
    
    print("평가 시작")
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits.detach().cpu().numpy()
        preds.extend(np.argmax(logits, axis=1))
        labels.extend(batch['labels'].cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return acc, f1

if __name__ == "__main__":
    evaluate_saved_model()