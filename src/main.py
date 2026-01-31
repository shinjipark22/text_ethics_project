import glob
import pandas as pd
from data_loader import load_data
from processor import process_data
from model import get_model
import torch

def main():
    # 1. GPU 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"현재 사용 장치: {device}")

    if device.type == 'cuda':
        print(f"GPU 모델명: {torch.cuda.get_device_name(0)}")

    # 2. 데이터 로드 (테스트용으로 100개만)
    print("\n 데이터 로드")
    train_file_paths = glob.glob('data/train/*.json')
    
    all_train_df = []
    for path in train_file_paths:
        print(f" - 로딩 중:{path}")
        df = load_data(path)
        all_train_df.append(df)

    # 데이터 히니러 합치기
    train_df = pd.concat(all_train_df, ignore_index=True)

    # 테스트 데이터 가져오기 
    test_file_paths = glob.glob('data/test/*.json')
    test_df = load_data(test_file_paths[0]) if test_file_paths else pd.DataFrame()

    print(f"학습 데이터: {len(train_df)}개 | 테스트 데이터: {len(test_df)}개")
    print(f"라벨 분포:\n{train_df['label'].value_counts()}")

    # 3. 데이터 전처리 (토크나이징)
    print("\n 데이터 전처리 시작")
    tokenized_train, tokenized_test, tokenizer = process_data(train_df, test_df)

    print("\n 전처리 샘플 확인")
    print(f"원문: {train_df.iloc[0]['text']}")
    print(f"Input IDs: {tokenized_train[0]['input_ids'][:10]}...")
    print(f"Attention Mask: {tokenized_train[0]['attention_mask'][:10]}...")

    # 4. 모델 로드 및 GPU 탑재
    print("\n BERT 모델 불러오기, GPU 탑재")

    model = get_model(num_labels=2) # 0: Clean, 1: Immoral
    model.to(device)

    print("모델 로드 완료")

    # 5. Dry Run
    print("\n test dry run 시작")
    
    sample_input_id = torch.tensor([tokenized_train[0]['input_ids']]).to(device)
    sample_mask = torch.tensor([tokenized_train[0]['attention_mask']]).to(device)

    # 모델에 넣어보기 (예측)
    model.eval() # 평가 모드로 전환 (학습 X)
    with torch.no_grad():
        outputs = model(sample_input_id, attention_mask=sample_mask)

    print("테스트 주행 성공")
    print(f"출력 결과 크기: {outputs.logits.shape}")
    print(f"예측값: {outputs.logits}")

    predicted_label_id = torch.argmax(outputs.logits, dim=1).item()
    print(f"모델 판단: {'Immoral' if predicted_label_id == 1 else 'Clean'}")

if __name__ == "__main__":
    main()