import glob
import pandas as pd
import torch
import os

from data_loader import load_data
from processor import process_data
from model import get_model
from dataset import EthicsDataset
from trainer import train_model


def main():
    PROJECT_NAME = "text_ethics"

    MODEL_NAME = "bert-base-multilingual-cased" # 모델 바꾸고 싶으면 여기만 수정

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
    tokenized_train, tokenized_test, tokenizer = process_data(train_df, test_df, MODEL_NAME)

    # 4. Dataset 만들기
    print("Dataset으로 감싸는 중")
    train_dataset = EthicsDataset(tokenized_train, train_df['label'].to_list())
    val_dataset = EthicsDataset(tokenized_test, test_df['label'].to_list())

    # 5. 모델 로드
    print("BERT 모델 초기화")
    model = get_model(MODEL_NAME, num_labels=2)
    model.to(device)

    # 6. 학습 시작
    model = train_model(
        model,
        train_dataset,
        val_dataset,
        device,
        epochs=3,
        batch_size=32
    )

    # 7. 학습된 모델 저장
    print("모델 저장 중...")

    safe_model_name = MODEL_NAME.replace("/", "-")

    output_dir = f"./models/{PROJECT_NAME}_{safe_model_name}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("모든 작업 완료")

if __name__ == "__main__":
    main()