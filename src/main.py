import glob
import pandas as pd
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login

from data_loader import load_data
from processor import process_data
from model import get_model
from dataset import EthicsDataset
from trainer import train_model


def main():
    load_dotenv() # .env 파일 찾아서 로드

    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_USERNAME = os.getenv("HF_USERNAME")

   
    if HF_TOKEN:
        print(f"Hugging Face 로그인중... (User: {HF_USERNAME})")
        login(token=HF_TOKEN)
    else:
        print(".env 파일에서 HF_TOKEN을 열 수 없음. 업로드 에러가 발생할 수 있음.")

    EXP_ID = "EXP-04"
    PROJECT_NAME = "text_ethics"
    MODEL_NAME = "beomi/kcbert-base" # 모델 바꾸고 싶으면 여기만 수정

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

    # 데이터 히니로 합치기
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

    # 저장 경로 미리 설정
    safe_model_name = MODEL_NAME.replace("/", "-")
    output_dir = f"./models/{EXP_ID}_{PROJECT_NAME}_{safe_model_name}"

    # 6. 학습 시작
    model, best_metrics = train_model(
        model,
        train_dataset,
        val_dataset,
        device,
        tokenizer,
        output_dir,
        epochs=3,
        batch_size=32
    )

    # 7. 결과 출력
    print(f"\n 최종 Best 성적 - F1: {best_metrics['f1']:.4f}, Acc: {best_metrics['accuracy']:.4f}")
    print("모든 작업 완료")

    # 8. Hugging Face Hub에 업로드
    if HF_TOKEN and HF_USERNAME:
        repo_id = f"{HF_USERNAME}/{PROJECT_NAME}-{safe_model_name}-{EXP_ID}"

        print(f"\n Hugging Face Hub에 업로드 중... (Target: {repo_id})")

        model.push_to_hub(repo_id, private=False)
        tokenizer.push_to_hub(repo_id, private=False)

        print(f"업로드 완료! 주소: https://huggingface.co/{repo_id}")
    else:
        print("\n 토큰이 없어서 업로드 건너뜀.")

if __name__ == "__main__":
    main()