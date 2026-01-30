import glob
import pandas as pd
from data_loader import load_data
from processor import process_data

# 1. 학습 데이터 파일 경로들 가져오기
train_file_paths = glob.glob('data/train/*.json')
print(f"파일 목록: {train_file_paths}")

# 2. 모든 파일을 돌면서 데이터 합치기
all_train_df = []
for path in train_file_paths:
    df = load_data(path)
    all_train_df.append(df)

# 리스트에 담긴 데이터프레임들을 하나로 통합
train_df = pd.concat(all_train_df, ignore_index=True)

# 테스트 데이터도 똑같이 가져오기
test_file_paths = glob.glob('data/test/*.json')
test_df = load_data(test_file_paths[0]) if test_file_paths else pd.DataFrame()

# 결과 확인
print("-" * 30)
print(f"최종 통합 학습 데이터 개수: {len(train_df)}")
print(f"테스트 데이터 개수: {len(test_df)}")
print("-" * 30)
print(train_df['label'].value_counts())


# 3. 전처리 시작
print("--- 전처리 시작 ---")
tokenized_train, tokenized_test, tokenizer = process_data(train_df, test_df)

# 결과 확인
print("\n첫 번째 데이터 반환 결과")
print(f"원문: {train_df.iloc[0]['text']}")
print(f"변환된 숫자(input_ids): {tokenized_train[0]['input_ids'][:10]}...")
print(f"어텐션 마스크: {tokenized_train[0]['attention_mask'][:10]}...")
print("--- 전처리 완료 ---")
