from datasets import Dataset
from transformers import AutoTokenizer

def process_data(train_df, test_df, model_name='klue/bert-base'):
    # 1. 한국어 BERT 전용 토크나이저 불러오기
    print(f"tokenizer 로드 중: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Pandas를 HuggingFace Dataset으로 변환
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 3. 토크나이징 함수 정의
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length", # 길이 맞추기 위한 패딩 적용
            truncation=True, # max_length 넘으면 자름 
            max_length=128
        )
    
    # 4. 전체 데이터셋에 한꺼번에 적용
    print("문장들을 숫자로 변환하는 Tokenizing 실행 중...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    return tokenized_train, tokenized_test, tokenizer