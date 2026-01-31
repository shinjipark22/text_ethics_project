from transformers import AutoModelForSequenceClassification

def get_model(model_name="klue/bert-base", num_labels=2):
    """
    설명: 한국어 BERT 모델을 불러오고, 그 위에 문장 분류용 머리(Head)를 얹어주는 함수
    """

    # 1. 모델이 내뱉는 숫자 (0,1)에 이름표를 달아줌
    # 나중에 모델이 1이라고 하면 Immoral이라고 사람이 읽기 편하게 하기 위함
    id2label = {0: "Clean", 1: "Immoral"}
    label2id = {"Clean": 0, "Immoral": 1}

    # 2. AutoModel 시리즈 중 '문장 분류용' 클래스를 사용해서 모델 불러옴
    # from_pretrained 메서드 사용
    print(f"모델 로드 중: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels, # 맞춰야할 정답 개수 (2개)
        id2label=id2label,
        label2id=label2id
    )

    return model