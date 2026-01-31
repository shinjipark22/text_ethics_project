# ⚖️ Text Ethics Classification Project (AI 윤리 검증 모델)

이 프로젝트는 **비윤리적인 문장(Immoral Sentences)**을 탐지하기 위해 BERT 기반의 사전 학습된 모델(Pre-trained Model)을 파인튜닝(Fine-tuning)하는 파이프라인입니다.

Hugging Face `transformers`, `datasets`와 `PyTorch`를 사용하여 데이터 로드부터 전처리, 학습, 평가, 모델 저장까지의 전체 과정을 수행합니다.

## 📂 폴더 구조 (Directory Structure)

```text
text_ethics_project/
├── data/
│   ├── train/             # 학습용 JSON 데이터들이 위치 (.json)
│   └── test/              # 테스트용 JSON 데이터들이 위치 (.json)
├── models/                # 학습 완료된 모델이 저장되는 곳 (자동 생성)
├── src/
│   ├── main.py            # 실행 진입점 (Main Entry Point)
│   ├── data_loader.py     # JSON 데이터 파싱 및 로드
│   ├── processor.py       # 텍스트 토크나이징 (HF Dataset 변환)
│   ├── dataset.py         # PyTorch Dataset 클래스 정의
│   ├── model.py           # BERT 모델 초기화 및 로드
│   └── trainer.py         # 학습 루프 및 검증 (Training Loop)
└── README.md
```

## 🛠️ 요구 사항 (Requirements)

이 프로젝트를 실행하기 위해 필요한 주요 라이브러리입니다.
(`processor.py`에서 `datasets` 라이브러리를 사용하므로 꼭 설치해야 합니다.)

* Python 3.8+
* PyTorch (CUDA 권장)
* Transformers
* Datasets (Hugging Face)
* Pandas
* Scikit-learn
* Tqdm

```bash
pip install torch transformers datasets pandas scikit-learn tqdm
```

## 🚀 실행 방법 (Usage)

프로젝트 루트 경로에서 아래 명령어를 실행하면 학습이 시작됩니다.

```bash
python src/main.py
```

실행 시 다음과 같은 작업이 순차적으로 진행됩니다.

1.  **GPU 확인**: CUDA 사용 가능 여부를 확인하고 RTX 4060 등 GPU 정보를 출력합니다.
2.  **데이터 로드**: `data/train` 및 `data/test` 폴더의 모든 JSON 파일을 읽어옵니다.
3.  **전처리**: 설정된 모델(`bert-base-multilingual-cased`)의 Tokenizer로 텍스트를 변환합니다.
4.  **학습(Train)**: 설정된 Epoch(기본 3)만큼 학습을 진행하며 Loss를 출력합니다.
5.  **평가(Validation)**: Epoch마다 정확도(Accuracy)와 F1-Score를 계산합니다.
6.  **저장(Save)**: 학습된 모델과 토크나이저를 `./models/프로젝트명_모델명` 폴더에 자동 저장합니다.

## ⚙️ 설정 변경 (Configuration)

### 모델 변경하기
다른 모델(예: KcBERT, RoBERTa 등)로 실험하고 싶다면 `src/main.py` 파일의 상단 변수를 수정하세요.

```python
# src/main.py

def main():
    PROJECT_NAME = "text_ethics"
    
    # 원하는 모델명으로 변경 (예: "beomi/kcbert-base", "klue/roberta-base")
    MODEL_NAME = "bert-base-multilingual-cased" 
    ...
```

### 학습 파라미터 변경
`src/main.py`의 `train_model` 호출 부분에서 에폭(Epochs)과 배치 크기(Batch Size)를 조절할 수 있습니다.

```python
    model = train_model(
        ...,
        epochs=3,       # 학습 반복 횟수
        batch_size=32   # 메모리 부족 시 16 또는 8로 감소
    )
```

## 📄 소스 코드 설명

| 파일명 | 설명 |
|---|---|
| **`main.py`** | 전체 파이프라인을 총괄하는 컨트롤 타워입니다. 데이터 로드, 처리, 학습, 저장을 순서대로 실행합니다. |
| **`data_loader.py`** | 복잡한 JSON 구조에서 `text`와 `is_immoral` 라벨을 추출하여 Pandas DataFrame으로 변환합니다. |
| **`processor.py`** | Hugging Face의 `Dataset` 객체로 변환하고, `AutoTokenizer`를 사용해 고속으로 토크나이징합니다. |
| **`dataset.py`** | 토크나이징된 데이터를 PyTorch 모델에 주입할 수 있도록 텐서(Tensor) 형태로 포장하여 반환합니다. |
| **`model.py`** | `AutoModelForSequenceClassification`을 사용하여 모델을 로드하고, `Clean(0)`/`Immoral(1)` 라벨을 설정합니다. |
| **`trainer.py`** | 실제 학습이 일어나는 곳입니다. `AdamW` 옵티마이저, 스케줄러, F1-Score 계산 등을 담당합니다. |

## 📊 데이터셋 형식 (Data Format)

학습 데이터(`data/train/*.json`)는 아래와 같은 구조를 가져야 합니다.

```json
[
  {
    "sentences": [
      {
        "text": "이 문장은 예시입니다.",
        "is_immoral": false,
        "types": []
      },
      {
        "text": "비윤리적인 문장 예시...",
        "is_immoral": true,
        "types": ["CENSURE", "HATRED"]
      }
    ]
  }
]
```

---
**Author:** Shinji Park  
**Last Updated:** 2026.01.31