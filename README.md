# ⚖️ Text Ethics Classification Project (AI 윤리 검증 모델)

이 프로젝트는 **비윤리적인 문장(Immoral Sentences)**을 탐지하기 위해 BERT 기반의 사전 학습된 모델(Pre-trained Model)을 파인튜닝(Fine-tuning)하는 파이프라인입니다.

Hugging Face `transformers`, `datasets`와 `PyTorch`를 사용하여 데이터 로드부터 전처리, 학습, 평가, 모델 저장, 그리고 **Hugging Face Hub 자동 업로드**까지의 전체 과정을 수행합니다.

## 📂 폴더 구조 (Directory Structure)

```text
text_ethics_project/
├── .env                   # HF 토큰 및 아이디 저장 (Git 업로드 X)
├── .gitignore             # 보안 파일(.env) 및 데이터 제외 설정
├── data/
│   ├── train/             # 학습용 JSON 데이터들이 위치 (.json)
│   └── test/              # 테스트용 JSON 데이터들이 위치 (.json)
├── models/                # 학습 완료된 모델이 저장되는 곳 (자동 생성)
├── scripts/               # 유틸리티 스크립트 모음
│   ├── eval_only.py       # 학습된 모델 성능 평가
│   └── push_to_hf.py      # Hugging Face Hub 수동 업로드
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

* Python 3.8+
* PyTorch (CUDA 권장)
* Transformers
* Datasets (Hugging Face)
* Hugging Face Hub (모델 업로드)
* Python-dotenv (환경 변수 로드)
* Pandas, Scikit-learn, Tqdm

```bash
pip install torch transformers datasets huggingface_hub python-dotenv pandas scikit-learn tqdm
```

## 🔐 환경 설정 (Environment Setup)

모델을 **Hugging Face Hub**에 자동으로 업로드하기 위해, 프로젝트 최상위 경로에 `.env` 파일을 생성해야 합니다.

1. **Hugging Face Token 발급**: [Settings > Access Tokens](https://huggingface.co/settings/tokens)에서 `Write` 권한으로 토큰 생성.
2. **`.env` 파일 생성**: 프로젝트 루트에 파일을 만들고 아래 내용을 입력하세요.

```ini
# .env 파일 예시
HF_TOKEN=hf_여기에_발급받은_토큰_입력
HF_USERNAME=본인_허깅페이스_아이디
```

> **⚠️ 주의**: `.env` 파일에는 개인 토큰이 포함되어 있으므로 절대 GitHub에 올리지 마세요. (`.gitignore`에 추가 필수)

## 🚀 실행 방법 (Usage)

### 1️⃣ 학습 시작 (Training)
프로젝트 루트 경로에서 아래 명령어를 실행하면 학습이 시작됩니다.

```bash
python src/main.py
```

### 2️⃣ 모델 평가 (Evaluation)
학습된 모델을 로드하여 테스트 데이터셋에 대한 성능(F1, Accuracy)만 빠르게 확인합니다.

```bash
python scripts/eval_only.py
```

### 3️⃣ 수동 업로드 (Manual Upload)
학습 중 네트워크 오류 등으로 업로드가 실패했거나, 로컬 모델을 나중에 업로드할 때 사용합니다.

```bash
python scripts/push_to_hf.py
```

## ⚙️ 설정 변경 (Configuration)

### 모델 변경하기
다른 모델(예: KcBERT, RoBERTa 등)로 실험하고 싶다면 `src/main.py` 파일의 상단 변수를 수정하세요.
업로드 시 리포지토리 이름은 `프로젝트명-모델명` 규칙으로 자동 생성됩니다.

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
| **`src/main.py`** | 전체 파이프라인을 총괄하는 컨트롤 타워입니다. 환경 변수 로드부터 학습, HF 업로드까지 수행합니다. |
| **`src/data_loader.py`** | 복잡한 JSON 구조에서 `text`와 `is_immoral` 라벨을 추출하여 Pandas DataFrame으로 변환합니다. |
| **`src/processor.py`** | Hugging Face의 `Dataset` 객체로 변환하고, `AutoTokenizer`를 사용해 고속으로 토크나이징합니다. |
| **`src/dataset.py`** | 토크나이징된 데이터를 PyTorch 모델에 주입할 수 있도록 텐서(Tensor) 형태로 포장하여 반환합니다. |
| **`src/model.py`** | `AutoModelForSequenceClassification`을 사용하여 모델을 로드하고, `Clean(0)`/`Immoral(1)` 라벨을 설정합니다. |
| **`src/trainer.py`** | 실제 학습이 일어나는 곳입니다. `AdamW` 옵티마이저, 스케줄러, F1-Score 계산 등을 담당합니다. |
| **`scripts/eval_only.py`** | 저장된 모델을 불러와 성능(Metric)만 측정하는 스크립트 |
| **`scripts/push_to_hf.py`** | `huggingface_hub` API를 이용해 폴더 전체를 수동 업로드 |

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
**Last Updated:** 2026.02.01