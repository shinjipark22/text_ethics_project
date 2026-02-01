# ⚖️ Text Ethics Classification Project (AI 윤리 검증 모델)

이 프로젝트는 비윤리적인 문장(Immoral Sentences)을 탐지하기 위해 BERT 기반의 사전 학습된 모델(Pre-trained Model)을 파인튜닝(Fine-tuning)하는 파이프라인입니다.

Hugging Face `transformers`, `datasets`와 `PyTorch`를 사용하여 데이터 로드부터 전처리, 학습, 평가, 모델 저장, 그리고 Hugging Face Hub 자동 업로드까지의 전체 과정을 수행합니다.

## 📂 폴더 구조 (Directory Structure)

```text
text_ethics_project/
├── .env                   # [중요] HF 토큰 및 아이디 저장 (Git 업로드 X)
├── .gitignore             # 보안 파일(.env) 및 데이터 제외 설정
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
(환경 변수 관리 및 HF Hub 업로드를 위한 라이브러리가 추가되었습니다.)

* Python 3.8+
* PyTorch (CUDA 권장)
* Transformers
* Datasets (Hugging Face)
* Hugging Face Hub (모델 업로드)
* Python-dotenv (환경 변수 로드)
* Pandas
* Scikit-learn
* Tqdm

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

프로젝트 루트 경로에서 아래 명령어를 실행하면 학습이 시작됩니다.

```bash
python src/main.py
```

실행 시 다음과 같은 작업이 순차적으로 진행됩니다.

1.  **환경 설정 로드**: `.env` 파일을 읽어 Hugging Face에 로그인합니다.
2.  **GPU 확인**: CUDA 사용 가능 여부를 확인하고 RTX 4060 등 GPU 정보를 출력합니다.
3.  **데이터 로드**: `data/train` 및 `data/test` 폴더의 모든 JSON 파일을 읽어옵니다.
4.  **전처리**: 설정된 모델(`bert-base-multilingual-cased`)의 Tokenizer로 텍스트를 변환합니다.
5.  **학습(Train)**: 설정된 Epoch(기본 3)만큼 학습을 진행하며 Loss를 출력합니다.
6.  **평가(Validation)**: Epoch마다 정확도(Accuracy)와 F1-Score를 계산합니다.
7.  **저장(Save)**: 학습된 모델과 토크나이저를 `./models/` 폴더에 로컬 저장합니다.
8.  **업로드(Upload)**: 학습된 모델을 **Hugging Face Hub**의 본인 계정 리포지토리로 자동 업로드합니다.

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
| **`main.py`** | 전체 파이프라인을 총괄하는 컨트롤 타워입니다. 환경 변수 로드부터 학습, HF 업로드까지 수행합니다. |
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
