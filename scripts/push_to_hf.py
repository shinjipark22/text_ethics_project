import os
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login

def push_to_hub():
    # 0. 설정 로드
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_USERNAME = os.getenv("HF_USERNAME")

    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]
    
    # 1. 경로 및 이름 설정 
    model_path = "models/text_ethics_beomi-kcbert-base"
    repo_name = "text_ethics-beomi-kcbert-base"
    
    if not HF_TOKEN or not HF_USERNAME:
        print(".env 파일에서 HF_TOKEN이나 HF_USERNAME을 찾을 수 없음")
        return

    # 2. 로그인 및 모델 로드
    login(token=HF_TOKEN)
    print(f"로컬 모델 로드 중: {model_path}")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 3. 업로드 실행
    repo_id = f"{HF_USERNAME}/{repo_name}"
    print(f"Hugging Face Hub 업로드 시작... (Target: {repo_id})")
    
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    print(f"\n업로드 완료! 주소: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    push_to_hub()