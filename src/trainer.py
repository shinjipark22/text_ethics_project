import torch
import numpy as np 
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import f1_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    label_flats = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train_model(model, train_dataset, val_dataset, device, tokenizer, output_dir, epochs=3, batch_size=32):

    # 1. 데이터 로더 준비
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset), # 순서를 섞어서 줌
        batch_size=batch_size
    )

    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )

    # 2. 학습 도구 설정
    # Optimizer : AdamW
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # scheduler
    # 학습 초반에는 천천히 예열, 나중에는 속도 조절
    # 전체 훈련 횟수 = (데이터 개수 / 배치 크기) * 에폭 수
    total_steps = len(train_dataloader) * epochs
    num_warmup_steps = int(total_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    ) # 시작하자마자 2e-5, 끝날 때는 속도를 점점 줄여서 0으로 만들기

    # 최고 점수 기록용 변수
    best_f1 = 0.0
    best_metrics = {}

    # 3. 에폭 반복 시작
    for epoch_i in range(0, epochs):

        print(f'\n======= Epoch {epoch_i + 1} / {epochs} =======')
        print('Training...')

        # [training mode]
        model.train()
        total_train_loss = 0

        progress_bar = tqdm(train_dataloader, desc="Training Loop")

        for step, batch in enumerate(progress_bar):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            # 기울기 초기화
            model.zero_grad()

            # Forward Pass
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels
            )

            # Loss 확인
            loss = outputs.loss
            total_train_loss += loss.item()

            # Backward Pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Parameter Update
            optimizer.step()
            scheduler.step() # 다음 스케줄로 넘어가기

            progress_bar.set_postfix({'loss': loss.item()})

        # 한 에폭 끝나면 평균 점수 출력
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"    평균 학습 오차(Loss): {avg_train_loss:.4f}")

        # [Validation mode]
        print("Running Validation...")

        model.eval() # eval mode 켜기
        
        all_predictions = []
        all_true_labels = []

        for batch in validation_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            # torch.no_grad()를 통해 메모리 절약, 속도 향상
            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )

            # 모델이 뱉은 Logits 가져오기
            logits = outputs.logits

            # GPU에 있는 데이터를 CPU로 가져와야 계산 가능
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # 가장 높은 점수를 받은 걸 정답으로 선택 (argmax)
            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()

            all_predictions.extend(pred_flat)
            all_true_labels.extend(labels_flat)

        # 4. F1-Score , Accuracy 계산
        # 정확도
        val_accuracy = np.sum(np.array(all_predictions) == np.array(all_true_labels)) / len(all_true_labels)

        # F1-Score 
        val_f1 = f1_score(all_true_labels, all_predictions, average='binary')

        print(f"    Accuracy: {val_accuracy:.4f}")
        print(f"    F1-Score: {val_f1:.4f}")

        # 최고 점수 갱신 시 자동 저장
        if val_f1 > best_f1:
            print(f"Best Score 갱신 ({best_f1:.4f} -> {val_f1:.4f}) 모델 저장 중...")
            best_f1 = val_f1
            best_metrics = {'accuracy': val_accuracy, 'f1': val_f1}

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            print(f"    (Best Score : {best_f1:.4f} 유지 - 저장 안 함)")

    print("모든 학습 과정 완료")
    return model, best_metrics