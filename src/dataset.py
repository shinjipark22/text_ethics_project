import torch
from torch.utils.data import Dataset

class EthicsDataset(Dataset):
    """
    토크나이징된 데이터와 정답 라벨을 파이토치 모델이 읽을 수 있게 포장하는 클래스
    """
    def __init__(self, encodings, labels):
        # 데이터를 받아서 저장해두는 곳(생성자)
        # encodings: {'input_ids': [[101, ...], ...], 'attention_mask': [[1, ...], ...]} 형태
        # labels: [0, 1, 0, ...] 형태의 정답 리스트
        self.encodings = encodings # 문장 (숫자 변환된 것)
        self.labels = labels # 정답 (0:Clean, 1:Immoral)

    def __getitem__(self, idx):
        # 데이터를 하나씩 꺼내주는 곳
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        # 전체 데이터의 개수를 알려줌
        return len(self.labels)
