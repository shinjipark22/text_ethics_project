import json
import pandas as pd

def load_data(file_path):
    # 1. 파일을 열고 (with open...)
    with open(file_path, 'r', encoding='utf-8') as f:
        # 2. json.load()를 사용해서 데이터를 가져온다.
        json_data = json.load(f)

    rows = []
    # 3. 가져온 데이터(리스트)를 반복문(for)으로 돌면서
    for conversation in json_data:
        for sentences in conversation['sentences']:
            # 4. text와 is_immoral을 추출해서 리스트에 담는다.
            rows.append({
                'text': sentences['text'], 
                'label': 1 if sentences['is_immoral'] else 0,  # True = 1, False = 0으로 변환
                'types': sentences['types']})

    # 5. 마지막으로 pd.DataFrame()으로 감싸서 반환한다!
    df = pd.DataFrame(rows)
    return df