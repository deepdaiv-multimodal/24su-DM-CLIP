import json
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                line = line.strip()
                if line:
                    json_obj = json.loads(line)
                    data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_number} in {file_path}: {e}")
            except Exception as e:
                print(f"Unexpected error on line {line_number} in {file_path}: {e}")
    return data

def process_data(data, model_name):
    processed = {}
    for item in data:
        dataset = item['dataset'].split('/')[-1]
        metrics = item['metrics']
        if 'image_retrieval_recall@5' in metrics:
            processed[dataset] = metrics['image_retrieval_recall@5']
        elif 'text_retrieval_recall@5' in metrics:
            processed[dataset] = metrics['text_retrieval_recall@5']
        elif 'acc5' in metrics:
            processed[dataset] = metrics['acc5']
        elif 'mean_average_precision' in metrics:
            processed[dataset] = metrics['mean_average_precision']
        else:
            print(f"No suitable metric found for dataset {dataset} in model {model_name}")
    return {model_name: processed}

def visualize_data(all_data, selected_datasets, models):
    plot_data = {model: [all_data[model].get(dataset, 0) for dataset in selected_datasets] for model in models if model in all_data}

    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2*np.pi, len(selected_datasets), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for model, color in zip(models, colors):
        if model in plot_data:  # 모델 데이터 존재 확인
            values = plot_data[model]
            values = np.concatenate((values, [values[0]]))
            ax.plot(angles, values, 'o-', linewidth=2, color=color, label=model)
            ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(selected_datasets, fontsize=16)
    ax.set_ylim(0, 1.0)  # y축 범위를 0~1.0 (100%)로 설정
    ax.set_yticks([0.5, 1.0])  # y축 눈금을 50%와 100%로 설정
    ax.set_yticklabels(['50%', '100%'])  # y축 레이블을 50%와 100%만 표시

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=18)  # 모델명 글씨 크기를 크게 설정
    plt.tight_layout()
    plt.show()

    output_filename = 'plot.png'
    plt.savefig(output_filename)  # plt.show() 대신 plt.savefig() 사용
    print(f"Plot saved to {output_filename}")  # 저장 메시지 출력

    plt.close(fig)

# --- 메인 실행 부분 ---
models = ['mobileclip_s1_3m', 'mobilemclip_s1_3m']

# 모든 데이터셋을 포함하도록 수정
selected_datasets = ['stl10', 'cifar10', 'clevr_count_all', 'dmlab', 'eurosat', 'smallnorb_label_elevation', "pets", "voc2007_multilabel", 'clevr_closest_object_distance']
all_data = {}

for model in models:
    file_path = f'results/{model}.jsonl'
    try:
        data = load_data(file_path)
        all_data.update(process_data(data, model))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

visualize_data(all_data, selected_datasets, models)
