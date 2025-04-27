import os
import yaml
import shutil
from tqdm import tqdm

def unify_labels(dataset_dir, output_dir, label_mapping):
    """
    将 YOLO 数据集的标签统一为一个类别，并更新 data.yaml。

    Args:
        dataset_dir (str): YOLO 数据集的根目录。
        output_dir (str): 统一后的数据集输出目录。
        label_mapping (dict): 标签映射字典，例如 {"basketball": "ball", "tennisball": ""}
    """
    # 1. 读取 data.yaml 文件
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(data_yaml_path):
        print(f"错误：找不到 data.yaml 文件: {data_yaml_path}")
        return

    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # 2. 获取类别名称并创建新的类别映射
    original_names = data.get('names', [])
    if not original_names:
        print("错误:data.yaml 文件中缺少 'names' 列表。")
        return

    # 过滤掉值为空或仅包含空白字符的标签映射
    label_mapping = {k: v.strip() for k, v in label_mapping.items() if v and v.strip()}
    
    # 创建新的类别列表和映射关系
    new_names = sorted(set(v for v in label_mapping.values() if v))
    name_to_new_id = {}
    for i, name in enumerate(original_names):
        mapped_name = label_mapping.get(name)
        if mapped_name and mapped_name.strip():
            name_to_new_id[name] = new_names.index(mapped_name)

    print(f"原始类别: {original_names}")
    print(f"新的类别: {new_names}")
    print(f"类别映射: {name_to_new_id}")

    # 3. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    image_output_dir = os.path.join(output_dir, "images")
    label_output_dir = os.path.join(output_dir, "labels")
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    # 4. 复制图像文件并转换标签
    subsets = ["train", "val", "test"]
    for subset in subsets:
        image_dir = os.path.join(dataset_dir, subset, "images")
        label_dir = os.path.join(dataset_dir, subset, "labels")

        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"警告：缺少 {subset} 文件夹，跳过...")
            continue

        print(f"处理 {subset} 子集...")
        for image_file in tqdm(os.listdir(image_dir), desc=f"复制 {subset} 数据"):
            image_path = os.path.join(image_dir, image_file)
            label_filename = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(label_dir, label_filename)

            if not os.path.exists(label_path):
                print(f"警告：图像 {image_file} 缺少对应的标签文件，跳过...")
                continue

            # 复制图像文件
            new_image_path = os.path.join(image_output_dir, image_file)
            shutil.copy(image_path, new_image_path)

            # 转换标签文件
            new_label_path = os.path.join(label_output_dir, label_filename)
            valid_labels = []  # 存储有效的标签
            
            # 读取所有标签并过滤
            with open(label_path, 'r') as infile:
                for line in infile:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        original_id = int(parts[0])
                        original_name = original_names[original_id]
                        if original_name in name_to_new_id:  # 检查标签是否有效（未被移除）
                            new_id = name_to_new_id[original_name]
                            valid_labels.append(f"{new_id} {' '.join(parts[1:])}")
                    except (ValueError, IndexError) as e:
                        print(f"警告：处理标签文件 {label_filename} 中的行 '{line.strip()}' 时出错: {e}")

            # 如果没有有效标签，删除图像和标签文件
            if not valid_labels:
                print(f"移除无效图像及标签: {image_file}")
                os.remove(new_image_path)
                continue
            
            # 写入有效标签
            with open(new_label_path, 'w') as outfile:
                for label in valid_labels:
                    outfile.write(f"{label}\n")

    # 5. 更新 data.yaml 文件
    data['names'] = new_names
    data['nc'] = len(new_names)

    new_data_yaml_path = os.path.join(output_dir, "data.yaml")
    with open(new_data_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"已更新 data.yaml 文件: {new_data_yaml_path}")
    print("完成！")

if __name__ == "__main__":
    dataset_dir = "./external_datasets/Ball_set.v4i.yolov8"  # 替换为你的数据集路径
    output_dir = "./external_datasets/Ball_set.v4i.yolov8_no_tennis"  # 替换为输出目录
    label_mapping = {"Soccer ball": "ball", "tennis-ball": ""}  # 替换为你的标签映射
    unify_labels(dataset_dir, output_dir, label_mapping)