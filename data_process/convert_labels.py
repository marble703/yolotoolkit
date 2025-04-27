import os

def convert_labels(label_folder):
    """
    自动识别 YOLO 标签文件中的格式问题，并进行修正。

    Args:
        label_folder (str): YOLO 标签文件夹路径。
    """
    for root, _, files in os.walk(label_folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                corrected_lines = []

                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f"跳过无效行: {line.strip()} in {file}")
                            continue

                        try:
                            # 转换类别为整数
                            class_id = int(float(parts[0]))
                            # 转换后续四个为浮点数
                            bbox = [f"{float(coord):.6f}" for coord in parts[1:]]
                            corrected_lines.append(f"{class_id} {' '.join(bbox)}\n")
                        except ValueError as e:
                            print(f"无法解析行: {line.strip()} in {file}, 错误: {e}")

                # 写回修正后的内容
                with open(file_path, 'w') as f:
                    f.writelines(corrected_lines)
                print(f"已修正文件: {file_path}")

def convert_labels_in_all_folders(root_folder):
    """
    自动识别根目录下的所有 labels 文件夹，并修正其中的标签文件。

    Args:
        root_folder (str): 数据集的根目录。
    """
    for root, dirs, _ in os.walk(root_folder):
        for dir_name in dirs:
            if dir_name == "labels":
                label_folder = os.path.join(root, dir_name)
                print(f"正在处理标签文件夹: {label_folder}")
                convert_labels(label_folder)

if __name__ == "__main__":
    root_folder = "./my_dataset_0424/splited"  # 替换为你的数据集根目录路径
    convert_labels_in_all_folders(root_folder)