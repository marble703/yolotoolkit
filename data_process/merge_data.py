import os
import shutil
from pathlib import Path
import yaml  # 需要安装 PyYAML: pip install pyyaml

def merge_yolo_datasets(dataset_dirs, output_dir):
    """
    合并多个 YOLO 数据集，包括 data.yaml 文件，并调整标签 ID。
    支持 train、val、test 文件夹结构。

    Args:
        dataset_dirs (list): 包含多个数据集路径的列表。
        output_dir (str): 合并后的数据集输出路径。
    """
    all_names = []
    dataset_name_maps = {}  # 存储每个数据集的原始 ID 到全局 ID 的映射

    print("开始读取和合并类别名称...")
    # 1. 读取所有 data.yaml 并合并类别名称
    for dataset_dir in dataset_dirs:
        data_yaml_path = os.path.join(dataset_dir, "data.yaml")
        if not os.path.exists(data_yaml_path):
            print(f"警告：数据集 {dataset_dir} 缺少 data.yaml 文件，跳过类别合并。")
            continue

        try:
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f)
                original_names = data_yaml.get('names', [])
                if not original_names:
                    print(f"警告：数据集 {dataset_dir} 的 data.yaml 文件中缺少 'names' 列表。")
                    continue

                # 为当前数据集创建原始名称到原始 ID 的映射
                original_name_to_id = {name: i for i, name in enumerate(original_names)}
                dataset_name_maps[dataset_dir] = {'original_names': original_names, 'name_to_id': original_name_to_id}

                # 添加新的类别名称到全局列表
                for name in original_names:
                    if name not in all_names:
                        all_names.append(name)

        except Exception as e:
            print(f"读取 {data_yaml_path} 时出错: {e}")

    if not all_names:
        print("错误：未能从任何数据集中成功读取类别名称。无法继续合并。")
        return

    # 创建全局名称到全局 ID 的映射
    global_name_to_id = {name: i for i, name in enumerate(all_names)}
    print(f"合并后的类别列表: {all_names}")
    print(f"总类别数: {len(all_names)}")

    # 为每个数据集计算原始 ID 到全局 ID 的映射
    for dataset_dir, data in dataset_name_maps.items():
        id_map = {}
        for original_name, original_id in data['name_to_id'].items():
            if original_name in global_name_to_id:
                global_id = global_name_to_id[original_name]
                id_map[original_id] = global_id
            else:
                # 这理论上不应该发生，因为 all_names 包含了所有名称
                print(f"警告：在 {dataset_dir} 中找到的类别 '{original_name}' 未在全局列表中找到。")
        dataset_name_maps[dataset_dir]['id_map'] = id_map


    print("\n开始合并图像和标签文件...")
    # 2. 合并图像和标签文件，并调整标签 ID
    subsets = ["train", "val", "test"]
    global_counters = {"train": 0, "val": 0, "test": 0} # 为每个子集维护独立的计数器

    for subset in subsets:
        print(f"\n处理 {subset} 子集...")
        image_output_dir = os.path.join(output_dir, subset, "images")
        label_output_dir = os.path.join(output_dir, subset, "labels")

        # 创建输出文件夹
        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(label_output_dir, exist_ok=True)

        subset_image_count = 0

        for dataset_dir in dataset_dirs:
            image_dir = os.path.join(dataset_dir, subset, "images")
            label_dir = os.path.join(dataset_dir, subset, "labels")

            if not os.path.exists(image_dir) or not os.path.exists(label_dir):
                # print(f"数据集 {dataset_dir} 缺少 {subset} 的 images 或 labels 文件夹，跳过...")
                continue

            if dataset_dir not in dataset_name_maps or 'id_map' not in dataset_name_maps[dataset_dir]:
                 print(f"警告：数据集 {dataset_dir} 没有有效的类别映射，跳过 {subset} 子集处理。")
                 continue

            id_map = dataset_name_maps[dataset_dir]['id_map']

            print(f"  合并来自 {dataset_dir} 的 {subset} 数据...")
            for image_file in os.listdir(image_dir):
                image_path = os.path.join(image_dir, image_file)
                label_filename = os.path.splitext(image_file)[0] + ".txt"
                label_path = os.path.join(label_dir, label_filename)

                if not os.path.exists(label_path):
                    print(f"  警告：图像 {image_file} 缺少对应的标签文件，跳过...")
                    continue

                # 生成唯一的文件名 (使用全局计数器)
                current_count = global_counters[subset]
                # 确定文件扩展名
                _, image_ext = os.path.splitext(image_file)
                if not image_ext: image_ext = ".jpg" # 默认扩展名

                new_image_name = f"{current_count:06d}{image_ext}"
                new_label_name = f"{current_count:06d}.txt"

                new_image_path = os.path.join(image_output_dir, new_image_name)
                new_label_path = os.path.join(label_output_dir, new_label_name)

                # 复制图像文件
                try:
                    shutil.copy(image_path, new_image_path)
                except Exception as e:
                    print(f"  错误：复制图像 {image_path} 到 {new_image_path} 失败: {e}")
                    continue

                # 读取、修改并写入标签文件
                try:
                    with open(label_path, 'r') as infile, open(new_label_path, 'w') as outfile:
                        for line in infile:
                            parts = line.strip().split()
                            if not parts: continue
                            try:
                                original_id = int(parts[0])
                                if original_id in id_map:
                                    new_id = id_map[original_id]
                                    new_line = f"{new_id} {' '.join(parts[1:])}\n"
                                    outfile.write(new_line)
                                else:
                                    print(f"  警告：在标签文件 {label_filename} 中发现未知原始类别 ID {original_id}，跳过此行。")
                            except (ValueError, IndexError) as e:
                                print(f"  警告：处理标签文件 {label_filename} 中的行 '{line.strip()}' 时出错: {e}")
                except Exception as e:
                    print(f"  错误：处理标签文件 {label_path} 或写入 {new_label_path} 失败: {e}")
                    # 如果标签处理失败，可以选择删除已复制的图像
                    if os.path.exists(new_image_path):
                        os.remove(new_image_path)
                    continue # 跳过这个文件的计数增加

                global_counters[subset] += 1 # 仅在成功处理后增加计数器
                subset_image_count += 1

        print(f"{subset} 子集合并完成，共处理了 {subset_image_count} 张图像。")

   # 3. 创建最终的 data.yaml 文件
    print("\n创建合并后的 data.yaml 文件...")
    final_data_yaml_path = os.path.join(output_dir, "data.yaml")
    # 获取输出目录的绝对路径
    absolute_output_dir = os.path.abspath(output_dir)
    final_data_yaml = {
        'path': absolute_output_dir, # 添加数据集根目录的绝对路径
        'train': 'train/images',  # 相对于 path 的路径
        'val': 'val/images',      # 相对于 path 的路径
        'test': 'test/images',     # 相对于 path 的路径
        'nc': len(all_names),
        'names': all_names
    }

    try:
        with open(final_data_yaml_path, 'w', encoding='utf-8') as f:
            # 使用 sort_keys=False 保持原始顺序
            yaml.dump(final_data_yaml, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"合并后的 data.yaml 文件已保存到: {final_data_yaml_path}")
    except Exception as e:
        print(f"错误：保存最终 data.yaml 文件失败: {e}")

    print("\n所有数据集合并完成！")


if __name__ == "__main__":
    # 输入多个数据集的路径 (确保每个数据集根目录下有 data.yaml)
    dataset_dirs = [
        "./my_origindata/Augumentation/splited",                       # 自己拍的
        "./external_datasets/Ball_set.v4i.yolov8_no_tennis_reduced"    # 球，减少到600
        # "./external_datasets/smartfarm_basket.v3i.yolov8_rect",      # 篮子
        # "./external_datasets/cola.v2-cola2.yolov8"                   # 可乐
    ]

    # 合并后的输出路径
    output_dir = "./datasets/merged_dataset_0426_1" # 使用新目录以防覆盖

    merge_yolo_datasets(dataset_dirs, output_dir)
    
    print("数据集合并完成！")
