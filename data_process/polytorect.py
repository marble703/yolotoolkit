import json
import os
import shutil
def polygon_to_bounding_box(polygon):
    """
    将多边形转换为外接矩形框。

    Args:
        polygon (list): 多边形的点列表，格式为 [[x1, y1], [x2, y2], ...]。

    Returns:
        list: 外接矩形框，格式为 [x_min, y_min, x_max, y_max]。
    """
    if len(polygon) == 4:  # 如果只有四个点，直接计算外接矩形
        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        return [x_min, y_min, x_max, y_max]
    else:
        # 对于更多点的多边形，使用轮廓拟合
        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        return [x_min, y_min, x_max, y_max]

def convert_polygons_to_bounding_boxes_in_dataset(dataset_folder, output_folder):
    """
    将 YOLO 数据集中的多边形数据转换为矩形框数据，并移动对应的图像和 data.yaml 文件。

    Args:
        dataset_folder (str): 数据集的根目录，包含 train、val、test 文件夹。
        output_folder (str): 转换后保存的文件夹路径。
    """
    subsets = ["train", "val", "test"]

    # 检查并移动 data.yaml 文件
    data_yaml_path = os.path.join(dataset_folder, "data.yaml")
    if os.path.exists(data_yaml_path):
        os.makedirs(output_folder, exist_ok=True)
        shutil.copy(data_yaml_path, os.path.join(output_folder, "data.yaml"))
        print(f"已移动 data.yaml 文件到 {output_folder}")

    for subset in subsets:
        json_folder = os.path.join(dataset_folder, subset, "labels")
        image_folder = os.path.join(dataset_folder, subset, "images")
        subset_output_label_folder = os.path.join(output_folder, subset, "labels")
        subset_output_image_folder = os.path.join(output_folder, subset, "images")

        if not os.path.exists(json_folder):
            print(f"警告：{subset} 文件夹不存在，跳过...")
            continue

        os.makedirs(subset_output_label_folder, exist_ok=True)
        os.makedirs(subset_output_image_folder, exist_ok=True)

        for json_file in os.listdir(json_folder):
            if json_file.endswith('.txt'):
                json_path = os.path.join(json_folder, json_file)
                image_name = os.path.splitext(json_file)[0] + ".jpg"
                image_path = os.path.join(image_folder, image_name)

                with open(json_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                yolo_labels = []
                valid = True
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        print(f"跳过无效行: {line.strip()} in {json_file}")
                        valid = False
                        break

                    try:
                        class_id = int(parts[0])
                        points = [float(coord) for coord in parts[1:]]
                        polygon = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
                        x_min, y_min, x_max, y_max = polygon_to_bounding_box(polygon)

                        # 转换为 YOLO 格式
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        width = x_max - x_min
                        height = y_max - y_min

                        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    except ValueError as e:
                        print(f"无法解析行: {line.strip()} in {json_file}, 错误: {e}")
                        valid = False
                        break

                if valid:
                    # 保存为 YOLO 格式的标签文件
                    output_label_path = os.path.join(subset_output_label_folder, json_file)
                    with open(output_label_path, 'w', encoding='utf-8') as f:
                        f.writelines(yolo_labels)

                    # 移动对应的图像
                    if os.path.exists(image_path):
                        output_image_path = os.path.join(subset_output_image_folder, image_name)
                        shutil.copy(image_path, output_image_path)
                else:
                    print(f"跳过文件: {json_file}，因为包含无效标签。")

if __name__ == "__main__":
    dataset_folder = "./external_datasets/smartfarm_basket.v3i.yolov8"  # 替换为你的数据集根目录路径
    output_folder = "./external_datasets/smartfarm_basket.v3i.yolov8_rect"  # 替换为输出文件夹路径
    convert_polygons_to_bounding_boxes_in_dataset(dataset_folder, output_folder)