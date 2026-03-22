# examples/onnx-inference.py
"""
ONNX Runtime 推理

使用 ONNX Runtime 进行模型推理，比 PyTorch 更快
"""
import onnxruntime as ort
import cv2
import numpy as np

class ONNXDetector:
    """ONNX 模型推理器"""

    def __init__(self, onnx_path):
        # 加载模型
        self.session = ort.InferenceSession(onnx_path)

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        print(f"输入形状：{self.input_shape}")

    def preprocess(self, image, img_size=640):
        """
        预处理图片

        步骤:
        1. BGR 转 RGB
        2. 调整尺寸
        3. 归一化到 [0, 1]
        4. 添加 batch 维度
        """
        # BGR → RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整尺寸
        img_resized = cv2.resize(img_rgb, (img_size, img_size))

        # 归一化
        img_normalized = img_resized.astype(np.float32) / 255.0

        # HWC → CHW
        img_transposed = np.transpose(img_normalized, (2, 0, 1))

        # 添加 batch 维度 (1, 3, H, W)
        img_batch = np.expand_dims(img_transposed, axis=0)

        return img_batch

    def detect(self, image, conf_threshold=0.25):
        """检测图片中的目标"""
        # 预处理
        input_tensor = self.preprocess(image)

        # 推理
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # 后处理
        results = self.postprocess(outputs, conf_threshold)

        return results

    def postprocess(self, outputs, conf_threshold):
        """解析模型输出"""
        predictions = outputs[0][0]
        results = []
        for i in range(predictions.shape[1]):
            box = predictions[:4, i]
            scores = predictions[4:, i]
            if scores.max() > conf_threshold:
                class_id = scores.argmax()
                conf = scores[class_id]
                results.append({
                    'box': box,
                    'class': int(class_id),
                    'conf': float(conf)
                })
        return results

def main():
    # 加载模型
    detector = ONNXDetector('best.onnx')

    # 读取图片
    img = cv2.imread('test.jpg')

    # 检测
    results = detector.detect(img)

    print(f"检测到 {len(results)} 个目标")

if __name__ == "__main__":
    main()
