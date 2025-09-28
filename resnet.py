import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, optimizers, losses, metrics
import matplotlib.pyplot as plt
import os


class BasicBlock(tf.keras.Model):
    def __init__(self, in_channels, out_channels, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            out_channels,
            kernel_size=3,
            strides=strides,
            padding="same",
            use_bias=False,
        )
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            out_channels,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        # 修复快捷连接
        if strides != 1 or in_channels != out_channels:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    out_channels,
                    kernel_size=1,
                    strides=strides,
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
            ])
            self.has_shortcut = True
        else:
            self.has_shortcut = False

    def call(self, x, training=False):
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)
        if training:
            out = tf.nn.dropout(out, 0.1)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        
        # 快捷连接
        if self.has_shortcut:
            shortcut = self.shortcut(x, training=training)
        else:
            shortcut = x
            
        out = out + shortcut
        out = tf.nn.relu(out)
        
        if training:
            out = tf.nn.dropout(out, 0.1)
            
        return out


class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10):  # 修改默认类别数
        super(ResNet, self).__init__()
        self.in_channels = 64  # 修复初始通道数

        self.conv1 = tf.keras.layers.Conv2D(
            64,
            3,
            1,
            padding="same",
            use_bias=False,
        )
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avg_pool2d = tf.keras.layers.GlobalAveragePooling2D()  # 使用全局平均池化
        self.linear = tf.keras.layers.Dense(units=num_classes)  # 移除softmax，在训练时单独处理

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return tf.keras.Sequential(layers)

    def call(self, x, training=False):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)
        
        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.layer4(out, training=training)

        out = self.avg_pool2d(out)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def preprocess_data():
    """数据预处理"""
    # 加载CIFAR-10数据集
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    
    # 数据归一化
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 数据增强
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    # 转换标签为分类编码
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test), train_datagen


def plot_training_history(history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # 准确率曲线
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def save_model(model, model_name):
    """
    保存训练好的模型到多种格式
    
    Args:
        model: 训练好的TensorFlow模型
        model_name: 模型名称（不包含扩展名）
    """
    print(f"\n正在保存模型: {model_name}")
    
    # 创建保存目录
    save_dir = "saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. 保存完整模型（推荐格式）- SavedModel格式
    savedmodel_path = os.path.join(save_dir, f"{model_name}_savedmodel")
    model.save(savedmodel_path, save_format='tf')
    print(f"✓ SavedModel格式已保存到: {savedmodel_path}")
    
    # 2. 保存为H5格式（Keras格式）
    h5_path = os.path.join(save_dir, f"{model_name}.h5")
    model.save(h5_path, save_format='h5')
    print(f"✓ H5格式已保存到: {h5_path}")
    
    # 3. 只保存权重
    weights_path = os.path.join(save_dir, f"{model_name}_weights.h5")
    model.save_weights(weights_path)
    print(f"✓ 权重文件已保存到: {weights_path}")
    
    # 4. 保存为TensorFlow Lite格式（用于移动端部署）
    try:
        tflite_path = os.path.join(save_dir, f"{model_name}.tflite")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✓ TensorFlow Lite格式已保存到: {tflite_path}")
    except Exception as e:
        print(f"⚠ TensorFlow Lite转换失败: {e}")
    
    # 5. 导出为ONNX格式（可选，需要tf2onnx库）
    try:
        import tf2onnx
        onnx_path = os.path.join(save_dir, f"{model_name}.onnx")
        
        # 创建具体函数用于转换
        concrete_func = model.call.get_concrete_function(
            tf.TensorSpec(shape=(None, 32, 32, 3), dtype=tf.float32)
        )
        
        tf2onnx.convert.from_function(
            concrete_func,
            output_path=onnx_path,
            input_signature=[tf.TensorSpec(shape=(None, 32, 32, 3), dtype=tf.float32)]
        )
        print(f"✓ ONNX格式已保存到: {onnx_path}")
    except ImportError:
        print("⚠ tf2onnx未安装，跳过ONNX格式导出")
    except Exception as e:
        print(f"⚠ ONNX转换失败: {e}")
    
    print(f"\n模型保存完成！所有文件都在 {save_dir} 目录中")


def load_saved_model(model_path):
    """
    加载保存的模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        加载的模型
    """
    print(f"正在加载模型: {model_path}")
    
    if model_path.endswith('.h5'):
        # 加载H5格式模型
        model = tf.keras.models.load_model(model_path)
    elif os.path.isdir(model_path):
        # 加载SavedModel格式
        model = tf.keras.models.load_model(model_path)
    else:
        raise ValueError("不支持的模型格式")
    
    print("✓ 模型加载成功")
    return model


def test_saved_model(model_path):
    """
    测试保存的模型
    
    Args:
        model_path: 模型文件路径
    """
    print(f"\n测试保存的模型: {model_path}")
    
    # 加载模型
    model = load_saved_model(model_path)
    
    # 加载测试数据
    (_, _), (x_test, y_test) = datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    
    # 如果模型使用分类编码，转换标签
    if len(model.output.shape) > 1 and model.output.shape[-1] > 1:
        y_test_categorical = tf.keras.utils.to_categorical(y_test, 10)
        test_loss, test_accuracy = model.evaluate(x_test, y_test_categorical, verbose=0)
    else:
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"加载模型测试结果:")
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_accuracy:.4f}")
    
    # 预测几个样本
    print("\n预测前5个测试样本:")
    predictions = model.predict(x_test[:5], verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = y_test[:5].flatten()
    
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i in range(5):
        print(f"样本 {i+1}: 真实类别={cifar10_classes[true_classes[i]]}, "
              f"预测类别={cifar10_classes[predicted_classes[i]]}, "
              f"置信度={np.max(tf.nn.softmax(predictions[i])):.3f}")
    
    return model


def train_resnet():
    """训练ResNet-18模型"""
    print("正在准备数据...")
    (x_train, y_train), (x_test, y_test), train_datagen = preprocess_data()
    
    print("正在构建模型...")
    model = ResNet18(num_classes=10)
    
    # 编译模型
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # 回调函数
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=0.00001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_resnet18.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True
        )
    ]
    
    print("开始训练...")
    # 使用数据增强训练
    history = model.fit(
        train_datagen.flow(x_train, y_train, batch_size=128),
        steps_per_epoch=len(x_train) // 128,
        epochs=100,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # 评估模型
    print("\n评估模型性能...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_accuracy:.4f}")
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 保存模型
    save_model(model, "resnet18_full_training")
    
    return model, history


def simple_train_example():
    """简单训练示例（不使用数据增强）"""
    print("简单训练示例...")
    
    # 加载并预处理数据
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 构建模型
    model = ResNet18(num_classes=10)
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # 训练模型
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=10,  # 较少的epoch用于快速测试
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    # 评估
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n简单训练结果:")
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_accuracy:.4f}")
    
    # 保存模型
    save_model(model, "resnet18_simple_training")
    
    return model, history


if __name__ == "__main__":
    print("ResNet-18 CIFAR-10 训练和测试程序")
    print("=" * 50)
    print("选择操作:")
    print("1. 完整训练（数据增强 + 回调函数）")
    print("2. 简单训练（快速测试）")
    print("3. 测试已保存的模型")
    print("4. 退出")
    
    choice = input("\n请输入选择 (1-4): ")
    
    if choice == "1":
        model, history = train_resnet()
        print("训练完成！")
        
        # 询问是否测试保存的模型
        test_choice = input("\n是否测试刚保存的模型？(y/n): ")
        if test_choice.lower() == 'y':
            test_saved_model("saved_models/resnet18_full_training.h5")
            
    elif choice == "2":
        model, history = simple_train_example()
        print("训练完成！")
        
        # 询问是否测试保存的模型
        test_choice = input("\n是否测试刚保存的模型？(y/n): ")
        if test_choice.lower() == 'y':
            test_saved_model("saved_models/resnet18_simple_training.h5")
            
    elif choice == "3":
        print("\n可用的模型文件:")
        save_dir = "saved_models"
        if os.path.exists(save_dir):
            model_files = [f for f in os.listdir(save_dir) if f.endswith('.h5') or f.endswith('_savedmodel')]
            if model_files:
                for i, file in enumerate(model_files, 1):
                    print(f"{i}. {file}")
                
                try:
                    file_choice = int(input(f"\n选择要测试的模型 (1-{len(model_files)}): "))
                    if 1 <= file_choice <= len(model_files):
                        selected_file = model_files[file_choice - 1]
                        model_path = os.path.join(save_dir, selected_file)
                        test_saved_model(model_path)
                    else:
                        print("无效选择！")
                except ValueError:
                    print("请输入有效数字！")
            else:
                print("没有找到保存的模型文件")
        else:
            print("saved_models目录不存在，请先训练模型")
            
    elif choice == "4":
        print("退出程序")
    else:
        print("无效选择！")
        
    print("\n程序结束")