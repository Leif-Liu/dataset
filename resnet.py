import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, optimizers, losses, metrics
import matplotlib.pyplot as plt
import os


class BasicBlock(tf.keras.Model):
    def __init__(self, in_channels, out_channels, strides=1):
        super(BasicBlock, self).__init__()
        
        # 保存构造参数用于序列化
        self.in_channels_config = in_channels
        self.out_channels_config = out_channels
        self.strides_config = strides
        
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

    def get_config(self):
        """返回模型配置，用于序列化"""
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels_config,
            'out_channels': self.out_channels_config,
            'strides': self.strides_config
        })
        return config

    @classmethod
    def from_config(cls, config):
        """从配置重建模型"""
        in_channels = config.pop('in_channels')
        out_channels = config.pop('out_channels')
        strides = config.pop('strides', 1)
        return cls(in_channels, out_channels, strides)


class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10):  # 修改默认类别数
        super(ResNet, self).__init__()
        self.in_channels = 64  # 修复初始通道数
        
        # 保存构造参数用于序列化
        self.block_class = block
        self.num_blocks_config = num_blocks
        self.num_classes_config = num_classes

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

    def get_config(self):
        """返回模型配置，用于序列化"""
        config = super().get_config()
        
        # 获取block类型名称
        block_type = getattr(self.block_class, '__name__', 'BasicBlock')
        
        config.update({
            'block_type': block_type,
            'num_blocks': self.num_blocks_config,
            'num_classes': self.num_classes_config,
            'in_channels': 64  # 初始通道数（固定值）
        })
        return config

    @classmethod
    def from_config(cls, config):
        """从配置重建模型"""
        block_type = config.pop('block_type', 'BasicBlock')
        num_blocks = config.pop('num_blocks', [2, 2, 2, 2])
        num_classes = config.pop('num_classes', 10)
        config.pop('in_channels', None)  # 移除不需要的配置项
        
        # 目前只支持BasicBlock
        if block_type == 'BasicBlock':
            return cls(BasicBlock, num_blocks, num_classes)
        else:
            raise ValueError(f"不支持的block类型: {block_type}")


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
    
    # 1. 保存完整模型（推荐格式）- SavedModel格式（为Netron优化）
    savedmodel_path = os.path.join(save_dir, f"{model_name}_savedmodel")
    
    # 为子类化模型创建明确的输入签名，确保Netron能正确解析
    try:
        # 确保模型已经构建
        dummy_input = tf.random.normal((1, 32, 32, 3))
        _ = model(dummy_input, training=False)
        
        # 创建具体的推理函数
        @tf.function
        def inference_func(x):
            return model(x, training=False)
        
        # 获取concrete function并指定输入签名
        concrete_func = inference_func.get_concrete_function(
            tf.TensorSpec(shape=(None, 32, 32, 3), dtype=tf.float32, name='input_image')
        )
        
        # 首先保存标准的Keras SavedModel（包含元数据）
        model.save(savedmodel_path, save_format='tf')
        print(f"✓ SavedModel格式已保存到: {savedmodel_path}")
        print(f"  ✓ 包含Keras元数据，可用tf.keras.models.load_model加载")
        
        # 另外保存Netron优化版本（使用固定batch size展开计算图）
        netron_path = os.path.join(save_dir, f"{model_name}_netron_savedmodel")
        
        # 创建固定batch size的推理函数（这样能完全展开计算图）
        @tf.function
        def fixed_inference_func(x):
            return model(x, training=False)
        
        # 使用固定形状的输入（batch_size=1）来完全展开图
        fixed_concrete_func = fixed_inference_func.get_concrete_function(
            tf.TensorSpec(shape=(1, 32, 32, 3), dtype=tf.float32, name='input_image')
        )
        
        tf.saved_model.save(
            model, 
            netron_path,
            signatures={
                'serving_default': fixed_concrete_func,
                'inference': fixed_concrete_func
            }
        )
        print(f"✓ Netron优化版本已保存到: {netron_path}")
        print(f"  ✓ 使用固定batch size展开计算图，适合Netron可视化")
        
        # 额外创建冻结图版本（最适合Netron）
        try:
            frozen_path = os.path.join(save_dir, f"{model_name}_frozen.pb")
            
            # 获取具体的函数
            full_model = tf.function(lambda x: model(x, training=False))
            full_model = full_model.get_concrete_function(
                tf.TensorSpec(shape=(1, 32, 32, 3), dtype=tf.float32)
            )
            
            # 冻结图
            from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
            frozen_func = convert_variables_to_constants_v2(full_model)
            
            # 保存冻结的图
            tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                            logdir=save_dir,
                            name=f"{model_name}_frozen.pb",
                            as_text=False)
            
            print(f"✓ 冻结图已保存到: {frozen_path}")
            print(f"  ✓ 这是最适合Netron可视化的格式")
            
        except Exception as e:
            print(f"⚠ 冻结图保存失败: {e}")
        
    except Exception as e:
        print(f"⚠ 签名保存失败，使用标准方式: {e}")
        model.save(savedmodel_path, save_format='tf')
        print(f"✓ SavedModel格式已保存到: {savedmodel_path}")
    
    # 2. 保存为Keras原生格式（推荐替代H5）
    keras_path = os.path.join(save_dir, f"{model_name}.keras")
    try:
        model.save(keras_path, save_format='keras')
        print(f"✓ Keras格式已保存到: {keras_path}")
    except Exception as e:
        print(f"⚠ Keras格式保存失败: {e}")
        # 对于子类模型，尝试使用SavedModel格式
        print("  尝试使用SavedModel格式作为替代...")
    
    # 3. 只保存权重（这个总是可以工作的）
    weights_path = os.path.join(save_dir, f"{model_name}_weights.h5")
    model.save_weights(weights_path)
    print(f"✓ 权重文件已保存到: {weights_path}")
    
    # 4. 保存模型配置到JSON文件（用于重建模型架构）
    try:
        config_path = os.path.join(save_dir, f"{model_name}_config.json")
        model_config = {
            'model_type': 'ResNet18',
            'num_classes': model.linear.units,
            'input_shape': [32, 32, 3],
            'description': f'ResNet-18 model trained on CIFAR-10, saved at {model_name}'
        }
        
        import json
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        print(f"✓ 模型配置已保存到: {config_path}")
    except Exception as e:
        print(f"⚠ 模型配置保存失败: {e}")
    
    # 5. 保存为TensorFlow Lite格式（用于移动端部署）
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
    
    # 6. 导出为ONNX格式（可选，需要tf2onnx库）
    try:
        import tf2onnx
        onnx_path = os.path.join(save_dir, f"{model_name}.onnx")
        
        # 使用from_keras方法进行ONNX转换
        # 首先确保模型已经构建
        dummy_input = tf.random.normal((1, 32, 32, 3))
        _ = model(dummy_input, training=False)
        
        # 定义输入规格
        spec = (tf.TensorSpec((None, 32, 32, 3), tf.float32, name="input"),)
        
        # 直接从Keras模型转换为ONNX
        tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            output_path=onnx_path
        )
        
        print(f"✓ ONNX格式已保存到: {onnx_path}")
    except ImportError:
        print("⚠ tf2onnx未安装，跳过ONNX格式导出")
    except Exception as e:
        print(f"⚠ ONNX转换失败: {e}")
    
    print(f"\n模型保存完成！所有文件都在 {save_dir} 目录中")


def create_model_from_config(config_path):
    """
    从配置文件创建模型
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        创建的模型
    """
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if config['model_type'] == 'ResNet18':
        model = ResNet18(num_classes=config['num_classes'])
        # 构建模型
        dummy_input = tf.random.normal((1, *config['input_shape']))
        _ = model(dummy_input, training=False)
        return model
    else:
        raise ValueError(f"不支持的模型类型: {config['model_type']}")


def load_saved_model(model_path):
    """
    加载保存的模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        加载的模型
    """
    print(f"正在加载模型: {model_path}")
    
    try:
        if model_path.endswith('.keras'):
            # 加载Keras原生格式模型
            model = tf.keras.models.load_model(model_path)
        elif model_path.endswith('.h5'):
            # 加载H5格式模型
            model = tf.keras.models.load_model(model_path)
        elif os.path.isdir(model_path):
            # 加载SavedModel格式
            model = tf.keras.models.load_model(model_path)
        elif model_path.endswith('_weights.h5'):
            # 如果是权重文件，需要重建模型架构
            print("检测到权重文件，正在重建模型架构...")
            model = ResNet18(num_classes=10)  # 默认CIFAR-10
            
            # 构建模型（通过一次前向传播）
            dummy_input = tf.random.normal((1, 32, 32, 3))
            _ = model(dummy_input, training=False)
            
            # 加载权重
            model.load_weights(model_path)
            print("✓ 权重加载成功")
        else:
            raise ValueError(f"不支持的模型格式: {model_path}")
        
        print("✓ 模型加载成功")
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        
        # 如果加载失败，尝试从权重文件重建
        base_path = model_path.replace('.keras', '').replace('.h5', '').replace('_savedmodel', '')
        weights_path = f"{base_path}_weights.h5"
        
        if os.path.exists(weights_path):
            print(f"尝试从权重文件重建模型: {weights_path}")
            return load_saved_model(weights_path)
        else:
            raise e


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
    
    # 先进行一次预测来判断输出格式
    sample_prediction = model.predict(x_test[:1], verbose=0)
    num_classes = sample_prediction.shape[-1] if len(sample_prediction.shape) > 1 else 1
    
    print(f"模型输出形状: {sample_prediction.shape}")
    print(f"检测到类别数: {num_classes}")
    
    # 根据输出格式选择合适的损失函数和标签格式
    if num_classes > 1:
        # 多分类输出，使用分类交叉熵
        y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes)
        
        # 重新编译模型以确保正确的损失函数
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        test_loss, test_accuracy = model.evaluate(x_test, y_test_categorical, verbose=0)
    else:
        # 二分类或回归输出
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"加载模型测试结果:")
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_accuracy:.4f}")
    
    # 预测几个样本
    print("\n预测前5个测试样本:")
    predictions = model.predict(x_test[:5], verbose=0)
    
    if num_classes > 1:
        predicted_classes = np.argmax(predictions, axis=1)
    else:
        predicted_classes = predictions.flatten().astype(int)
    
    true_classes = y_test[:5].flatten()
    
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i in range(5):
        if num_classes > 1:
            confidence = np.max(tf.nn.softmax(predictions[i]))
        else:
            confidence = predictions[i][0] if len(predictions[i].shape) > 0 else predictions[i]
            
        print(f"样本 {i+1}: 真实类别={cifar10_classes[true_classes[i]]}, "
              f"预测类别={cifar10_classes[predicted_classes[i]]}, "
              f"置信度={confidence:.3f}")
    
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
        epochs=1,
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
        epochs=2,  # 较少的epoch用于快速测试
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
            test_saved_model("saved_models/resnet18_full_training_savedmodel")
            
    elif choice == "2":
        model, history = simple_train_example()
        print("训练完成！")
        
        # 询问是否测试保存的模型
        test_choice = input("\n是否测试刚保存的模型？(y/n): ")
        if test_choice.lower() == 'y':
            test_saved_model("saved_models/resnet18_simple_training_savedmodel")
            
    elif choice == "3":
        print("\n可用的模型文件:")
        save_dir = "saved_models"
        if os.path.exists(save_dir):
            model_files = [f for f in os.listdir(save_dir) if f.endswith('.keras') or f.endswith('_savedmodel') or f.endswith('_weights.h5')]
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