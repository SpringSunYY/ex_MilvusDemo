# Milvus 图片搜索 Demo

基于 Milvus 向量数据库的以图搜图示例项目。

## 功能特性

- 图片特征提取与向量化
- Milvus 向量数据库存储与检索
- 以图搜图 (Image-to-Image Search)
- REST API 接口
- 命令行交互界面

## 技术栈

- Java 21
- Spring Boot 3.2
- Milvus SDK Java 2.4
- ONNX Runtime 1.18（CLIP 模型推理）
- OpenCV 4.9

## 快速开始

### 1. 下载 CLIP 模型（首次运行必做）

模型存放在 `src/main/resources/models/clip-vit-base-patch32/`，打包时随 JAR 一起发布。

```powershell
# 在项目根目录执行
.\scripts\download-clip-model.ps1
```

脚本会从 HuggingFace 下载量化版 ONNX 模型（约 150MB），无需联网运行推理。

如果下载失败，可以手动：
1. 打开 https://huggingface.co/Xenova/clip-vit-base-patch32/tree/main/onnx
2. 下载 `vision_model_quantized.onnx`（约 150MB）
3. 重命名/放到 `src/main/resources/models/clip-vit-base-patch32/model.onnx`

如果想把模型放在 JAR 外面（比如多项目共享），设环境变量：
```bash
SET CLIP_MODEL_DIR=D:\models\clip-vit-base-patch32
```

### 1. 启动 Milvus

使用 Docker 启动 Milvus:

```bash
docker run -d \
  --name milvus-etcd \
  -v ./volumes/etcd:/etcd \
  -p 2379:2379 \
  quay.io/coreos/etcd:v3.5.5 \
  etcd -advertise-client-urls=http://127.0.0.1:2379

docker run -d \
  --name milvus-minio \
  -p 9001:9001 -p 9000:9000 \
  -v ./volumes/minio:/minio_data \
  minio/minio:latest \
  server /minio_data --console-address ":9001"

docker run -d \
  --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  -v ./volumes/milvus:/var/lib/milvus \
  milvusdb/milvus:v2.4.5 \
  milvus run standalone
```

### 2. 构建项目

```bash
mvn clean package -DskipTests
```

### 3. 运行

```bash
java -jar target/MilvusDemo-1.0-SNAPSHOT.jar
```

### 4. 使用

#### 命令行界面
运行后会自动进入交互式命令行菜单:

```
╔════════════════════════════════════════════╗
║     Milvus Image Search Demo v1.0          ║
╚════════════════════════════════════════════╝

┌───────────────────菜单───────────────────┐
│  1. 索引图片 (批量导入向量数据)          │
│  2. 以图搜图                             │
│  3. 查看集合信息                         │
│  4. 重置集合 (清空所有数据)              │
│  5. 运行示例演示                         │
│  0. 退出                                 │
└─────────────────────────────────────────┘
```

#### REST API

**上传图片:**
```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:8080/api/image-search/upload
```

**以图搜图:**
```bash
curl -X POST -F "file=@/path/to/query.jpg" -F "topK=10" http://localhost:8080/api/image-search/search
```

**批量上传:**
```bash
curl -X POST -F "files=@/path/to/image1.jpg" -F "files=@/path/to/image2.jpg" http://localhost:8080/api/image-search/upload/batch
```

## 配置说明

在 `src/main/resources/application.yml` 中配置:

```yaml
server:
  port: 8080

milvus:
  host: localhost
  port: 19530
  collection-name: image_search
```

## 项目结构

```
src/main/java/com/yy/milvus/
├── MilvusDemoApplication.java     # Spring Boot 启动类
├── ImageSearchDemo.java            # 命令行演示程序
├── config/
│   └── MilvusConfig.java          # Milvus 配置
├── controller/
│   └── ImageSearchController.java  # REST API 控制器
└── service/
    ├── ImageFeatureExtractor.java  # 图片特征提取
    ├── MilvusService.java          # Milvus 服务
    └── CommandExecutor.java        # 命令行执行器
```

## 图片特征提取

使用 **CLIP ViT-B/32**（OpenAI 提出的对比语言-图像预训练模型，Apache-2.0 协议）作为本地推理模型。

- 模型：clip-vit-base-patch32 量化版 ONNX
- 推理：ONNX Runtime（CPU）
- 输出：512 维 L2 归一化的语义向量
- 优势：**能识别图像内容**（区分"粉色小羊"和"粉色小熊"），不像颜色直方图只看颜色分布

### 切换其它模型

实现 `FeatureExtractor` 接口，再加一个 `@Service` Bean 即可：

```java
public interface FeatureExtractor {
    float[] extractFeature(File imageFile) throws IOException;
    float[] extractFeature(String imagePath) throws IOException;
    float[] extractFeature(InputStream inputStream) throws IOException;
    float[] extractFeature(byte[] imageBytes) throws IOException;
    int getFeatureDim();
}
```

比如想用 **DINOv2**（更强的自监督视觉模型）：
```java
@Service
@Primary
public class DinoV2FeatureExtractor implements FeatureExtractor { ... }
```

**注意**：换模型时务必清空 Milvus 集合（不同模型的向量在不同的空间，不能混用）。

## License

MIT
