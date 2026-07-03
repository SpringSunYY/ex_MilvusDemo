package com.yy.milvus.service.feature;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.ExecutorService;

/**
 * 特征提取器统一接口。
 *
 * <p>激活方式（仅此一处）：
 * <ul>
 *   <li>{@link org.springframework.boot.autoconfigure.condition.ConditionalOnProperty}
 *       读取 yml 的 {@code embedding.model} 单字段（如 clip / dino / siglip），
 *       单进程只实例化一个 FeatureExtractor bean</li>
 *   <li>不同模型的独立参数放在 {@code embedding.{model}} 子节点下</li>
 *   <li>不同模型对应不同的 Milvus collection，命名由 {@link #getModelName()} 拼接</li>
 * </ul>
 *
 * <p>扩展新模型：
 * <ol>
 *   <li>在 {@code EmbeddingProperties} 中追加该模型的 Properties 子类</li>
 *   <li>实现 {@code FeatureExtractor}，bean 上加
 *       {@code @ConditionalOnProperty(name="embedding.model", havingValue="xxx")}</li>
 *   <li>yml 的 {@code embedding.model} 改成新模型 id 即可启用</li>
 * </ol>
 */
public interface FeatureExtractor {

    /**
     * 从图片文件提取特征向量
     *
     * @param imageFile  图片文件
     * @return 归一化后的特征向量，维度由 getFeatureDim() 返回
     * @throws IOException 文件不存在、无法解码或推理失败
     */
    float[] extractFeature(File imageFile) throws IOException;

    /**
     * 从输入流提取特征向量（不落盘，适合上传文件）
     *
     * @param inputStream 图片数据流
     * @return 归一化后的特征向量
     * @throws IOException 无法解码或推理失败
     */
    float[] extractFeature(InputStream inputStream) throws IOException;

    /**
     * 从字节数组提取特征向量（适合上传文件等场景）
     *
     * @param imageBytes 图片字节数据
     * @return 归一化后的特征向量
     * @throws IOException 无法解码或推理失败
     */
    float[] extractFeature(byte[] imageBytes) throws IOException;

    /**
     * 可选：把特征提取时传入线程池。
     * <p>
     * 对于多尺度 / HFlip 等"同一张图需要多次 ONNX 推理"的模型，传入 ExecutorService 后可以让多次推理
     * 真正并发执行（每次推理借不同 session），单张图耗时 ≈ max 而不是 sum。
     * <p>
     * 调用方约定：
     * <ul>
     *   <li>传 {@code null} → 不并发（与不带 executor 的版本等价）</li>
     *   <li>传非空 → 调用方负责保证容量充足，否则实现内部可能同步等待 session</li>
     * </ul>
     * <p>
     * 默认实现把任务转发到无 executor 的版本，等价于不并发；
     * 支持并发的模型（如 DINOv2）应当 override 此方法。
     */
    default float[] extractFeature(File imageFile, ExecutorService exec) throws IOException {
        return extractFeature(imageFile);
    }

    /**
     * 同 {@link #extractFeature(File, ExecutorService)}，但数据来自输入流
     */
    default float[] extractFeature(InputStream inputStream, ExecutorService exec) throws IOException {
        return extractFeature(inputStream);
    }

    /**
     * 同 {@link #extractFeature(File, ExecutorService)}，但数据来自字节数组
     */
    default float[] extractFeature(byte[] imageBytes, ExecutorService exec) throws IOException {
        return extractFeature(imageBytes);
    }

    /**
     * 特征向量维度。
     * 必须与 Milvus collection 的 dim 字段一致，启动期会校验。
     */
    int getFeatureDim();

    /**
     * 模型标识符，用于 Milvus collection 命名（拼到 collection-name 后）和请求路由。
     * 返回值应与 application.yml 中 embedding.model 的字段值一致（不区分大小写）。
     *
     * <p>默认从类名推断：{@code ClipFeatureExtractor → clip}，
     * 依赖类名以 "FeatureExtractor" 结尾的约定。
     */
    default String getModelName() {
        return this.getClass().getSimpleName().replace("FeatureExtractor", "").toLowerCase();
    }

    /**
     * 模型展示名（用于 UI 显示）。
     * 默认与 {@link #getModelName()} 一致，不做大小写转换。
     */
    default String getModelDisplayName() {
        String n = getModelName();
        return n == null ? "" : n;
    }

    /**
     * 模型简介（用于 UI 显示一行说明）。默认空，前端拿到也不渲染。
     */
    default String getModelDescription() {
        return "";
    }

    /**
     * Session 池快照：供性能日志展示利用率。
     * 默认返回零值（无池或不可观测的实现）。
     *
     * @return 数组 [poolSize, available, peakBorrowed]
     */
    default int[] getSessionPoolSnapshot() {
        return new int[] { 0, 0, 0 };
    }
}
