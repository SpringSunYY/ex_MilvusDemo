package com.yy.milvus.service.feature;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.slf4j.Logger;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/**
 * 三个视觉 extractor（CLIP / SigLIP / DINOv2）共享的、与具体模型无关的小工具方法。
 * <p>
 * 抽取原则：<b>只放逻辑完全等价的方法</b>（如 toRgb / l2Normalize / parseScales）。
 * 任何与具体 extractor 状态绑定的差异化逻辑（如 Dino 的 diag 诊断通道、CLIP 的特殊
 * resize 插值、SigLIP 的会话池类型）<b>不抽</b>，保留在各 extractor 内部维护。
 * <p>
 * 与 {@link ImagePreprocessor} 的区别：
 * <ul>
 *   <li>ImagePreprocessor：图片像素级处理（pad / crop / resize）</li>
 *   <li>FeatureExtractorUtils：图片之外的杂项（IO、归一化、路径解析、模型校验）</li>
 * </ul>
 */
final class FeatureExtractorUtils {

    private FeatureExtractorUtils() {}

    /**
     * L2 归一化（in-place，避免每张图 new float[]）。
     * 用 double 累加后再 sqrt，避免 float 累加误差。三个 extractor 之前各自实现一致，统一到这里。
     */
    static float[] l2Normalize(float[] v) {
        double sum = 0;
        for (float f : v) sum += (double) f * f;
        double norm = Math.sqrt(sum);
        if (norm == 0) return v;
        float inv = (float) (1.0 / norm);
        for (int i = 0; i < v.length; i++) v[i] *= inv;
        return v;
    }

    /**
     * 转为 TYPE_INT_RGB（去 alpha、统一像素布局）。
     * 已经是 TYPE_INT_RGB 直接返回 src，零拷贝。
     * <p>
     * 修复：原 Clip / SigLIP 实现里有 {@code out.createGraphics().drawImage(...); out.createGraphics().dispose();}
     * 这种"调了两次 createGraphics、第二次才 dispose"的 bug，统一为标准的 create + dispose 配对。
     */
    static BufferedImage toRgb(BufferedImage img) {
        if (img.getType() == BufferedImage.TYPE_INT_RGB) return img;
        BufferedImage out = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        try {
            g.drawImage(img, 0, 0, null);
        } finally {
            g.dispose();
        }
        return out;
    }

    /**
     * 解析多尺度配置 [192, 224, 256] → int[]。
     * 三处 extractor 之前各自写，差别只有 default 值（518 / 224 / 224）和错误日志前缀（[DINO] / [CLIP] / [SigLIP]）。
     *
     * @param list          yml 里的 scales 列表；null 或空时返回 {@code new int[]{defaultVal}}
     * @param defaultVal    默认尺度（Dino 用 518，CLIP/SigLIP 用 224）
     * @param modelTag      错误日志前缀，如 "DINO" / "CLIP" / "SigLIP"
     * @param minAllowed    scale 下限；Dino/CLIP/SigLIP 都用 32
     */
    static int[] parseScales(List<Integer> list, int defaultVal, String modelTag, int minAllowed) {
        if (list == null || list.isEmpty()) {
            return new int[]{defaultVal};
        }
        int[] result = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            int v = list.get(i);
            if (v < minAllowed) {
                throw new IllegalArgumentException("[" + modelTag + "] scales 最小值为 " + minAllowed + "，当前: " + v);
            }
            result[i] = v;
        }
        return result;
    }

    /**
     * 解析模型路径：yml 写啥就是啥，按优先级尝试四种来源。
     * 1) 绝对路径且存在 → 直接用
     * 2) classpath 相对路径（resources/ 下随 JAR 分发）
     * 3) 工作目录相对路径
     * 4) 都不存在 → 返回原路径，由调用方决定抛错文案
     *
     * @param configPath    yml 里的路径
     * @param tower         "视觉塔" / "文本塔" / "DINOv2" 等，仅用于日志
     * @param modelTag      日志前缀，如 [DINO] / [CLIP] / [SigLIP]
     */
    static Path resolveModelPath(String configPath, String tower, String modelTag, Class<?> cls) {
        if (configPath == null || configPath.isEmpty()) {
            throw new IllegalArgumentException("[" + modelTag + "] " + tower + "模型路径未配置");
        }
        Path p = Paths.get(configPath);

        if (p.isAbsolute() && Files.exists(p)) {
            return p;
        }

        try {
            String normalized = configPath.replace('\\', '/');
            var url = cls.getClassLoader().getResource(normalized);
            if (url != null) {
                return Paths.get(url.toURI());
            }
        } catch (Exception ignored) {
        }

        if (Files.exists(p)) {
            return p.toAbsolutePath();
        }

        return p.isAbsolute() ? p : p.toAbsolutePath();
    }

    /**
     * 校验模型文件存在；不存在抛带具体路径的 IOException。
     * 通用部分（路径检查 + 文件未找到提示）。<b>不抽</b> yml 字段名 / 下载链接这种 extractor-specific 提示，
     * 由各 extractor 在 catch 块自己再包一层。
     */
    static void ensureModelFileExists(Path p) throws IOException {
        if (!Files.exists(p)) {
            throw new IOException("模型文件找不到: " + p.toAbsolutePath());
        }
    }

    /**
     * 启动期 dummy 0.5 输入：把 0.5 当作"灰点"，填充到 [1, 3, H, W] 大小的 NCHW float[]。
     * Dino / CLIP / SigLIP 之前都自己写循环赋值；统一到一次方法调用。
     *
     * @param chwBuf  长度为 {@code 3 * size * size} 的 float[]
     * @param size    输入图边长（H == W）
     * @param mean    3 通道 mean（RGB）
     * @param std     3 通道 std（RGB）
     */
    static void fillDummyCenter(float[] chwBuf, int size, float[] mean, float[] std) {
        int plane = size * size;
        Arrays.fill(chwBuf, 0, plane, (0.5f - mean[0]) / std[0]);
        Arrays.fill(chwBuf, plane, plane * 2, (0.5f - mean[1]) / std[1]);
        Arrays.fill(chwBuf, plane * 2, plane * 3, (0.5f - mean[2]) / std[2]);
    }

    /**
     * 探测阶段：从 {@code OrtSession.Result} 里一个 {@link ai.onnxruntime.OnnxValue#getValue()} 的 Java 对象中，
     * 推断其有效特征维度。覆盖 Dino / CLIP / SigLIP probe 阶段遇到的全部 Java 容器类型：
     * <pre>
     *   float[][][]   → a3[0][0].length     （DINOv2 last_hidden_state）
     *   float[][]     → a[0].length          （image_embeds / 投影后特征）
     *   float[]       → fa.length            （SigLIP 单向量头）
     *   List&lt;float[]&gt;     → first.length         （旧版 ONNX Runtime 包装）
     *   List&lt;Number&gt;     → list.size            （稀疏 float 列表）
     *   List&lt;List&lt;?&gt;&gt;     → ((List&lt;?&gt;)first).size （嵌套列表）
     *   Object[] (元素为 float[]) → arr[0].length
     *   其它              → -1
     * </pre>
     * <p>
     * 同时返回 Java 类型名（用于诊断日志 "type=..." 列），避免三个 extractor 各自手写 if/else 链。
     * <p>
     * <b>只负责"从 Java 对象拿 dim"，不负责任何业务判断</b>（找哪个头、是否 throw、模型名提示等都留给 caller）。
     */
    static int inferDimFromOnnxValue(Object onnxVal) {
        if (onnxVal instanceof float[][][] a3) {
            return (a3.length > 0 && a3[0].length > 0) ? a3[0][0].length : -1;
        }
        if (onnxVal instanceof float[][] a) {
            return a.length > 0 ? a[0].length : -1;
        }
        if (onnxVal instanceof float[] fa) {
            return fa.length;
        }
        if (onnxVal instanceof List<?> l && !l.isEmpty()) {
            Object row = l.get(0);
            if (row instanceof float[] fr) return fr.length;
            if (row instanceof Number) return l.size();
            if (row instanceof List<?> inner) return inner.size();
            return -1;
        }
        if (onnxVal instanceof Object[] a && a.length > 0 && a[0] instanceof float[] fr) {
            return fr.length;
        }
        return -1;
    }

    /**
     * 配套 {@link #inferDimFromOnnxValue}：返回对应的 Java 类型名（"float[][][]" / "float[][]" /
     * "float[]" / "List" / "Object[]" / "?"），用于 probe 阶段的诊断日志。
     */
    static String onnxValueTypeName(Object onnxVal) {
        if (onnxVal instanceof float[][][]) return "float[][][]";
        if (onnxVal instanceof float[][]) return "float[][]";
        if (onnxVal instanceof float[]) return "float[]";
        if (onnxVal instanceof List<?>) return "List";
        if (onnxVal instanceof Object[]) return "Object[]";
        return "?";
    }

    /**
     * 三个视觉 extractor（CLIP / SigLIP / DINOv2）共用的 ONNX SessionOptions 构造。
     * <p>
     * 抽取范围（<b>与具体模型无关的等价部分</b>）：
     * <ol>
     *   <li>intraOp 计算公式：{@code max(scales * 2, cpuCores / sessionPoolSize)}，batch-mode 时固定为 1</li>
     *   <li>{@code setOptimizationLevel(ALL_OPT)} + {@code setIntraOpNumThreads} + {@code setInterOpNumThreads}</li>
     *   <li>启动期 batch-mode / cpuCores / pool 日志</li>
     * </ol>
     * <b>不抽的部分</b>（保留在 caller）：
     * <ul>
     *   <li>SigLIP 独有的 {@code setExecutionMode(SEQUENTIAL)}</li>
     *   <li>任何模型特定的 OptLevel / memory arena / cuda 设置</li>
     * </ul>
     *
     * @param log             extractor 自己的 slf4j logger（保证日志源仍是各 extractor 类）
     * @param modelTag        日志前缀，如 "DINO" / "CLIP" / "SigLIP"
     * @param batchMode       {@code embedding.batch-mode}，true 时 intraOp=1
     * @param scaleCount      yml 里 scales 数量（用于 {@code scales * 2} 上限）
     * @param sessionPoolSize session 池容量，用于 cpuCores / pool 的分流
     */
    static OrtSession.SessionOptions buildOrtSessionOptions(Logger log, String modelTag,
                                                             boolean batchMode,
                                                             int scaleCount,
                                                             int sessionPoolSize) {
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        int intraOp;
        int cpuCores = Runtime.getRuntime().availableProcessors();
        if (batchMode) {
            // 批量入库模式：IntraOp=1，让 session 走单线程、不抢 CPU；
            // N 张图并发时整体 CPU 占用可控，每个 session 排队等待本身不会变慢。
            intraOp = 1;
            log.info("[{} 启动] batch-mode=true → IntraOp=1（批量友好，避免抢 CPU）", modelTag);
        } else {
            // 单图搜图模式：让 session 内部多线程榨干 CPU。
            // 关键：不能超过"session 池同时全借出时的总线程数 ≤ CPU 物理核数"。
            // 否则上下文切换和锁竞争反而拖慢每张图。
            int idealIntra = Math.max(1, cpuCores / Math.max(1, sessionPoolSize));
            // 还要兼容"pool < CPU" 的情况（少数高性能场景，每个 session 多核）
            // 可以使用多个线程，因为只是查询一张图
            intraOp = Math.max(scaleCount * 2, idealIntra);
            log.info("[{} 启动] batch-mode=false → IntraOp={}（cpuCores={}，pool={}，",
                    modelTag, intraOp, cpuCores, sessionPoolSize);
        }
        try {
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            opts.setIntraOpNumThreads(intraOp);
            // InterOp>1 在 ViT 链上收益 < 2%，但线程切换成本反而增加；统一跟随 intra。
            opts.setInterOpNumThreads(intraOp);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        return opts;
    }
}