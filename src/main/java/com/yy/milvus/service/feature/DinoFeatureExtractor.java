package com.yy.milvus.service.feature;

import ai.onnxruntime.*;
import com.yy.milvus.config.EmbeddingProperties;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.extern.slf4j.Slf4j;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 基于 ONNX Runtime 的 DINOv2 图像特征提取器。
 *
 * 与 CLIP 的核心差异：
 * - DINOv2 是纯视觉自监督模型，不理解"语义类别"，只关心"视觉结构/纹理"
 * - CLIP 把"黑色格子衬衫"和"白色圆点衬衫"都归类为"衬衫"，相似度高
 * - DINOv2 能区分印花图案、格子纹理、纽扣细节等局部差异
 * - 以图搜图/商品检索场景，DINOv2 是首选
 *
 * 输入预处理（DINOv2 标准）：
 *  1. resize 到短边 518（或配置的 scales[0]）
 *  2. CenterCrop 到正方形
 *  3. 转 RGB float32 [0, 1]
 *  4. ImageNet mean/std 归一化
 *  5. NCHW
 *
 * 注意：DINOv2 对水平翻转敏感，不建议开启 hflipEnabled。
 */
@Component("dinoExtractor")
@Slf4j
@RequiredArgsConstructor
@ConditionalOnProperty(name = "embedding.model", havingValue = "dino")
public class DinoFeatureExtractor implements FeatureExtractor {

    private final EmbeddingProperties embeddingProps;

    // DINOv2 常用输入尺寸（短边）
    private static final int DEFAULT_SHORT_EDGE = 518;

    // ImageNet 归一化参数（DINOv2 用这个，CLIP 用自己的 mean/std）
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD  = {0.229f, 0.224f, 0.225f};

    private static final Deque<OrtSession> POOL = new ArrayDeque<>();
    private static volatile boolean initialized = false;
    private static final AtomicInteger INSTANCE_COUNT = new AtomicInteger(0);
    private final int instanceId = INSTANCE_COUNT.incrementAndGet();

    private OrtEnvironment env;
    private Path modelPath;
    private int[] scales;

    @PostConstruct
    public void init() throws Exception {
        if (initialized) {
            log.warn("[DINO 启动] 实例 #{} 跳过 init（static pool 已由实例 #1 初始化，pool size={}）",
                    instanceId, POOL.size());
            return;
        }

        env = OrtEnvironment.getEnvironment();
        log.info("[DINO 启动] 实例 #{} 开始初始化 ...", instanceId);

        var dino = embeddingProps.getDino();
        this.scales = parseScales(dino.getScales());

        log.info("[DINO 启动] ===== 参数确认 =====");
        log.info("[DINO 启动] featureDim={} | scales={} | hflipEnabled={}",
                dino.getFeatureDim(), Arrays.toString(this.scales), dino.isHflipEnabled());
        log.info("[DINO 启动] visionModelPath={}", dino.getVisionModelPath());
        log.info("[DINO 启动] ==========================");

        modelPath = resolveModelPath(dino.getVisionModelPath(), "DINOv2");
        ensureModelFile(modelPath, "DINOv2");

        int poolSize = resolvePoolSize(dino.getSessionPoolSize());
        log.info("[DINO 启动] 初始化 {} 个 session ...", poolSize);

        OrtSession.SessionOptions opts = buildOpts();
        for (int i = 0; i < poolSize; i++) {
            OrtSession s = env.createSession(modelPath.toString(), opts);
            POOL.addLast(s);
            log.info("[DINO 启动]   已 {}/{} 个 session 入池，当前 pool size={}", i + 1, poolSize, POOL.size());
        }
        this.initialPoolSize = POOL.size();
        log.info("[DINO 启动] session 池已就绪: {} 个", POOL.size());
        log.info("[DINO 启动] 输入: {} | 输出: {}",
                POOL.peekFirst().getInputNames(),
                POOL.peekFirst().getOutputNames());

        int poolBeforeProbe = POOL.size();
        log.info("[DINO 预热] 池预热 ...");
        int warmupErrors = 0;
        int warmupSize = scales[0]; // 用实际配置的最小尺度预热，不要用 518
        for (int i = 0; i < Math.min(POOL.size(), 1); i++) { // 只预热 1 个 session 就够了
            OrtSession s = POOL.pollFirst();
            try {
                long w0 = System.currentTimeMillis();
                float[][][][] dummy = new float[1][3][warmupSize][warmupSize];
                for (int c = 0; c < 3; c++) {
                    for (int y = 0; y < warmupSize; y++) {
                        for (int x = 0; x < warmupSize; x++) {
                            dummy[0][c][y][x] = (0.5f - MEAN[c]) / STD[c];
                        }
                    }
                }
                try (OnnxTensor tensor = OnnxTensor.createTensor(env, dummy)) {
                    Map<String, OnnxTensor> inputs = Collections.singletonMap("pixel_values", tensor);
                    try (OrtSession.Result result = s.run(inputs)) {
                        result.get(0).getValue();
                    }
                }
                long wMs = System.currentTimeMillis() - w0;
                log.info("[DINO 预热]   warmup={}ms (scale={})", wMs, warmupSize);
            } catch (Exception e) {
                warmupErrors++;
                log.warn("[DINO 预热]   warmup failed: {}", e.getMessage());
            } finally {
                POOL.addLast(s);
            }
        }
        if (warmupErrors > 0) {
            log.warn("[DINO 预热] 池预热完成，{}/{} 个失败", warmupErrors, poolSize);
        } else {
            log.info("[DINO 预热] 池预热完成");
        }
        probeHeads(POOL.peekFirst(), warmupSize);
        int poolAfterProbe = POOL.size();
        log.info("[DINO 启动] 探测后 pool size: {} -> {}", poolBeforeProbe, poolAfterProbe);

        initialized = true;
        log.info("[DINO 启动] 初始化完成，pool 最终 size={}", POOL.size());
    }

    private static int resolvePoolSize(int configured) {
        if (configured > 0) return configured;
        int cores = Math.max(1, Runtime.getRuntime().availableProcessors());
        return Math.min(cores, 8);
    }

    private static int[] parseScales(List<Integer> list) {
        if (list == null || list.isEmpty()) {
            return new int[] { DEFAULT_SHORT_EDGE };
        }
        int[] result = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            int v = list.get(i);
            if (v < 32) throw new IllegalArgumentException("[DINO] scales 最小值为 32，当前: " + v);
            result[i] = v;
        }
        return result;
    }

    private OrtSession borrowSession() throws OrtException {
        synchronized (POOL) {
            int waitCount = 0;
            while (POOL.isEmpty()) {
                waitCount++;
                if (waitCount == 1 || waitCount % 5 == 0) {
                    log.info("[DINO 池] 等待 session，空池，当前 pool size={}，等待次数={}", POOL.size(), waitCount);
                }
                try {
                    POOL.wait();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new OrtException("[DINO] 等待 session 被中断");
                }
            }
            OrtSession s = POOL.pollFirst();
            int borrowed = poolSizeEstimate() - POOL.size();
            if (borrowed > peakBorrowed) peakBorrowed = borrowed;
            log.debug("[DINO 借] pool {} -> {}", POOL.size() + 1, POOL.size());
            return s;
        }
    }

    private void returnSession(OrtSession s) {
        if (s == null) return;
        synchronized (POOL) {
            int before = POOL.size();
            POOL.addLast(s);
            int after = POOL.size();
            log.debug("[DINO 还] pool {} -> {}", before, after);
            POOL.notify();
        }
    }

    /** 池初始容量（建池时记录的；用于"借出量 = 总 - 可用"估算） */
    private int initialPoolSize = 0;
    private int poolSizeEstimate() {
        int s = initialPoolSize;
        return s > 0 ? s : POOL.size();
    }

    private OrtSession.SessionOptions buildOpts() {
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        try {
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            int threads = Math.max(1, Runtime.getRuntime().availableProcessors());
            opts.setIntraOpNumThreads(threads);
            try {
                opts.setInterOpNumThreads(threads);
            } catch (NoSuchMethodError ignore) {
            }
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        return opts;
    }

    private void ensureModelFile(Path p, String tower) throws IOException {
        if (!Files.exists(p)) {
            throw new IOException(
                "\n[DINO] " + tower + "模型文件找不到: " + p.toAbsolutePath() + "\n" +
                "[DINO] 请检查 application.yml 中 embedding.dino.vision-model-path 是否正确\n" +
                "[DINO] 推荐下载 DINOv2 ViT-B/14 ONNX:\n" +
                "[DINO]   https://huggingface.co/onnx-community/dinov2-base-ONNX\n" +
                "[DINO]   或自行导出: torch.onnx.export(dinov2_vitb14(), ...)"
            );
        }
    }

    private Path resolveModelPath(String configPath, String tower) {
        if (configPath == null || configPath.isEmpty()) {
            throw new IllegalArgumentException("[DINO] 模型路径未配置");
        }
        Path p = Paths.get(configPath);
        if (p.isAbsolute() && Files.exists(p)) {
            log.info("[DINO] {}模型使用绝对路径: {}", tower, p);
            return p;
        }
        try {
            String normalized = configPath.replace('\\', '/');
            var url = getClass().getClassLoader().getResource(normalized);
            if (url != null) {
                Path cp = Paths.get(url.toURI());
                log.info("[DINO] {}模型使用 classpath: {}", tower, cp);
                return cp;
            }
        } catch (Exception ignored) {
        }
        if (Files.exists(p)) {
            log.warn("[DINO] {}模型使用工作目录相对路径: {}", tower, p.toAbsolutePath());
            return p.toAbsolutePath();
        }
        return p.isAbsolute() ? p : p.toAbsolutePath();
    }

    @PreDestroy
    public void close() throws OrtException {
        log.info("[DINO 关闭] 开始清理 session 池，当前 pool size={}", POOL.size());
        synchronized (POOL) {
            OrtSession s;
            int count = 0;
            while ((s = POOL.pollFirst()) != null) {
                try { s.close(); count++; } catch (Exception ignored) {}
            }
            log.info("[DINO 关闭] 已关闭 {} 个 session，pool size={}", count, POOL.size());
        }
        if (env != null) env.close();
    }

    @Override
    public float[] extractFeature(File imageFile) throws IOException {
        if (imageFile == null || !imageFile.exists() || imageFile.length() == 0) {
            throw new IOException("图像文件不可读: " + (imageFile == null ? "<null>" : imageFile.getAbsolutePath()));
        }
        BufferedImage img = ImageIO.read(imageFile);
        if (img == null) {
            throw new IOException("无法解析图像: " + imageFile.getAbsolutePath());
        }
        return extractFeatureFromBufferedImage(img);
    }

    @Override
    public float[] extractFeature(InputStream inputStream) throws IOException {
        BufferedImage img = ImageIO.read(inputStream);
        if (img == null) {
            throw new IOException("无法解析图像 (InputStream)");
        }
        return extractFeatureFromBufferedImage(img);
    }

    @Override
    public float[] extractFeature(byte[] imageBytes) throws IOException {
        return extractFeatureFromBufferedImage(ImageIO.read(new java.io.ByteArrayInputStream(imageBytes)));
    }

    private float[] extractFeatureFromBufferedImage(BufferedImage image) throws IOException {
        if (image == null) {
            throw new IOException("图片读取失败");
        }
        long t0 = System.currentTimeMillis();

        BufferedImage rgb = toRgb(image);

        float[] embOriginal = null;
        float[] embFlipped = null;
        int avgLen = -1;

        try {
            embOriginal = runMultiScale(rgb);
            avgLen = embOriginal.length;

            if (embeddingProps.getDino().isHflipEnabled()) {
                BufferedImage flipped = flipHorizontal(rgb);
                embFlipped = runMultiScale(flipped);
                if (embFlipped.length == avgLen) {
                    for (int i = 0; i < avgLen; i++) {
                        embOriginal[i] = (embOriginal[i] + embFlipped[i]) * 0.5f;
                    }
                } else {
                    log.warn("[DINO HFlip] 翻转图维度 {} 与原图 {} 不一致，跳过平均", embFlipped.length, avgLen);
                }
            }
            float[] result = l2Normalize(embOriginal);
            log.info("[DINO 单图] scales={} | dim={} | 耗时 {}ms",
                    Arrays.toString(scales), result.length, System.currentTimeMillis() - t0);
            return result;

        } catch (Exception e) {
            throw new IOException("DINOv2 推理失败: " + e.getMessage(), e);
        }
    }

    /**
     * 多尺度推理：每个尺度跑一次 embedding 再平均。
     */
    private float[] runMultiScale(BufferedImage rgb) throws IOException {
        int avgLen = -1;
        float[] avg = null;
        int validScales = 0;

        for (int si = 0; si < scales.length; si++) {
            int targetShort = scales[si];
            int w = rgb.getWidth();
            int h = rgb.getHeight();
            int newW, newH;
            if (w < h) {
                newW = targetShort;
                newH = (int) Math.round((float) h * targetShort / w);
            } else {
                newH = targetShort;
                newW = (int) Math.round((float) w * targetShort / h);
            }
            BufferedImage resized = resizeDirect(rgb, newW, newH);
            BufferedImage tensorInput = centerCrop(resized, targetShort, targetShort);

            long ts0 = System.currentTimeMillis();
            try {
                float[] emb = runInference(tensorInput, targetShort);
                log.info("[DINO 尺度] scale={} | 推理耗时 {}ms | 当前 {}/{} 个尺度有效",
                        targetShort, System.currentTimeMillis() - ts0, si + 1, scales.length);
                if (avg == null) {
                    avgLen = emb.length;
                    avg = new float[avgLen];
                } else if (emb.length != avgLen) {
                    log.warn("[DINO 多尺度] 尺度 {} 维度 {} 与首尺度 {} 不一致，跳过",
                            targetShort, emb.length, avgLen);
                    continue;
                }
                validScales++;
                float scale = 1.0f / scales.length;
                for (int i = 0; i < avgLen; i++) avg[i] += emb[i] * scale;
            } catch (Exception e) {
                log.warn("[DINO 多尺度] 尺度 {} 推理失败: {}", targetShort, e.getMessage());
            }
        }

        if (avg == null || validScales == 0) {
            log.warn("[DINO 多尺度] 所有尺度都不可用，降级到单尺度");
            return runSingleScaleFallback(rgb);
        }
        return avg;
    }

    /**
     * 单次推理：从 OrtSession 池借 session，执行后归还。
     * <p>
     * tensorInput 必须是已经完成 resize/centerCrop 的目标尺寸 (targetShort × targetShort, TYPE_INT_RGB)；
     * 本方法只做 NCHW float 填充 + OnnxTensor 构造 + session.run，不再二次缩放。
     */
    private float[] runInference(BufferedImage tensorInput, int size) throws OrtException {
        OrtSession session = borrowSession();
        try {
            long t0 = System.nanoTime();
            float[][][][] chw = toNchwFloat(tensorInput, size);
            long prepNs = System.nanoTime() - t0;

            t0 = System.nanoTime();
            try (OnnxTensor tensor = OnnxTensor.createTensor(env, chw)) {
                Map<String, OnnxTensor> inputs = Collections.singletonMap("pixel_values", tensor);
                try (OrtSession.Result result = session.run(inputs)) {
                    long runNs = System.nanoTime() - t0;
                    recordPerf(prepNs, runNs);
                    return extractFirstRow(result.get(0).getValue());
                }
            }
        } finally {
            returnSession(session);
        }
    }

    // ============ 性能统计（线程安全：单次写者 - 性能侧读）============
    private final java.util.concurrent.atomic.AtomicLong perfPrepNs = new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong perfRunNs  = new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicInteger perfCalls = new java.util.concurrent.atomic.AtomicInteger();
    private volatile int peakBorrowed = 0;
    private void recordPerf(long prepNs, long runNs) {
        perfPrepNs.addAndGet(prepNs);
        perfRunNs.addAndGet(runNs);
        perfCalls.incrementAndGet();
    }

    private float[] runSingleScaleFallback(BufferedImage image) throws IOException {
        int targetShort = scales.length > 0 ? scales[0] : DEFAULT_SHORT_EDGE;
        try {
            int w = image.getWidth();
            int h = image.getHeight();
            int newW, newH;
            if (w < h) {
                newW = targetShort;
                newH = (int) Math.round((float) h * targetShort / w);
            } else {
                newH = targetShort;
                newW = (int) Math.round((float) w * targetShort / h);
            }
            BufferedImage resized = resizeDirect(image, newW, newH);
            BufferedImage cropped = centerCrop(resized, targetShort, targetShort);

            long t0 = System.currentTimeMillis();
            OrtSession session = borrowSession();
            try {
                float[][][][] chw = toNchwFloat(cropped, targetShort);
                try (OnnxTensor tensor = OnnxTensor.createTensor(env, chw)) {
                    Map<String, OnnxTensor> inputs = Collections.singletonMap("pixel_values", tensor);
                    try (OrtSession.Result result = session.run(inputs)) {
                        log.info("[DINO 降级] scale={} | 推理耗时 {}ms", targetShort, System.currentTimeMillis() - t0);
                        return extractFirstRow(result.get(0).getValue());
                    }
                }
            } catch (OrtException e) {
                throw new IOException("DINOv2 推理失败: " + e.getMessage(), e);
            } finally {
                returnSession(session);
            }
        } catch (Exception e) {
            throw new IOException("DINOv2 单尺度降级失败: " + e.getMessage(), e);
        }
    }

    /**
     * NCHW float 填充（保证语义与原 preprocess 完全一致）：
     * <ul>
     *   <li>输入必须是 targetShort × targetShort 的 RGB 图（由 runMultiScale/runSingleScaleFallback 准备好）</li>
     *   <li>读像素顺序：y 外、x 内（与原实现一致）</li>
     *   <li>归一化：ImageNet mean/std（与原实现一致）</li>
     *   <li>输出 shape：[1][3][targetShort][targetShort]</li>
     * </ul>
     * <p>
     * 性能：直接用 {@link java.awt.image.Raster#getPixels(int, int, int, int, int[])}，
     * 对 TYPE_INT_RGB 走的是 int[] 通道拷贝，无 ColorModel 转换开销；
     * 然后手工按位展开 R/G/B，与原 preprocess 内层循环字节级一致。
     */
    private float[][][][] toNchwFloat(BufferedImage src, int targetShort) {
        int cw = src.getWidth();
        int ch = src.getHeight();
        float[][][][] data = new float[1][3][ch][cw];
        int[] pixels = src.getRaster().getPixels(0, 0, cw, ch, (int[]) null);
        for (int y = 0; y < ch; y++) {
            for (int x = 0; x < cw; x++) {
                int idx = (y * cw + x);
                float r = ((pixels[idx] >> 16) & 0xFF) / 255f;
                float g = ((pixels[idx] >> 8) & 0xFF) / 255f;
                float b = (pixels[idx] & 0xFF) / 255f;
                data[0][0][y][x] = (r - MEAN[0]) / STD[0];
                data[0][1][y][x] = (g - MEAN[1]) / STD[1];
                data[0][2][y][x] = (b - MEAN[2]) / STD[2];
            }
        }
        return data;
    }

    private static BufferedImage toRgb(BufferedImage img) {
        if (img.getType() == BufferedImage.TYPE_INT_RGB) return img;
        BufferedImage out = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
        out.createGraphics().drawImage(img, 0, 0, null);
        out.createGraphics().dispose();
        return out;
    }

    private BufferedImage centerCrop(BufferedImage src, int targetW, int targetH) {
        int w = src.getWidth();
        int h = src.getHeight();
        if (w == targetW && h == targetH) return src;
        int x = Math.max(0, (w - targetW) / 2);
        int y = Math.max(0, (h - targetH) / 2);
        int cw = Math.min(targetW, w);
        int ch = Math.min(targetH, h);
        return src.getSubimage(x, y, cw, ch);
    }

    private static BufferedImage flipHorizontal(BufferedImage src) {
        int w = src.getWidth();
        int h = src.getHeight();
        BufferedImage out = new BufferedImage(w, h, src.getType());
        Graphics2D g = out.createGraphics();
        try {
            g.drawImage(src, w, 0, -w, h, null);
        } finally {
            g.dispose();
        }
        return out;
    }

    private static BufferedImage resizeDirect(BufferedImage src, int w, int h) {
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        var g = out.createGraphics();
        try {
            g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION,
                    java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(src, 0, 0, w, h, null);
        } finally {
            g.dispose();
        }
        return out;
    }

    /**
     * ONNX 输出提取：DINOv2 / SigLIP 视觉塔输出 shape = [batch, num_tokens, hidden_dim] -> float[][][]，
     * 只取 CLS token（索引 0）作为全局特征。同时兼容 float[][] / List / Object[] 等形态。
     */
    private static float[] extractFirstRow(Object output) {
        // DINOv2 / SigLIP 视觉塔：[batch, num_tokens, hidden_dim] -> float[][][]
        if (output instanceof float[][][] arr3d) {
            if (arr3d.length == 0 || arr3d[0].length == 0) {
                throw new IllegalStateException("ONNX 3D 输出为空");
            }
            return arr3d[0][0]; // batch=0, CLS token=0, 全部 hidden 维
        }
        if (output instanceof float[][] arr) return arr[0];
        if (output instanceof List<?> list) {
            if (list.isEmpty()) throw new IllegalStateException("ONNX 输出 List 为空");
            Object row = list.get(0);
            if (row instanceof float[] fr) return fr;
            if (row instanceof Number n) {
                float[] out = new float[list.size()];
                for (int i = 0; i < list.size(); i++) out[i] = ((Number) list.get(i)).floatValue();
                return out;
            }
            if (row instanceof List<?> innerList) {
                float[] out = new float[innerList.size()];
                for (int i = 0; i < innerList.size(); i++) {
                    Object v = innerList.get(i);
                    if (v instanceof Number nv) out[i] = nv.floatValue();
                    else throw new IllegalStateException("无法解析 List 内元素: " + (v == null ? "null" : v.getClass()));
                }
                return out;
            }
        }
        if (output instanceof Object[] arr) {
            Object row = arr[0];
            if (row instanceof float[] fr) return fr;
            if (row instanceof Object[] inner) {
                float[] out = new float[inner.length];
                for (int i = 0; i < inner.length; i++) {
                    Object v = inner[i];
                    if (v instanceof Number nv) out[i] = nv.floatValue();
                    else throw new IllegalStateException("无法解析 Object[] 内元素: " + (v == null ? "null" : v.getClass()));
                }
                return out;
            }
        }
        throw new IllegalStateException("未知的 ONNX 输出类型: " + output.getClass());
    }

    /**
     * 启动期探测所有输出头的维度，并校验与配置的 feature-dim 是否一致。
     */
    private void probeHeads(OrtSession session, int size) {
        try {
            log.info("[DINO 头探测] 启动期探测所有输出头 ...");
            float[][][][] dummy = new float[1][3][size][size];
            for (int c = 0; c < 3; c++) {
                for (int y = 0; y < size; y++) {
                    for (int x = 0; x < size; x++) {
                        dummy[0][c][y][x] = (0.5f - MEAN[c]) / STD[c];
                    }
                }
            }
            try (OnnxTensor tensor = OnnxTensor.createTensor(env, dummy)) {
                Map<String, OnnxTensor> inputs = Collections.singletonMap("pixel_values", tensor);
                try (OrtSession.Result result = session.run(inputs)) {
                    int maxDim = -1;
                    for (java.util.Iterator<Map.Entry<String, OnnxValue>> it = result.iterator(); it.hasNext(); ) {
                        Map.Entry<String, OnnxValue> e = it.next();
                        OnnxValue ov = e.getValue();
                        if (!(ov instanceof OnnxTensor ot)) continue;
                        long[] shape = ot.getInfo().getShape();
                        Object val = ot.getValue();
                        int dim = -1;
                        String type = "?";
                        if (val instanceof float[][][] a3) {
                            type = "float[][][]";
                            dim = (a3.length > 0 && a3[0].length > 0) ? a3[0][0].length : -1;
                        }
                        else if (val instanceof float[][] a) { type = "float[][]"; dim = a.length > 0 ? a[0].length : -1; }
                        else if (val instanceof List<?> l) {
                            type = "List";
                            if (!l.isEmpty()) {
                                Object row = l.get(0);
                                if (row instanceof float[] fr) dim = fr.length;
                                else if (row instanceof Number) dim = l.size();
                                else if (row instanceof List<?>) dim = ((List<?>) row).size();
                            }
                        }
                        else if (val instanceof Object[] a) {
                            type = "Object[]";
                            if (a.length > 0 && a[0] instanceof float[] fr) dim = fr.length;
                        }
                        log.info("[DINO 头探测]   {}  shape={}  type={}  dim={}", e.getKey(), Arrays.toString(shape), type, dim);
                        if (dim > maxDim) maxDim = dim;
                    }
                    int configured = embeddingProps.getDino().getFeatureDim();
                    log.info("[DINO 头探测] 配置 featureDim={} | 模型最大头维度={}", configured, maxDim);
                    // 维度不匹配只 warn 提示，不阻断启动（用户可能故意用大模型+小维度存，或者反之）
                    if (maxDim > 0 && maxDim != configured) {
                        log.warn("╔══════════════════════════════════════════════════════════════╗");
                        log.warn("║  [DINO 头探测] ⚠️  维度不匹配！                                       ║");
                        log.warn("║  application.yml: embedding.dino.feature-dim = {}               ║", configured);
                        log.warn("║  模型实际输出维度 = {} (即 {} 模型)                                ║", maxDim, maxDim == 384 ? "ViT-S/14" : maxDim == 768 ? "ViT-B/14" : maxDim == 1024 ? "ViT-L/14" : maxDim == 1536 ? "ViT-G/14" : "未知");
                        log.warn("║  修复步骤：                                                          ║");
                        log.warn("║    1. 修改 feature-dim 为 {}                                        ║", maxDim);
                        log.warn("║    2. Milvus 控制台执行: drop collection image_search_dino           ║");
                        log.warn("║    3. 重新执行批量入库                                                ║");
                        log.warn("╚══════════════════════════════════════════════════════════════╝");
                    } else {
                        log.info("[DINO 头探测] ✓ 维度校验通过: feature-dim={}", configured);
                    }
                }
            }
        } catch (Exception e) {
            // 探测失败时只 warn，允许启动（避免一个探测环节挂掉整个应用）
            log.warn("[DINO 头探测] ⚠️ 启动期维度探测失败，应用继续启动但请人工确认维度配置: {}", e.getMessage());
            log.debug("[DINO 头探测] 详细异常", e);
        }
    }

    private static float[] l2Normalize(float[] v) {
        float sum = 0;
        for (float f : v) sum += f * f;
        float norm = (float) Math.sqrt(sum);
        if (norm == 0) return v;
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) out[i] = v[i] / norm;
        return out;
    }

    @Override
    public int getFeatureDim() {
        return embeddingProps.getDino().getFeatureDim();
    }

    @Override
    public String getModelName() {
        return "dino";
    }

    /**
     * 返回 [poolSize, available, peakBorrowed]。
     * poolSize = 池容量建池时记录；available = POOL 当前 size；peakBorrowed = 历史借出峰值。
     */
    @Override
    public int[] getSessionPoolSnapshot() {
        synchronized (POOL) {
            return new int[] { initialPoolSize, POOL.size(), peakBorrowed };
        }
    }

    /**
     * 打印并清零本次会话的特征提取性能统计。
     * 由调用方（如 MilvusService）在一批完成后调用，避免与进程级计数混淆。
     */
    public void logAndResetPerf(String tag) {
        int calls = perfCalls.getAndSet(0);
        if (calls == 0) {
            log.info("[DINO 性能][{}] 本批无推理调用", tag);
            return;
        }
        long prepMs = perfPrepNs.getAndSet(0) / 1_000_000L;
        long runMs  = perfRunNs.getAndSet(0)  / 1_000_000L;
        log.info("[DINO 性能][{}] 调用 {} 次 | 预处理 {}ms ({}ms/次) | run() {}ms ({}ms/次) | 占比 prep:run = {}:{}",
                tag, calls,
                prepMs, prepMs / calls,
                runMs,  runMs  / calls,
                prepMs, runMs);
    }
}
