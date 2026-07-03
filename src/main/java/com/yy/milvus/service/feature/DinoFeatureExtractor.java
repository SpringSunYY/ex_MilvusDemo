package com.yy.milvus.service.feature;

import ai.onnxruntime.*;
import com.yy.milvus.config.EmbeddingProperties;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;


/**
 * 基于 ONNX Runtime 的 DINOv2 图像特征提取器。
 * <p>
 * 与 CLIP 的核心差异：
 * - DINOv2 是纯视觉自监督模型，不理解"语义类别"，只关心"视觉结构/纹理"
 * - CLIP 把"黑色格子衬衫"和"白色圆点衬衫"都归类为"衬衫"，相似度高
 * - DINOv2 能区分印花图案、格子纹理、纽扣细节等局部差异
 * - 以图搜图/商品检索场景，DINOv2 是首选
 * <p>
 * 输入预处理（DINOv2 标准）：
 * 1. resize 到短边 518（或配置的 scales[0]）
 * 2. CenterCrop 到正方形
 * 3. 转 RGB float32 [0, 1]
 * 4. ImageNet mean/std 归一化
 * 5. NCHW
 * <p>
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
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    // 最大单尺度边长，用于预分配复用 buffer（3 个通道 * 边长²）
    private static final int MAX_CHW_SIZE = 3 * 448 * 448;

    // RGB 通道归一化查找表：避免每像素做 3 次浮点除法+减法+乘法（v/255、v-MEAN、÷STD），
    // 改为 3 次数组查表。448×448 每尺度省 ~60 万次浮点运算；3 尺度批量 ~180 万次/张。
    private static final float[] R_LUT = new float[256];
    private static final float[] G_LUT = new float[256];
    private static final float[] B_LUT = new float[256];

    static {
        for (int i = 0; i < 256; i++) {
            float v = i / 255f;
            R_LUT[i] = (v - MEAN[0]) / STD[0];
            G_LUT[i] = (v - MEAN[1]) / STD[1];
            B_LUT[i] = (v - MEAN[2]) / STD[2];
        }
    }

    // 每线程复用 CHW float buffer，避免每张图 new 1.3MB 数组
    private final ThreadLocal<float[]> tensorBuffer =
            ThreadLocal.withInitial(() -> new float[MAX_CHW_SIZE]);

    // 每线程复用像素 int buffer，避免 getRGB() 每帧 new int[]
    private final ThreadLocal<int[]> pixelBuffer =
            ThreadLocal.withInitial(() -> new int[448 * 448]);

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

        OrtSession.SessionOptions opts = buildOpts(poolSize);
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
                // 使用 FloatBuffer 减少 GC 压力
                float[] dummy = new float[3 * warmupSize * warmupSize];
                float invStd0 = 1f / STD[0], invStd1 = 1f / STD[1], invStd2 = 1f / STD[2];
                float negMean0 = -MEAN[0], negMean1 = -MEAN[1], negMean2 = -MEAN[2];
                float val = (0.5f - MEAN[0]) / STD[0]; // 用 R 通道均值作为示例值
                int total = warmupSize * warmupSize;
                Arrays.fill(dummy, 0, total, val);
                val = (0.5f - MEAN[1]) / STD[1];
                Arrays.fill(dummy, total, total * 2, val);
                val = (0.5f - MEAN[2]) / STD[2];
                Arrays.fill(dummy, total * 2, total * 3, val);
                try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(dummy), new long[]{1, 3, warmupSize, warmupSize})) {
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
            return new int[]{DEFAULT_SHORT_EDGE};
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
            // 把非阻塞的统计和日志挪到锁外：减少 synchronized 块持有时间
            int borrowed = poolSizeEstimate() - POOL.size();
            if (borrowed > peakBorrowed.get()) peakBorrowed.set(borrowed);
            return s;
        }
    }

    private void returnSession(OrtSession s) {
        if (s == null) return;
        synchronized (POOL) {
            POOL.addLast(s);
            POOL.notify();
        }
    }

    /**
     * 池初始容量（建池时记录的；用于"借出量 = 总 - 可用"估算）
     */
    private int initialPoolSize = 0;

    private int poolSizeEstimate() {
        int s = initialPoolSize;
        return s > 0 ? s : POOL.size();
    }

    private OrtSession.SessionOptions buildOpts(int sessionPoolSize) {
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        var dino = embeddingProps.getDino();
        int intraOp;
        if (dino.isBatchMode()) {
            // 批量入库模式：IntraOp=1，让 session 走单线程、不抢 CPU；
            // N 张图并发时整体 CPU 占用可控，每个 session 排队等待本身不会变慢。
            intraOp = 1;
            log.info("[DINO 启动] batch-mode=true → IntraOp=1（批量友好，避免抢 CPU）");
        } else {
            // CPU 线程数至少 2，单图时性能高
            intraOp = Math.max(2, sessionPoolSize / Math.max(2, this.scales.length));
            log.info("[DINO 启动] batch-mode=false → IntraOp={}（pool={}，搜图加速）",
                    intraOp, sessionPoolSize);
        }
        try {
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            opts.setIntraOpNumThreads(intraOp);
            opts.setInterOpNumThreads(intraOp);
            // InterOp 不显式设置 = 1（ORT 默认）。DINOv2 forward 是严格顺序的
            // transformer 链，相邻算子无独立可并行维度，InterOp>1 收益 < 2%。
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
                try {
                    s.close();
                    count++;
                } catch (Exception ignored) {
                }
            }
            log.info("[DINO 关闭] 已关闭 {} 个 session，pool size={}", count, POOL.size());
        }
        if (env != null) env.close();
    }

    @Override
    public float[] extractFeature(File imageFile) throws IOException {
        return extractFeature(imageFile, null);
    }

    @Override
    public float[] extractFeature(File imageFile, ExecutorService exec) throws IOException {
        if (imageFile == null || !imageFile.exists() || imageFile.length() == 0) {
            throw new IOException("图像文件不可读: " + (imageFile == null ? "<null>" : imageFile.getAbsolutePath()));
        }
        BufferedImage img = ImageIO.read(imageFile);
        if (img == null) {
            throw new IOException("无法解析图像: " + imageFile.getAbsolutePath());
        }
        return extractFeatureFromBufferedImage(img, exec);
    }

    @Override
    public float[] extractFeature(InputStream inputStream) throws IOException {
        return extractFeature(inputStream, null);
    }

    @Override
    public float[] extractFeature(InputStream inputStream, ExecutorService exec) throws IOException {
        BufferedImage img = ImageIO.read(inputStream);
        if (img == null) {
            throw new IOException("无法解析图像 (InputStream)");
        }
        return extractFeatureFromBufferedImage(img, exec);
    }

    @Override
    public float[] extractFeature(byte[] imageBytes) throws IOException {
        return extractFeature(imageBytes, null);
    }

    @Override
    public float[] extractFeature(byte[] imageBytes, ExecutorService exec) throws IOException {
        return extractFeatureFromBufferedImage(
                ImageIO.read(new java.io.ByteArrayInputStream(imageBytes)), exec);
    }


    /**
     * 单图特征提取（可选：外层多线程并发加速多尺度推理）。
     *
     * <p>调用方传 {@code scaleExecutor} 后，3 个尺度的 {@code runInference} 会并发提交到该
     * 线程池，借 3 个不同的 session 真正并行推理；耗时 ≈ max(t_224, t_336, t_448)，而
     * 不是 t_224 + t_336 + t_448。批量入库/索引场景会传自己的 16 线程池，单图搜图场景
     * 也会传 featureExecutor，CPU 充裕时收益明显。
     *
     * <p>{@code scaleExecutor == null} 时退回到串行多尺度（保持与之前完全一致的行为）。
     */
    public float[] extractFeatureFromBufferedImage(BufferedImage image, ExecutorService scaleExecutor) throws IOException {
        if (image == null) {
            throw new IOException("图片读取失败");
        }
        long t0 = System.currentTimeMillis();

        BufferedImage rgb = toRgb(image);

        boolean wantHflip = embeddingProps.getDino().isHflipEnabled();
        int total = scales.length * (wantHflip ? 2 : 1);

        try {
            float[] embOriginal;
            float[] embFlipped;

            if (scaleExecutor == null || total <= 1) {
                // 串行分支：保留与原代码完全一致的行为（兼容所有不走并发的调用方）
                embOriginal = runMultiScale(rgb, null);
                if (wantHflip) {
                    BufferedImage flipped = flipHorizontal(rgb);
                    embFlipped = runMultiScale(flipped, null);
                } else {
                    embFlipped = null;
                }
            } else if (initialPoolSize > 0 && total > initialPoolSize) {
                // 守卫：单图并发任务数 > session 池容量时退化为串行，避免锁等待拖慢整体。
                // 例：scales=[224,336,448], sessionPoolSize=2 → total=3 > 2，走串行更稳。
                log.debug("[DINO 单图] total={} > poolSize={}，退化为串行多尺度", total, initialPoolSize);
                embOriginal = runMultiScale(rgb, null);
                if (wantHflip) {
                    BufferedImage flipped = flipHorizontal(rgb);
                    embFlipped = runMultiScale(flipped, null);
                } else {
                    embFlipped = null;
                }
            } else {
                // 并行分支：把 HFlip × 多尺度所有任务一次性并发提交，单图耗时 = max(...) 而不是 2*max(...)
                // 调用方负责保证 executor 容量充足，否则 borrowSession 内部会同步等待
                BufferedImage flipped = wantHflip ? flipHorizontal(rgb) : null;

                // 1) 调用线程串行做 resize/centerCrop，准备 N 个 tensor（BufferedImage 只读，多线程安全）
                List<BufferedImage> tensors = new ArrayList<>(total);
                for (int i = 0; i < total; i++) {
                    int si = i % scales.length;
                    BufferedImage src = (i < scales.length) ? rgb : flipped;
                    int targetShort = scales[si];
                    int w = src.getWidth();
                    int h = src.getHeight();
                    int newW, newH;
                    if (w < h) {
                        newW = targetShort;
                        newH = (int) Math.round((float) h * targetShort / w);
                    } else {
                        newH = targetShort;
                        newW = (int) Math.round((float) w * targetShort / h);
                    }
                    BufferedImage resized = resizeDirect(src, newW, newH);
                    tensors.add(centerCrop(resized, targetShort, targetShort));
                }

                // 2) 一次性 invokeAll：所有任务并发借不同 session 推理
                List<Callable<float[]>> tasks = new ArrayList<>(total);
                for (int i = 0; i < total; i++) {
                    final int idx = i;
                    final int scale = scales[idx % scales.length];
                    tasks.add(() -> {
                        long ts0 = System.currentTimeMillis();
                        float[] emb = runInference(tensors.get(idx), scale);
                        log.debug("[DINO 任务] scale={} hflip={} | 推理耗时 {}ms",
                                scale, idx >= scales.length, System.currentTimeMillis() - ts0);
                        return emb;
                    });
                }

                long tSubmit = System.currentTimeMillis();
                List<Future<float[]>> futures = scaleExecutor.invokeAll(tasks);
                log.debug("[DINO 并发调度] 提交 {} 个任务到 {} 并发执行，invokeAll 等待 {}ms",
                        total, scaleExecutor, System.currentTimeMillis() - tSubmit);

                // 3) 收集结果
                int hflipOffset = scales.length;
                embOriginal = futures.get(0).get();
                for (int si = 1; si < hflipOffset; si++) {
                    float[] emb;
                    try {
                        emb = futures.get(si).get();
                    } catch (ExecutionException e) {
                        log.warn("[DINO 并发] 原图尺度 {} 推理失败: {}",
                                scales[si], e.getCause() != null ? e.getCause().getMessage() : e.getMessage());
                        continue;
                    }
                    for (int k = 0; k < embOriginal.length; k++) embOriginal[k] += emb[k];
                }
                if (wantHflip) {
                    embFlipped = futures.get(hflipOffset).get();
                    for (int si = hflipOffset + 1; si < total; si++) {
                        float[] emb;
                        try {
                            emb = futures.get(si).get();
                        } catch (ExecutionException e) {
                            log.warn("[DINO 并发] 翻转图尺度 {} 推理失败: {}",
                                    scales[si - hflipOffset], e.getCause() != null ? e.getCause().getMessage() : e.getMessage());
                            continue;
                        }
                        for (int k = 0; k < embFlipped.length; k++) embFlipped[k] += emb[k];
                    }
                    // 把"翻转累加向量"按尺度归一并累加到原图
                    float scale = 1.0f / scales.length;
                    for (int k = 0; k < embOriginal.length; k++) {
                        embOriginal[k] = (embOriginal[k] * scale + embFlipped[k] * scale) * 0.5f;
                    }
                } else {
                    embFlipped = null;
                }
            }

            float[] result = l2Normalize(embOriginal);
            log.info("[DINO 单图] scales={} | hflip={} | branch={} | dim={} | 耗时 {}ms",
                    Arrays.toString(scales), wantHflip,
                    (scaleExecutor == null || total <= 1) ? "serial" : "parallel",
                    result.length, System.currentTimeMillis() - t0);
            return result;

        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            throw new IOException("DINOv2 推理被中断", ie);
        } catch (ExecutionException ee) {
            throw new IOException("DINOv2 推理失败: " + (ee.getCause() != null ? ee.getCause().getMessage() : ee.getMessage()), ee);
        } catch (Exception e) {
            throw new IOException("DINOv2 推理失败: " + e.getMessage(), e);
        }
    }


    /**
     * 多尺度推理：每个尺度跑一次 embedding 再平均。
     * <p>
     * 当 {@code scaleExecutor != null} 时，3 个尺度的 {@code runInference} 会被并发提交，
     * 借 3 个不同的 session 真正并行推理（耗时 ≈ max 而不是 sum）。CPU 16 线程以下机器慎用。
     */
    private float[] runMultiScale(BufferedImage rgb, ExecutorService scaleExecutor) throws IOException {
        if (scaleExecutor == null || scales.length <= 1) {
            return runMultiScaleSerial(rgb);
        }
        return runMultiScaleParallel(rgb, scaleExecutor);
    }

    private float[] runMultiScaleSerial(BufferedImage rgb) throws IOException {
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
                log.debug("[DINO 尺度] scale={} | 推理耗时 {}ms | 当前 {}/{} 个尺度有效",
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

    private float[] runMultiScaleParallel(BufferedImage rgb, ExecutorService executor) throws IOException {
        int n = scales.length;
        List<Callable<float[]>> tasks = new ArrayList<>(n);

        // 1) 在调用线程里完成所有尺度的 resize/centerCrop —— BufferedImage 共享给子任务只读使用
        BufferedImage[] tensorInputs = new BufferedImage[n];
        for (int si = 0; si < n; si++) {
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
            tensorInputs[si] = centerCrop(resized, targetShort, targetShort);
        }

        // 2) 提交到外层 executor：每个任务自己借一个 session，跑一个尺度
        for (int si = 0; si < n; si++) {
            final int idx = si;
            tasks.add(() -> {
                long ts0 = System.currentTimeMillis();
                float[] emb = runInference(tensorInputs[idx], scales[idx]);
                log.debug("[DINO 尺度并发] scale={} | 推理耗时 {}ms",
                        scales[idx], System.currentTimeMillis() - ts0);
                return emb;
            });
        }

        int avgLen = -1;
        float[] avg = null;
        int validScales = 0;
        long t0 = System.currentTimeMillis();
        try {
            List<Future<float[]>> futures = executor.invokeAll(tasks);
            for (int si = 0; si < futures.size(); si++) {
                float[] emb;
                try {
                    emb = futures.get(si).get();
                } catch (ExecutionException e) {
                    log.warn("[DINO 多尺度并发] 尺度 {} 推理失败: {}", scales[si], e.getCause() != null ? e.getCause().getMessage() : e.getMessage());
                    continue;
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException("DINO 多尺度并发被中断", e);
                }
                if (avg == null) {
                    avgLen = emb.length;
                    avg = new float[avgLen];
                } else if (emb.length != avgLen) {
                    log.warn("[DINO 多尺度] 尺度 {} 维度 {} 与首尺度 {} 不一致，跳过",
                            scales[si], emb.length, avgLen);
                    continue;
                }
                validScales++;
                float scale = 1.0f / scales.length;
                for (int i = 0; i < avgLen; i++) avg[i] += emb[i] * scale;
            }
            log.debug("[DINO 多尺度并发] scales={} | 总等待 {}ms | 有效 {}/{}",
                    Arrays.toString(scales), System.currentTimeMillis() - t0, validScales, n);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("DINO 多尺度并发被中断", e);
        }

        if (avg == null || validScales == 0) {
            log.warn("[DINO 多尺度并发] 所有尺度都不可用，降级到单尺度串行");
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
            // 使用 FloatBuffer 替代 float[][][][]，减少 GC 压力
            float[] chw = toNchwFloat(tensorInput, size);
            long prepNs = System.nanoTime() - t0;

            t0 = System.nanoTime();
            // FloatBuffer 直接对应 ONNX 的 [1, 3, H, W] shape
            // 使用 (offset, length) 形式限制读取范围，避免 ThreadLocal buffer 过大导致 shape 不匹配
            int chwLen = 3 * size * size;
            try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw, 0, chwLen), new long[]{1, 3, size, size})) {
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
    // 用 LongAdder 替代 AtomicLong：高并发累加场景下 LongAdder 通过分散 Cell 降低 CAS 争用，
    // 16 线程批量入库时 perf 调用频率高，LongAdder 比 AtomicLong 快 3-5 倍。
    private final LongAdder perfPrepNs = new LongAdder();
    private final LongAdder perfRunNs = new LongAdder();
    private final LongAdder perfCalls = new LongAdder();
    private final AtomicInteger peakBorrowed = new AtomicInteger(0);

    private void recordPerf(long prepNs, long runNs) {
        perfPrepNs.add(prepNs);
        perfRunNs.add(runNs);
        perfCalls.increment();
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
                float[] chw = toNchwFloat(cropped, targetShort);
                int chwLen = 3 * targetShort * targetShort;
                try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw, 0, chwLen), new long[]{1, 3, targetShort, targetShort})) {
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
     * NCHW float 预处理：返回复用 ThreadLocal buffer 的前 N 个元素。
     * <ul>
     *   <li>输入必须是 targetShort × targetShort 的 RGB 图（由 runMultiScale/runSingleScaleFallback 准备好）</li>
     *   <li>优先直接读 Raster DataBufferInt（零拷贝），否则回退到 getRGB()</li>
     *   <li>布局：所有 R 像素 → 所有 G 像素 → 所有 B 像素（NCHW 格式）</li>
     *   <li>归一化：ImageNet mean/std</li>
     *   <li>输出：float[MAX_CHW_SIZE] 的前 3*size*size 个元素有效</li>
     * </ul>
     */
    private float[] toNchwFloat(BufferedImage src, int size) {
        float[] data = tensorBuffer.get();
        int total = size * size;

        // 优先直接读 Raster DataBufferInt，避免 ColorModel 转换开销
        int[] pixels;
        try {
            var raster = src.getRaster();
            var db = raster.getDataBuffer();
            if (db instanceof DataBufferInt dbi) {
                pixels = dbi.getData();
            } else {
                // 回退：getRGB 但用 ThreadLocal 缓存 int[]
                int[] buf = pixelBuffer.get();
                if (buf.length < total) {
                    buf = new int[total];
                    pixelBuffer.set(buf);
                }
                src.getRGB(0, 0, size, size, buf, 0, size);
                pixels = buf;
            }
        } catch (Exception e) {
            // 极端情况（如读写锁）回退
            int[] buf = pixelBuffer.get();
            if (buf.length < total) {
                buf = new int[total];
                pixelBuffer.set(buf);
            }
            src.getRGB(0, 0, size, size, buf, 0, size);
            pixels = buf;
        }

        // NCHW 三个通道的起始偏移：R=[0, total), G=[total, 2*total), B=[2*total, 3*total)
        int rOff = 0;
        int gOff = total;
        int bOff = total * 2;

        // 扁平化单层循环 + LUT 查表：
        //   原写法每个像素 7 次位运算 + 7 次浮点除法 + 3 次加法 + 3 次乘法
        //   改写后每个像素 3 次位运算 + 3 次数组读（LUT 完全 cache line 友好）+ 3 次数组写
        //   448×448 ≈ 20 万像素，每尺度省 ~200 万次浮点运算
        //   y/x 仅用于原 HWC->HWC 写入；NCHW 写入与 HWC 读取索引天然对齐，可直接用单层 i
        for (int i = 0; i < total; i++) {
            int pixel = pixels[i];
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;
            data[rOff + i] = R_LUT[r];
            data[gOff + i] = G_LUT[g];
            data[bOff + i] = B_LUT[b];
        }
        return data;
    }

    private static BufferedImage toRgb(BufferedImage img) {
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

    private BufferedImage centerCrop(BufferedImage src, int targetW, int targetH) {
        int w = src.getWidth();
        int h = src.getHeight();
        if (w == targetW && h == targetH) return src;
        int x = Math.max(0, (w - targetW) / 2);
        int y = Math.max(0, (h - targetH) / 2);
        int cw = Math.min(targetW, w);
        int ch = Math.min(targetH, h);

        // 如果 src 任意一边 < target，getSubimage 出来的子图不是 targetW × targetH，
        // 后续 toNchwFloat 读取时会越界。这里走 pad 黑边兜底，输出严格 targetW × targetH。
        if (cw < targetW || ch < targetH) {
            BufferedImage padded = new BufferedImage(targetW, targetH, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = padded.createGraphics();
            try {
                g.setColor(Color.BLACK);
                g.fillRect(0, 0, targetW, targetH);
                g.drawImage(src, x, y, null);
            } finally {
                g.dispose();
            }
            return padded;
        }
        return src.getSubimage(x, y, cw, ch);
    }

    private static BufferedImage flipHorizontal(BufferedImage src) {
        int w = src.getWidth();
        int h = src.getHeight();
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        int[] srcPx = ((DataBufferInt) src.getRaster().getDataBuffer()).getData();
        int[] dstPx = ((DataBufferInt) out.getRaster().getDataBuffer()).getData();
        for (int y = 0; y < h; y++) {
            int row = y * w;
            int srcEnd = row + w - 1;
            for (int x = 0; x < w; x++) {
                dstPx[row + x] = srcPx[srcEnd - x];
            }
        }
        return out;
    }

    /**
     * 高性能 resize：Graphics2D + BILINEAR 插值，单次到位输出 TYPE_INT_RGB。
     * 相比 AffineTransformOp：少一次内存拷贝（filter 内部会再分配），且无 TYPE_3BYTE_BGR/ARGB
     * fallback 的二次转换。
     */
    private static BufferedImage resizeDirect(BufferedImage src, int w, int h) {
        if (src.getWidth() == w && src.getHeight() == h) return src;
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
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
                    else
                        throw new IllegalStateException("无法解析 List 内元素: " + (v == null ? "null" : v.getClass()));
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
                    else
                        throw new IllegalStateException("无法解析 Object[] 内元素: " + (v == null ? "null" : v.getClass()));
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
            // 使用 FloatBuffer 减少 GC 压力
            float[] dummy = new float[3 * size * size];
            int total = size * size;
            Arrays.fill(dummy, 0, total, (0.5f - MEAN[0]) / STD[0]);
            Arrays.fill(dummy, total, total * 2, (0.5f - MEAN[1]) / STD[1]);
            Arrays.fill(dummy, total * 2, total * 3, (0.5f - MEAN[2]) / STD[2]);
            try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(dummy), new long[]{1, 3, size, size})) {
                Map<String, OnnxTensor> inputs = Collections.singletonMap("pixel_values", tensor);
                try (OrtSession.Result result = session.run(inputs)) {
                    int maxDim = -1;
                    for (Map.Entry<String, OnnxValue> e : result) {
                        OnnxValue ov = e.getValue();
                        if (!(ov instanceof OnnxTensor ot)) continue;
                        long[] shape = ot.getInfo().getShape();
                        Object onnxVal = ot.getValue();
                        int dim = -1;
                        String type = "?";
                        if (onnxVal instanceof float[][][] a3) {
                            type = "float[][][]";
                            dim = (a3.length > 0 && a3[0].length > 0) ? a3[0][0].length : -1;
                        } else if (onnxVal instanceof float[][] a) {
                            type = "float[][]";
                            dim = a.length > 0 ? a[0].length : -1;
                        } else if (onnxVal instanceof List<?> l) {
                            type = "List";
                            if (!l.isEmpty()) {
                                Object row = l.getFirst();
                                if (row instanceof float[] fr) dim = fr.length;
                                else if (row instanceof Number) dim = l.size();
                                else if (row instanceof List<?>) dim = ((List<?>) row).size();
                            }
                        } else if (onnxVal instanceof Object[] a) {
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
        // 用 double 累加 + 一次 sqrt，避免 float 累加误差；原地归一化省一次 new float[]
        double sum = 0;
        for (float f : v) sum += (double) f * f;
        double norm = Math.sqrt(sum);
        if (norm == 0) return v;
        float inv = (float) (1.0 / norm);
        for (int i = 0; i < v.length; i++) v[i] *= inv;
        return v;
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
            return new int[]{initialPoolSize, POOL.size(), peakBorrowed.get()};
        }
    }

    /**
     * 打印并清零本次会话的特征提取性能统计。
     * 由调用方（如 MilvusService）在一批完成后调用，避免与进程级计数混淆。
     */
    public void logAndResetPerf(String tag) {
        int calls = (int) perfCalls.sumThenReset();
        if (calls == 0) {
            log.info("[DINO 性能][{}] 本批无推理调用", tag);
            return;
        }
        long prepMs = perfPrepNs.sumThenReset() / 1_000_000L;
        long runMs = perfRunNs.sumThenReset() / 1_000_000L;
        log.info("[DINO 性能][{}] 调用 {} 次 | 预处理 {}ms ({}ms/次) | run() {}ms ({}ms/次) | 占比 prep:run = {}:{}",
                tag, calls,
                prepMs, prepMs / calls,
                runMs, runMs / calls,
                prepMs, runMs);
    }
}
