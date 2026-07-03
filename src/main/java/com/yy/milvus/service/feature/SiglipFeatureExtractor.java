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
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;


/**
 * 基于 ONNX Runtime 的 SigLIP 2 图像特征提取器。
 * <p>
 * 与 CLIP 的差异：
 * - SigLIP 2 使用 sigmoid 对比损失（而非 CLIP 的 InfoNCE），训练更稳定
 * - 在零样本分类、图文检索任务上比 CLIP 精度高 1-3%
 * - 同样是视觉-语言对齐模型，适合语义理解 + 视觉特征混合场景
 * <p>
 * 预处理与 CLIP 相同：CLIP mean/std 归一化。
 */
@Component("siglipExtractor")
@Slf4j
@RequiredArgsConstructor
@ConditionalOnProperty(name = "embedding.model", havingValue = "siglip")
public class SiglipFeatureExtractor implements FeatureExtractor {

    private final EmbeddingProperties embeddingProps;

    private static final int DEFAULT_IMG_SIZE = 224;

    // ============================================================
    // SigLIP 2 官方归一化参数: image_mean=[0.5,0.5,0.5], image_std=[0.5,0.5,0.5]
    // 来自 google/siglip2-base-patch16-256/preprocessor_config.json
    // 与 CLIP 不同: CLIP 用 OpenAI AI-1 渲染图集算的 mean=[0.48145466, 0.4578275, 0.40821073]
    // ============================================================
    private static final float[] MEAN = {0.5f, 0.5f, 0.5f};
    private static final float[] STD = {0.5f, 0.5f, 0.5f};

    // 预计算倒数 + mean 归一化等价于 (x/255 - MEAN) / STD，但 (x*INV255 - MEAN)/STD = (x*INV255 - MEAN)*INV_STD
    // = x*INV255*INV_STD - MEAN*INV_STD。
    // 这样 preprocessing 主循环里只需要：f = x * KV[c] - BM[c]
    private static final float[] INV_255 = {1f / 255f, 1f / 255f, 1f / 255f};
    private static final float[] KV = new float[3]; // x*KV
    private static final float[] BM = new float[3]; // x*KV - BM

    static {
        for (int i = 0; i < 3; i++) {
            KV[i] = INV_255[i] / STD[i];
            BM[i] = MEAN[i] / STD[i];
        }
    }

    private static final java.util.concurrent.LinkedBlockingDeque<OrtSession> POOL = new java.util.concurrent.LinkedBlockingDeque<>();
    private static volatile boolean initialized = false;
    private static final AtomicInteger INSTANCE_COUNT = new AtomicInteger(0);
    private final int instanceId = INSTANCE_COUNT.incrementAndGet();

    // ===== 性能统计（线程安全） =====
    private static final LongAdder TOTAL_INFER_MS = new LongAdder();
    private static final LongAdder TOTAL_PREPROCESS_MS = new LongAdder();
    private static final LongAdder TOTAL_INFER_COUNT = new LongAdder();
    private static final AtomicInteger PEAK_BORROWED = new AtomicInteger(0);
    private static final AtomicInteger CURRENT_BORROWED = new AtomicInteger(0);
    private static volatile long LAST_STATS_LOG_MS = System.currentTimeMillis();
    private static final long STATS_LOG_INTERVAL_MS = 10_000L; // 10 秒打一次性能摘要

    private static void noteBorrow() {
        int cur = CURRENT_BORROWED.incrementAndGet();
        int peak = PEAK_BORROWED.get();
        if (cur > peak) PEAK_BORROWED.set(cur);
    }

    private static void noteReturn() {
        CURRENT_BORROWED.decrementAndGet();
    }

    private OrtEnvironment env;
    private Path modelPath;
    private int[] scales;
    /**
     * 模型输入图片尺寸，从 embedding.siglip.input-size 配置读取
     */
    private int inputSize;

    @PostConstruct
    public void init() throws Exception {
        if (initialized) {
            log.warn("[SigLIP 启动] 实例 #{} 跳过 init（static pool 已由实例 #1 初始化，pool size={}）",
                    instanceId, POOL.size());
            return;
        }

        env = OrtEnvironment.getEnvironment();
        log.info("[SigLIP 启动] 实例 #{} 开始初始化 ...", instanceId);

        var siglip = embeddingProps.getSiglip();
        this.scales = parseScales(siglip.getScales());
        this.inputSize = siglip.getInputSize() > 0 ? siglip.getInputSize() : DEFAULT_IMG_SIZE;

        log.info("[SigLIP 启动] ===== 配置值确认 =====");
        log.info("[SigLIP 启动] featureDim={} | inputSize={} | scales={} | hflipEnabled={}",
                siglip.getFeatureDim(), this.inputSize, Arrays.toString(this.scales), siglip.isHflipEnabled());
        log.info("[SigLIP 启动] visionModelPath={}", siglip.getVisionModelPath());
        log.info("[SigLIP 启动] ==========================");

        modelPath = resolveModelPath(siglip.getVisionModelPath(), "SigLIP");
        ensureModelFile(modelPath, "SigLIP");

        int poolSize = resolvePoolSize(siglip.getSessionPoolSize(), this.scales.length);
        log.info("[SigLIP 启动] 开始创建 {} 个 session ...", poolSize);

        OrtSession.SessionOptions opts = buildOpts(poolSize);
        for (int i = 0; i < poolSize; i++) {
            OrtSession s = env.createSession(modelPath.toString(), opts);
            POOL.addLast(s);
            log.info("[SigLIP 启动]   第 {}/{} 个 session 入池，当前 pool size={}", i + 1, poolSize, POOL.size());
        }
        log.info("[SigLIP 启动] session 池已就绪: {} 个", POOL.size());
        log.info("[SigLIP 启动] 输入: {} | 输出: {}",
                POOL.peekFirst().getInputNames(),
                POOL.peekFirst().getOutputNames());

        probeHeads(POOL.peekFirst());
        initialized = true;
        log.info("[SigLIP 启动] ✅ 初始化完成，pool 最终 size={}", POOL.size());
    }

    private static int resolvePoolSize(int configured, int scaleCount) {
        if (configured > 0) return configured;
        int cores = Math.max(1, Runtime.getRuntime().availableProcessors());
        return Math.min(cores, scaleCount);
    }

    /**
     * 解析多尺度列表，逻辑与 Dino/CLIP 同源，已统一到 {@link FeatureExtractorUtils#parseScales}。
     */
    private static int[] parseScales(List<Integer> list) {
        return FeatureExtractorUtils.parseScales(list, DEFAULT_IMG_SIZE, "SigLIP", 32);
    }

    private OrtSession borrowSession() throws OrtException {
        try {
            // LBQ.takeFirst() 自带锁，省掉 synchronized；阻塞但不耗 CPU
            OrtSession s = POOL.takeFirst();
            noteBorrow();
            return s;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new OrtException("[SigLIP] 等待被中断");
        }
    }

    private void returnSession(OrtSession s) {
        if (s == null) return;
        POOL.addLast(s);
        noteReturn();
    }

    private static void maybeLogStats() {
        long now = System.currentTimeMillis();
        long last = LAST_STATS_LOG_MS;
        if (now - last < STATS_LOG_INTERVAL_MS) return;
        if (!LAST_STATS_LOG_MS_CAS(last, now)) return;
        long cnt = TOTAL_INFER_COUNT.sum();
        if (cnt == 0) return;
        long totalInferMs = TOTAL_INFER_MS.sum();
        long totalPreMs = TOTAL_PREPROCESS_MS.sum();
        double avgInfer = (double) totalInferMs / cnt;
        double avgPre = (double) totalPreMs / cnt;
        log.info("[SigLIP 性能] 已推理 {} 张 | 平均 单次预处理 {}ms / 单次ONNX {}ms | 当前借出 {} | 峰值 {}",
                cnt, String.format("%.1f", avgPre), String.format("%.1f", avgInfer),
                CURRENT_BORROWED.get(), PEAK_BORROWED.get());
    }

    private static final AtomicLong LAST_STATS_LOG_MS_HOLDER = new AtomicLong(System.currentTimeMillis());

    private static boolean LAST_STATS_LOG_MS_CAS(long expected, long update) {
        return LAST_STATS_LOG_MS_HOLDER.compareAndSet(expected, update);
    }

    private OrtSession.SessionOptions buildOpts(int sessionPoolSize) {
        EmbeddingProperties.SiglipProperties siglip = embeddingProps.getSiglip();
        // 通用部分已统一到 utils；SigLIP 额外显式 setExecutionMode(SEQUENTIAL)
        // —— SigLIP ONNX 导出有显式 control flow，不允许 ORT 并行调度 node 之间的依赖。
        OrtSession.SessionOptions opts = FeatureExtractorUtils.buildOrtSessionOptions(
                log, "SigLIP", embeddingProps.isBatchMode(),
                siglip.getScales().size(), sessionPoolSize);
        try {
            opts.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        return opts;
    }

    private void ensureModelFile(Path p, String tower) throws IOException {
        if (!Files.exists(p)) {
            throw new IOException(
                    "\n[SigLIP] " + tower + "模型文件找不到: " + p.toAbsolutePath() + "\n" +
                            "[SigLIP] 请检查 application.yml 中 embedding.siglip.vision-model-path 是否正确\n" +
                            "[SigLIP] 推荐下载 ONNX 模型:\n" +
                            "[SigLIP]   https://huggingface.co/onnx-community/siglip2-base-patch16-256-ONNX"
            );
        }
    }

    private Path resolveModelPath(String configPath, String tower) {
        if (configPath == null || configPath.isEmpty()) {
            throw new IllegalArgumentException("[SigLIP] 模型路径未配置");
        }
        Path p = Paths.get(configPath);
        if (p.isAbsolute() && Files.exists(p)) {
            log.info("[SigLIP] {}模型使用绝对路径: {}", tower, p);
            return p;
        }
        try {
            String normalized = configPath.replace('\\', '/');
            var url = getClass().getClassLoader().getResource(normalized);
            if (url != null) {
                Path cp = Paths.get(url.toURI());
                log.info("[SigLIP] {}模型使用 classpath: {}", tower, cp);
                return cp;
            }
        } catch (Exception ignored) {
        }
        if (Files.exists(p)) {
            log.warn("[SigLIP] {}模型使用工作目录相对路径: {}", tower, p.toAbsolutePath());
            return p.toAbsolutePath();
        }
        return p.isAbsolute() ? p : p.toAbsolutePath();
    }

    @PreDestroy
    public void close() throws OrtException {
        log.info("[SigLIP 关闭] 开始销毁 session 池，当前 pool size={}", POOL.size());
        OrtSession s;
        int count = 0;
        while ((s = POOL.pollFirst()) != null) {
            try {
                s.close();
                count++;
            } catch (Exception ignored) {
            }
        }
        log.info("[SigLIP 关闭] 已销毁 {} 个 session", count);
        if (env != null) env.close();
    }

    @Override
    public float[] extractFeature(File imageFile) throws IOException {
        return extractFeature(imageFile, null);
    }

    @Override
    public float[] extractFeature(InputStream inputStream) throws IOException {
        return extractFeature(inputStream, null);
    }

    @Override
    public float[] extractFeature(byte[] imageBytes) throws IOException {
        return extractFeature(imageBytes, null);
    }

    /**
     * 单图特征提取（可选：传入 ExecutorService 触发单图内多尺度并发）。
     *
     * <p>传入 executor 后，SigLIP 的多个尺度 [224,256,384] 会并发提交到线程池，借不同 session
     * 真正并行推理。3 个尺度的总耗时 ≈ max(t_224, t_256, t_384) 而不是 sum。
     * session 池容量需 ≥ scales.length（默认 16 个 session，3 个尺度完全够用）。
     *
     * <p>executor == null 时退化为串行多尺度（保持与原行为一致）。
     */
    @Override
    public float[] extractFeature(File imageFile, ExecutorService exec) throws IOException {
        if (imageFile == null || !imageFile.exists() || imageFile.length() == 0) {
            throw new IOException("图像文件不可读: " + (imageFile == null ? "<null>" : imageFile.getAbsolutePath()));
        }
        BufferedImage img = ImageIO.read(imageFile);
        if (img == null) throw new IOException("无法解码图像: " + imageFile.getAbsolutePath());
        return extractFeatureFromBufferedImage(img, exec);
    }

    @Override
    public float[] extractFeature(InputStream inputStream, ExecutorService exec) throws IOException {
        BufferedImage img = ImageIO.read(inputStream);
        if (img == null) throw new IOException("无法解码图像 (InputStream)");
        return extractFeatureFromBufferedImage(img, exec);
    }

    @Override
    public float[] extractFeature(byte[] imageBytes, ExecutorService exec) throws IOException {
        return extractFeatureFromBufferedImage(
                ImageIO.read(new java.io.ByteArrayInputStream(imageBytes)), exec);
    }

    /**
     * @param scaleExecutor 可选：传入非空时启用"单图内多尺度并发"，N 个尺度借不同 session 并行推理。
     *                      session 池容量必须 ≥ scales.length，否则会因借不到 session 而退化为串行（仍能跑通）。
     */
    private float[] extractFeatureFromBufferedImage(BufferedImage image, ExecutorService scaleExecutor) throws IOException {
        if (image == null) throw new IOException("图片读取失败");
        BufferedImage rgb = toRgb(image);
        // 服装设计图场景适配：先去掉白边再推理（仅当白边显著时才生效）
        rgb = autoCropWhitespace(rgb);
        int avgLen = -1;

        float[] embOriginal = null;
        float[] embFlipped = null;

        try {
            embOriginal = runMultiScale(rgb, scaleExecutor);
            avgLen = embOriginal.length;

            if (embeddingProps.getSiglip().isHflipEnabled()) {
                // 翻转版本复用 runMultiScale 路径（这里仅仅是把图翻了再喂），保持效果等价。
                // 真实热点在 ONNX 推理本身，详见 preprocess / session 选项优化。
                BufferedImage flippedRgb = flipHorizontal(rgb);
                embFlipped = runMultiScale(flippedRgb, scaleExecutor);
                if (embFlipped.length == avgLen) {
                    for (int i = 0; i < avgLen; i++) embOriginal[i] = (embOriginal[i] + embFlipped[i]) * 0.5f;
                }
            }
            return l2Normalize(embOriginal);
        } catch (Exception e) {
            throw new IOException("SigLIP 推理失败: " + e.getMessage(), e);
        }
    }

    /**
     * 对给定图片跑多尺度推理，返回 L2 归一化前的平均 embedding。
     *
     * <p>并发策略（按调用方意图区分）：
     * <ul>
     *   <li>{@code scaleExecutor == null}：调用方是同步直接调用（典型场景：单图搜图）。
     *       走 {@link ForkJoinPool#commonPool()} 多尺度并发。</li>
     *   <li>{@code scaleExecutor != null}：调用方正在批量池里跑（典型场景：批量入库）。
     *       <b>强制走串行多尺度</b>。原因：外层池已塞满，并发只会争资源、拖慢整体 throughput。</li>
     * </ul>
     *
     * @param scaleExecutor 非 null 表示"我在批量池里跑，请串行"；null 表示"我是孤立的单图调用，请并发加速"
     */
    private float[] runMultiScale(BufferedImage rgb, ExecutorService scaleExecutor) throws IOException {
        if (scales.length <= 1) {
            return runMultiScaleSerial(rgb);
        }
        if (scaleExecutor != null) {
            return runMultiScaleSerial(rgb);
        }
        ForkJoinPool pool = ForkJoinPool.commonPool();
        if (pool.getParallelism() < 2) {
            return runMultiScaleSerial(rgb);
        }
        return runMultiScaleParallel(rgb);
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
            // pad-or-center-crop：比 crop 小的尺度 pad 黑边到 inputSize；其余物理 crop（避免子图越界）
            BufferedImage tensorInput;
            if (resized.getWidth() < inputSize || resized.getHeight() < inputSize) {
                tensorInput = ImagePreprocessor.padToSize(resized, inputSize);
            } else {
                tensorInput = ImagePreprocessor.centerCropPhysical(resized, inputSize);
            }

            try {
                float[] emb = runInference(tensorInput);
                if (avg == null) {
                    avgLen = emb.length;
                    avg = new float[avgLen];
                } else if (emb.length != avgLen) {
                    log.warn("[SigLIP 多尺度] 尺度 {} 维度 {} 与首尺度 {} 不一致，跳过",
                            targetShort, emb.length, avgLen);
                    continue;
                }
                validScales++;
                float scale = 1.0f / scales.length;
                for (int i = 0; i < avgLen; i++) avg[i] += emb[i] * scale;
            } catch (OrtException e) {
                log.warn("[SigLIP 多尺度] 尺度 {} 推理失败: {}", targetShort, e.getMessage());
            }
        }

        if (avg == null || validScales == 0) {
            log.warn("[SigLIP 多尺度] 所有尺度都不可用，降级到单尺度");
            return runSingleScaleFallback(rgb);
        }
        return avg;
    }

    /**
     * 多尺度并发：调用线程串行完成所有尺度的 resize/crop（输出 N 张 inputSize×inputSize BufferedImage），
     * 然后 invokeAll 提交到 {@link ForkJoinPool#commonPool()} 跑 N 个 inference 任务。
     *
     * <p>此方法仅在调用方传 {@code null} 时进入（即单图搜图场景，请求方不在任何池里跑），
     * 用 JVM 全局 commonPool 即可，无需借用调用方的池。
     */
    private float[] runMultiScaleParallel(BufferedImage rgb) throws IOException {
        int n = scales.length;

        // 1) 在调用线程里完成所有尺度的 resize/crop —— BufferedImage 共享给子任务只读使用
        BufferedImage[] tensorInputs = new BufferedImage[n];
        int[] targetShorts = new int[n];
        for (int si = 0; si < n; si++) {
            int targetShort = scales[si];
            targetShorts[si] = targetShort;
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
            // pad-or-center-crop：比 crop 小的尺度 pad 黑边到 inputSize；其余物理 crop
            if (resized.getWidth() < inputSize || resized.getHeight() < inputSize) {
                tensorInputs[si] = ImagePreprocessor.padToSize(resized, inputSize);
            } else {
                tensorInputs[si] = ImagePreprocessor.centerCropPhysical(resized, inputSize);
            }
        }

        // 2) 提交到 ForkJoinPool.commonPool：每个任务自己借一个 session，跑一个尺度
        //    关键：使用 commonPool 而非调用方传入的池，避免与 featureExecutor 互锁导致死锁。
        List<Callable<float[]>> tasks = new ArrayList<>(n);
        for (int si = 0; si < n; si++) {
            final int idx = si;
            tasks.add(() -> runInference(tensorInputs[idx]));
        }

        int avgLen = -1;
        float[] avg = null;
        int validScales = 0;
        long t0 = System.currentTimeMillis();
        try {
            // 单图搜图场景下的多尺度并发加速：N 个尺度的 ONNX 推理并行执行，单图耗时 ≈ max 而不是 sum。
            // 走 commonPool 而非调用方传入的池——因为这个分支只有调用方传 null 时才会进入（即"我不在任何池里跑"），
            // 没有外层池可以借用，只能用 JVM 全局的 commonPool。
            List<Future<float[]>> futures = ForkJoinPool.commonPool().invokeAll(tasks);
            for (int si = 0; si < futures.size(); si++) {
                float[] emb;
                try {
                    emb = futures.get(si).get();
                } catch (ExecutionException e) {
                    log.warn("[SigLIP 多尺度并发] 尺度 {} 推理失败: {}", targetShorts[si],
                            e.getCause() != null ? e.getCause().getMessage() : e.getMessage());
                    continue;
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException("SigLIP 多尺度并发被中断", e);
                }
                if (avg == null) {
                    avgLen = emb.length;
                    avg = new float[avgLen];
                } else if (emb.length != avgLen) {
                    log.warn("[SigLIP 多尺度并发] 尺度 {} 维度 {} 与首尺度 {} 不一致，跳过",
                            targetShorts[si], emb.length, avgLen);
                    continue;
                }
                validScales++;
                float scale = 1.0f / n;
                for (int i = 0; i < avgLen; i++) avg[i] += emb[i] * scale;
            }
            log.debug("[SigLIP 多尺度并发] scales={} | 总等待 {}ms | 有效 {}/{}",
                    Arrays.toString(scales), System.currentTimeMillis() - t0, validScales, n);
        } catch (Exception e) {
            Thread.currentThread().interrupt();
            throw new IOException("SigLIP 多尺度并发被中断", e);
        }

        if (avg == null || validScales == 0) {
            log.warn("[SigLIP 多尺度并发] 所有尺度都不可用，降级到单尺度串行");
            return runSingleScaleFallback(rgb);
        }
        return avg;
    }

    private float[] runInference(BufferedImage tensorInput) throws OrtException {
        long tPre = System.nanoTime();
        float[][][][] chw = preprocess(tensorInput);
        TOTAL_PREPROCESS_MS.add((System.nanoTime() - tPre) / 1_000_000L);

        OrtSession session = borrowSession();
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, chw)) {
            // HashMap 复用：每线程一个，避免每张图重建 Map。
            java.util.HashMap<String, OnnxTensor> map = HASHMAP_HOLDER.get();
            map.put("pixel_values", tensor);
            long tRun = System.nanoTime();
            OrtSession.Result result = session.run(map);
            try {
                TOTAL_INFER_MS.add((System.nanoTime() - tRun) / 1_000_000L);
                TOTAL_INFER_COUNT.increment();
                float[] out = pickEmbedOutput(result);
                maybeLogStats();
                return out;
            } finally {
                try {
                    result.close();
                } catch (Exception ignore) {
                }
                map.clear(); // 去掉对 OnnxTensor 的引用，避免在 OnnxTensor close 前还有别的引用
            }
        } finally {
            returnSession(session);
        }
    }

    // 每线程一个 HashMap，避免每张图 new 一个
    private static final ThreadLocal<java.util.HashMap<String, OnnxTensor>> HASHMAP_HOLDER = ThreadLocal.withInitial(java.util.HashMap::new);

    private float[] runSingleScaleFallback(BufferedImage image) throws IOException {
        try {
            BufferedImage rgb = toRgb(image);
            int w = rgb.getWidth(), h = rgb.getHeight();
            int newW, newH;
            if (w < h) {
                newW = inputSize;
                newH = (int) Math.round((float) h * inputSize / w);
            } else {
                newH = inputSize;
                newW = (int) Math.round((float) w * inputSize / h);
            }
            BufferedImage resized = resizeDirect(rgb, newW, newH);
            // 极小图兜底：此处 resized 一定是 ≥ inputSize（短边缩到 inputSize 后另一边只会更大），走物理 crop 分支
            BufferedImage cropped = ImagePreprocessor.centerCropPhysical(resized, inputSize);
            float[][][][] chw = preprocess(cropped);
            OrtSession session = borrowSession();
            try (OnnxTensor tensor = OnnxTensor.createTensor(env, chw)) {
                java.util.HashMap<String, OnnxTensor> map = HASHMAP_HOLDER.get();
                map.put("pixel_values", tensor);
                try (OrtSession.Result result = session.run(map)) {
                    return pickEmbedOutput(result);
                } finally {
                    map.clear();
                }
            } catch (OrtException e) {
                throw new IOException("SigLIP 推理失败: " + e.getMessage(), e);
            } finally {
                returnSession(session);
            }
        } catch (Exception e) {
            throw new IOException("SigLIP 单尺度降级失败: " + e.getMessage(), e);
        }
    }

    private float[][][][] preprocess(BufferedImage cropped) {
        int cw = cropped.getWidth(), ch = cropped.getHeight();
        float[][][][] data = new float[1][3][ch][cw];
        int need = cw * ch;
        int[] pixels = borrowedIntBuffer(need);
        // 用 BufferedImage.getRGB(x, y, w, h, int[], offset, scansize) —— 比 WritableRaster.getPixels 更稳定
        // getRGB 永远按 int[] 写 w*h 个元素，没有 stride/sample model 边界怪问题
        cropped.getRGB(0, 0, cw, ch, pixels, 0, cw);
        // KV[c] = 1/(255*STD[c]); BM[c] = MEAN[c]/STD[c]
        // data[0][c][y][x] = pixel * KV[c] - BM[c]
        float kvR = KV[0], kvG = KV[1], kvB = KV[2];
        float bmR = BM[0], bmG = BM[1], bmB = BM[2];
        float[][] rAll = data[0][0], gAll = data[0][1], bAll = data[0][2];
        int idx = 0;
        for (int y = 0; y < ch; y++) {
            float[] rRow = rAll[y];
            float[] gRow = gAll[y];
            float[] bRow = bAll[y];
            for (int x = 0; x < cw; x++) {
                int px = pixels[idx++];
                // getRGB 返回 0xAARRGGBB 形式
                rRow[x] = ((px >> 16) & 0xFF) * kvR - bmR;
                gRow[x] = ((px >> 8) & 0xFF) * kvG - bmG;
                bRow[x] = (px & 0xFF) * kvB - bmB;
            }
        }
        return data;
    }

    // 简单的 ThreadLocal int[] 复用（每线程最多缓存 1 个 buffer）
    private static final ThreadLocal<int[]> PIXEL_BUFFER_HOLDER = new ThreadLocal<>();

    private static int[] borrowedIntBuffer(int need) {
        int[] buf = PIXEL_BUFFER_HOLDER.get();
        if (buf == null || buf.length < need) {
            buf = new int[Math.max(need, 8192)];
            PIXEL_BUFFER_HOLDER.set(buf);
        }
        return buf;
    }

    private static void returnBorrowedIntBuffer(int[] buf) {
        // 当前实现直接复用，无需归还
    }

    /**
     * 转 TYPE_INT_RGB。逻辑已统一到 {@link FeatureExtractorUtils#toRgb}，保留薄包装对齐 Dino 风格。
     */
    private static BufferedImage toRgb(BufferedImage img) {
        return FeatureExtractorUtils.toRgb(img);
    }

    private static BufferedImage flipHorizontal(BufferedImage src) {
        return ImagePreprocessor.flipHorizontal(src);
    }

    /**
     * 自动裁掉白边（服装设计图场景适配）：扫描图片边界，找到非白内容 bbox 并裁切，
     * 然后外面 pad 黑边到原尺寸。padding 比例 <= 8% 时直接返回原图（避免小图误裁）。
     * <p>
     * 算法：
     * - 把整图缩到 128×128（粗扫描，速度优先）
     * - 在缩略图上：像素亮度 > THRESHOLD(235) 视为"白/接近白"
     * - 找最外层非白行的 row range、col range
     * - 把这行映射回原图坐标作为 crop bbox
     * - 如果裁掉的边 < 4% 或裁出来的图本身 < 原图 70%，跳过（噪声保护）
     */
    private static BufferedImage autoCropWhitespace(BufferedImage src) {
        int W = src.getWidth(), H = src.getHeight();
        if (W < 64 || H < 64) return src;
        final int SCAN_W = 128;
        final int SCAN_H = Math.max(1, SCAN_W * H / W);
        BufferedImage probe = resizeDirect(src, SCAN_W, SCAN_H);
        int probeSize = SCAN_W * SCAN_H;
        // 复用 ThreadLocal pixel buffer：避免每张图 new int[16384]，批量入库场景下省一次小对象分配
        int[] pixels = PIXEL_BUFFER_HOLDER.get();
        if (pixels == null || pixels.length < probeSize) {
            pixels = new int[Math.max(probeSize, 8192)];
            PIXEL_BUFFER_HOLDER.set(pixels);
        }
        probe.getRGB(0, 0, SCAN_W, SCAN_H, pixels, 0, SCAN_W);
        int TH = 235;  // 灰度阈值，超过视为白
        int minX = SCAN_W, minY = SCAN_H, maxX = -1, maxY = -1;
        for (int y = 0; y < SCAN_H; y++) {
            for (int x = 0; x < SCAN_W; x++) {
                int p = pixels[y * SCAN_W + x];
                int r = (p >> 16) & 0xFF, g = (p >> 8) & 0xFF, b = p & 0xFF;
                int lum = (r * 299 + g * 587 + b * 114) / 1000;
                if (lum < TH) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }
        if (maxX < 0) return src; // 整张图全是白
        // 把扫描坐标映射回原图
        int x0 = minX * W / SCAN_W;
        int y0 = minY * H / SCAN_H;
        int x1 = (maxX + 1) * W / SCAN_W;
        int y1 = (maxY + 1) * H / SCAN_H;
        // 保留 4% padding（避免裁到线条边缘）
        int padX = (x1 - x0) / 25;
        int padY = (y1 - y0) / 25;
        x0 = Math.max(0, x0 - padX);
        y0 = Math.max(0, y0 - padY);
        x1 = Math.min(W, x1 + padX);
        y1 = Math.min(H, y1 + padY);
        int cropW = x1 - x0, cropH = y1 - y0;
        if (cropW <= 0 || cropH <= 0) return src;
        // 噪声保护：去掉白边比例 < 5% 或裁后图 < 原图 70%，跳过
        int removedArea = (W * H) - cropW * cropH;
        if (removedArea * 20 < W * H) return src;          // 裁白边 < 5%
        if (cropW * cropH * 10 < W * H * 7) return src;    // 裁后剩余 < 70%
        // 物理裁图（注意：不用 getSubimage，raster 视图问题前面修过了，再次强调）
        BufferedImage out = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        try {
            g.setColor(Color.WHITE);
            g.fillRect(0, 0, W, H);
            g.drawImage(src, 0, 0, W, H, x0, y0, x1, y1, null);
        } finally {
            g.dispose();
        }
        log.debug("[SigLIP 白底裁切] 原图 {}x{} -> 内容 bbox ({},{},{},{}), 裁掉面积 {:.1f}%",
                W, H, x0, y0, x1, y1, removedArea * 100.0 / (W * H));
        return out;
    }

    private static BufferedImage resizeDirect(BufferedImage src, int w, int h) {
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        var g = out.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(src, 0, 0, w, h, null);
        } finally {
            g.dispose();
        }
        return out;
    }

    private static float[] extractFirstRow(Object output) {
        // float[] 直接返回（SigLIP ONNX 常见输出）
        if (output instanceof float[] fa) return fa;
        if (output instanceof float[][] arr) return arr[0];
        if (output instanceof float[][][] arr3) {
            // 形状 [1, N, D] → 取 [0][0]
            if (arr3.length > 0 && arr3[0] != null && arr3[0].length > 0) return arr3[0][0];
            throw new IllegalStateException("三维 ONNX 输出为空: " + Arrays.deepToString(new float[0][0][0]));
        }
        if (output instanceof List<?> list) {
            if (list.isEmpty()) throw new IllegalStateException("ONNX 输出 List 为空");
            Object row = list.get(0);
            if (row instanceof float[] fr) return fr;
            if (row instanceof Number) {
                float[] out = new float[list.size()];
                for (int i = 0; i < list.size(); i++) out[i] = ((Number) list.get(i)).floatValue();
                return out;
            }
        }
        if (output instanceof Object[] arr) {
            Object row = arr[0];
            if (row instanceof float[] fr) return fr;
            if (row instanceof Number) {
                float[] out = new float[arr.length];
                for (int i = 0; i < arr.length; i++) out[i] = ((Number) arr[i]).floatValue();
                return out;
            }
        }
        throw new IllegalStateException("未知的 ONNX 输出类型: " + output.getClass());
    }

    /**
     * 从 ONNX 多个输出头中智能选取 image embedding。
     * 优先级：
     * 1) 维度 == featureDim 的头（patch 数 256 一般=特征维）
     * 2) shape 中含 image_embeds / pooler_output / embedding 关键字
     * 3) 取第一个头（兜底）
     */
    private float[] pickEmbedOutput(OrtSession.Result result) throws OrtException {
        int target = embeddingProps.getSiglip().getFeatureDim();
        String[] preferredNames = {"image_embeds", "pooler_output", "embedding", "pooled_output", "logits_per_image"};

        // 1. 优先按名字匹配
        for (String name : preferredNames) {
            try {
                Optional<OnnxValue> opt = result.get(name);
                if (opt.isPresent() && opt.get() instanceof OnnxTensor t) {
                    float[] got = extractFirstRow(t.getValue());
                    if (got != null && got.length == target) {
                        return got;
                    }
                }
            } catch (Exception ignore) {
            }
        }
        // 2. 按维度匹配：先看最后一维 == target
        for (java.util.Iterator<Map.Entry<String, OnnxValue>> it = result.iterator(); it.hasNext(); ) {
            Map.Entry<String, OnnxValue> e = it.next();
            if (!(e.getValue() instanceof OnnxTensor t)) continue;
            try {
                Object val = t.getValue();
                int lastDim = lastDimOf(val);
                if (lastDim == target) {
                    float[] got = extractFirstRow(val);
                    if (got != null && got.length == target) {
                        log.info("[SigLIP 推理] 按维度匹配输出头 {} dim={}", e.getKey(), got.length);
                        return got;
                    }
                }
            } catch (Exception ignore) {
            }
        }
        // 3. 兜底：对第一个非空头做 mean pool 到 target 维（处理 [1,N,target] 这种 hidden state）
        for (java.util.Iterator<Map.Entry<String, OnnxValue>> it = result.iterator(); it.hasNext(); ) {
            Map.Entry<String, OnnxValue> e = it.next();
            if (!(e.getValue() instanceof OnnxTensor t)) continue;
            try {
                Object val = t.getValue();
                if (val instanceof float[][][] a3) {
                    float[] pooled = meanPool(a3[0], target);
                    log.warn("[SigLIP 推理] ONNX 模型未输出 image_embeds，使用 {} 头 mean-pool 得到 dim={}。" +
                                    "建议重导出 ONNX 时保留 image_embeds 头以获得更好的语义向量",
                            e.getKey(), pooled.length);
                    return pooled;
                }
            } catch (Exception ignore) {
            }
        }
        // 4. 真·兜底
        log.warn("[SigLIP 推理] 未找到 dim={} 的输出头，取第一个头", target);
        return extractFirstRow(result.get(0).getValue());
    }

    private static int lastDimOf(Object val) {
        if (val instanceof float[] fa) return fa.length;
        if (val instanceof float[][] a) return a.length > 0 ? a[0].length : -1;
        if (val instanceof float[][][] a3) return a3.length > 0 && a3[0].length > 0 ? a3[0][0].length : -1;
        return -1;
    }

    private static float[] meanPool(float[][] tokens, int dim) {
        int n = tokens.length;
        if (n == 0) return new float[dim];
        int d = Math.min(tokens[0].length, dim);
        float[] out = new float[dim];
        for (int i = 0; i < n; i++) {
            float[] row = tokens[i];
            for (int j = 0; j < d; j++) out[j] += row[j];
        }
        for (int j = 0; j < d; j++) out[j] /= n;
        return out;
    }

    private void probeHeads(OrtSession session) {
        try {
            log.info("[SigLIP 头探测] 启动期探测所有输出头 ...");
            // 扁平化 float[] + Arrays.fill 已统一到 utils
            int plane = inputSize * inputSize;
            float[] dummy = new float[3 * plane];
            FeatureExtractorUtils.fillDummyCenter(dummy, inputSize, MEAN, STD);
            try (OnnxTensor tensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(dummy),
                    new long[]{1, 3, inputSize, inputSize})) {
                Map<String, OnnxTensor> inputs = Collections.singletonMap("pixel_values", tensor);
                try (OrtSession.Result result = session.run(inputs)) {
                    int configured = embeddingProps.getSiglip().getFeatureDim();
                    int matched = -1;
                    String matchedName = null;
                    for (java.util.Iterator<Map.Entry<String, OnnxValue>> it = result.iterator(); it.hasNext(); ) {
                        Map.Entry<String, OnnxValue> e = it.next();
                        OnnxValue ov = e.getValue();
                        if (!(ov instanceof OnnxTensor ot)) continue;
                        int dim = FeatureExtractorUtils.inferDimFromOnnxValue(ot.getValue());
                        log.info("[SigLIP 头探测]   {}  shape={}  dim={}", e.getKey(), Arrays.toString(ot.getInfo().getShape()), dim);
                        if (dim == configured) {
                            matched = dim;
                            matchedName = e.getKey();
                        }
                    }
                    log.info("[SigLIP 头探测] 配置 featureDim={} | 匹配输出头={} dim={}", configured, matchedName, matched);
                    if (matched < 0) {
                        throw new IllegalStateException(String.format(
                                "[SigLIP] 没有输出头的维度等于配置的 feature-dim=%d。\n" +
                                        "[SigLIP] ONNX 模型通常会输出多个头：image_embeds (768) + last_hidden_state (1024) 等。\n" +
                                        "[SigLIP] 请检查 yml embedding.siglip.feature-dim 是否与实际 image embedding 维度一致",
                                configured));
                    }
                    log.info("[SigLIP 头探测] ✅ 维度校验通过（命中输出头 {} dim={}）", matchedName, matched);
                }
            }
        } catch (Exception e) {
            log.error("[SigLIP 头探测] 探测失败，不影响主流程", e);
        }
    }

    /**
     * L2 归一化。逻辑已统一到 {@link FeatureExtractorUtils#l2Normalize}，保留薄包装对齐 Dino 风格。
     */
    private static float[] l2Normalize(float[] v) {
        return FeatureExtractorUtils.l2Normalize(v);
    }

    @Override
    public int getFeatureDim() {
        return embeddingProps.getSiglip().getFeatureDim();
    }

    @Override
    public String getModelName() {
        return "siglip";
    }
}
