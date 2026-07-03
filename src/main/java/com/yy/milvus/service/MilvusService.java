package com.yy.milvus.service;

import com.google.common.collect.Lists;
import com.yy.milvus.config.MilvusProperties;
import com.yy.milvus.domain.SearchResult;
import com.yy.milvus.domain.VectorRecord;
import com.yy.milvus.service.feature.FeatureExtractor;
import io.milvus.client.MilvusServiceClient;
import io.milvus.common.clientenum.ConsistencyLevelEnum;
import io.milvus.grpc.CollectionSchema;
import io.milvus.grpc.DataType;
import io.milvus.grpc.FieldSchema;
import io.milvus.grpc.KeyValuePair;
import io.milvus.grpc.MutationResult;
import io.milvus.param.IndexType;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.RpcStatus;
import io.milvus.param.collection.*;
import io.milvus.param.dml.InsertParam;
import io.milvus.param.dml.QueryParam;
import io.milvus.param.dml.SearchParam;
import io.milvus.param.index.CreateIndexParam;
import io.milvus.response.QueryResultsWrapper;
import io.milvus.response.SearchResultsWrapper;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Milvus 向量数据库服务。
 *
 * <p>职责边界（强约束）：
 * <ul>
 *   <li>只对"向量 + 主键"负责——CRUD/索引/查询 Milvus 集合</li>
 *   <li>不感知任何业务字段（如 imagePath、文件名、ID 生成规则）——这些归编排层</li>
 *   <li>不调特征提取——调用方（如 {@link ImageIndexService}）应先把图片提为向量</li>
 *   <li>不做文件 IO、ID 生成、业务去重</li>
 * </ul>
 *
 * <p>所有可调参数（连接地址 / 集合名 / 批次大小 / 是否重建集合等）集中在
 * {@link MilvusProperties}，本类<b>不持有任何可调常量</b>——避免硬编码。
 *
 * <p>与 FeatureExtractor 的耦合点：仅读取 {@link FeatureExtractor#getModelName()}
 * （用于动态集合名）和 {@link FeatureExtractor#getFeatureDim()}（用于 schema 校验/建表）。
 * 不调用 {@code extractFeature(...)}，不持有特征提取线程池。
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class MilvusService {

    private final MilvusServiceClient milvusClient;
    private final FeatureExtractor featureExtractor;
    private final MilvusProperties milvusProps;

    /** 解析后的集合名（启动期根据 dynamic-collection 决定是否拼上模型名） */
    private String collectionName;

    /** Milvus schema 字段名——属于协议元数据，不是配置，写死合理 */
    private static final String PRIMARY_KEY_FIELD = "id";
    private static final String IMAGE_PATH_FIELD = "image_path";
    private static final String FEATURE_VECTOR_FIELD = "feature_vector";
    private static final String CREATED_AT_FIELD = "created_at";

    /**
     * 初始化：解析集合名 + 创建集合（如果不存在或 schema 不兼容则重建）。
     */
    @PostConstruct
    public void initCollection() {
        // 集合名初始化：先用配置的 base name
        this.collectionName = milvusProps.getCollectionName();
        // 动态集合名：开启时把 collectionName 拼上当前模型名，实现多模型数据隔离
        if (milvusProps.isDynamicCollection()) {
            String model = featureExtractor.getModelName();
            String resolved = collectionName + "_" + model;
            log.info("[动态集合名] 开关已开启: base='{}' + model='{}' -> '{}'", collectionName, model, resolved);
            this.collectionName = resolved;
        } else {
            log.info("[静态集合名] 开关已关闭: 使用固定 collectionName='{}'", collectionName);
        }

        try {
            if (collectionExists()) {
                if (milvusProps.isRecreateOnSchemaChange() && !isSchemaCompatible()) {
                    log.warn("Existing collection '{}' schema is incompatible — dropping and recreating",
                            collectionName);
                    dropCollection();
                }
            }
            if (!collectionExists()) {
                createCollection();
            }
            log.info("Milvus collection '{}' initialized successfully", collectionName);
        } catch (Exception e) {
            log.warn("Failed to initialize Milvus collection: {}", e.getMessage());
        }
    }

    /**
     * 探测集合 schema 是否与当前配置兼容。
     * 1. 能查询（主键字段存在）
     * 2. 向量维度与 featureExtractor 实际输出一致
     */
    private boolean isSchemaCompatible() {
        try {
            // 1. 检查能否查询（主键字段存在）
            R<io.milvus.grpc.QueryResults> r = milvusClient.query(
                    io.milvus.param.dml.QueryParam.newBuilder()
                            .withCollectionName(collectionName)
                            .withOutFields(Collections.singletonList(PRIMARY_KEY_FIELD))
                            .withLimit(1L)
                            .build()
            );
            if (r.getStatus() != R.Status.Success.getCode()) {
                log.warn("[Schema检查] 查询失败，判定不兼容: {}", r.getMessage());
                return false;
            }

            // 2. 校验向量维度
            int configuredDim = featureExtractor.getFeatureDim();
            int collectionDim = getCollectionVectorDimension();
            if (collectionDim > 0 && collectionDim != configuredDim) {
                log.warn("[Schema检查] 向量维度不匹配！配置 feature-dim={}，但集合维度={}。将重建集合。",
                        configuredDim, collectionDim);
                return false;
            }
            log.info("[Schema检查] 集合 '{}' 向量维度={}，配置={}，兼容。",
                    collectionName, collectionDim, configuredDim);
            return true;
        } catch (Exception e) {
            log.debug("Schema probe failed: {}", e.getMessage());
            return false;
        }
    }

    /** 查询集合中向量字段的实际维度（通过 gRPC 原生 API） */
    private int getCollectionVectorDimension() {
        try {
            R<io.milvus.grpc.DescribeCollectionResponse> r = milvusClient.describeCollection(
                    DescribeCollectionParam.newBuilder()
                            .withCollectionName(collectionName)
                            .build()
            );
            if (r.getStatus() != R.Status.Success.getCode()) return -1;
            CollectionSchema schema = r.getData().getSchema();
            for (FieldSchema field : schema.getFieldsList()) {
                if (field.getName().equals(FEATURE_VECTOR_FIELD)) {
                    for (KeyValuePair kvp : field.getTypeParamsList()) {
                        if ("dim".equals(kvp.getKey())) {
                            return Integer.parseInt(kvp.getValue());
                        }
                    }
                }
            }
        } catch (Exception e) {
            log.debug("[Schema检查] 获取集合维度失败: {}", e.getMessage());
        }
        return -1;
    }

    /**
     * 检查集合是否存在
     */
    public boolean collectionExists() {
        try {
            R<Boolean> response = milvusClient.hasCollection(
                    HasCollectionParam.newBuilder()
                            .withCollectionName(collectionName)
                            .build()
            );
            return response.getData();
        } catch (Exception e) {
            log.error("Error checking collection existence", e);
            return false;
        }
    }

    /**
     * 创建集合
     */
    public void createCollection() {
        // schema 字段：id + image_path + feature_vector + created_at
        // MilvusService 知道这些是协议字段，但不解释 imagePath 的业务含义、不读这个文件。
        List<FieldType> fields = new ArrayList<>();

        // 主键字段（业务生成：图片名+时间戳，由编排层传入）
        fields.add(FieldType.newBuilder()
                .withName(PRIMARY_KEY_FIELD)
                .withDataType(DataType.VarChar)
                .withMaxLength(200)
                .withPrimaryKey(true)
                .withAutoID(false)
                .build());

        // 业务字段：图片路径（字符串，MilvusService 不解析、不读文件）
        fields.add(FieldType.newBuilder()
                .withName(IMAGE_PATH_FIELD)
                .withDataType(DataType.VarChar)
                .withMaxLength(500)
                .build());

        // 向量字段
        fields.add(FieldType.newBuilder()
                .withName(FEATURE_VECTOR_FIELD)
                .withDataType(DataType.FloatVector)
                .withDimension(featureExtractor.getFeatureDim())
                .build());

        // 业务字段：入库时间戳（毫秒，Int64）
        fields.add(FieldType.newBuilder()
                .withName(CREATED_AT_FIELD)
                .withDataType(DataType.Int64)
                .build());

        // 2. 创建集合参数
        CreateCollectionParam createCollectionParam = CreateCollectionParam.newBuilder()
                .withCollectionName(collectionName)
                .withDescription("Image search collection - vector + business metadata")
                .withConsistencyLevel(ConsistencyLevelEnum.BOUNDED)
                .addFieldType(fields.get(0))
                .addFieldType(fields.get(1))
                .addFieldType(fields.get(2))
                .addFieldType(fields.get(3))
                .build();

        // 3. 执行创建
        R<RpcStatus> response = milvusClient.createCollection(createCollectionParam);
        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Failed to create collection: " + response.getMessage());
        }

        // 4. 创建索引
        createIndex();

        // 5. 预加载，使索引立即可用（空集合也安全）
        try {
            milvusClient.loadCollection(
                    LoadCollectionParam.newBuilder()
                            .withCollectionName(collectionName)
                            .build()
            );
        } catch (Exception e) {
            log.warn("Load after createCollection failed (non-fatal): {}", e.getMessage());
        }

        log.info("Collection '{}' created successfully with dimension {}",
                collectionName, featureExtractor.getFeatureDim());
    }

    /**
     * 创建向量字段索引
     * HNSW：基于图的近似最近邻搜索，召回率高，适合百万级数据
     * 参数说明：
     *   M=16：每层最多 16 个邻居，控制召回率与内存 trade-off
     *   efConstruction=200：构建时动态候选列表大小，越大越准但越慢
     */
    public void createIndex() {
        CreateIndexParam indexParam = CreateIndexParam.newBuilder()
                .withCollectionName(collectionName)
                .withFieldName(FEATURE_VECTOR_FIELD)
                .withIndexType(IndexType.HNSW)
                .withMetricType(MetricType.COSINE)
                .withExtraParam("{\"M\":16,\"efConstruction\":200}")
                .build();

        milvusClient.createIndex(indexParam);
        log.info("HNSW index created for field '{}' (M=16, efConstruction=200)", FEATURE_VECTOR_FIELD);
    }

    /**
     * 插入单条向量记录到 Milvus。
     *
     * <p>MilvusService 看到这个 {@link VectorRecord} 实体，把它整体持久化到 Milvus；
     * 不解析 imagePath、不读文件、不解释业务语义——只负责 CRUD。
     *
     * @param record 业务组装好的向量记录实体
     */
    public String insertVector(VectorRecord record) throws Exception {
        long t0 = System.currentTimeMillis();
        if (record == null || record.getId() == null || record.getId().isEmpty()) {
            throw new IllegalArgumentException("记录 id 不可为空");
        }
        if (record.getVector() == null || record.getVector().length == 0) {
            throw new IllegalArgumentException("记录向量不可为空");
        }

        InsertParam insertParam = InsertParam.newBuilder()
                .withCollectionName(collectionName)
                .withFields(Arrays.asList(
                        new InsertParam.Field(PRIMARY_KEY_FIELD, Collections.singletonList(record.getId())),
                        new InsertParam.Field(IMAGE_PATH_FIELD, Collections.singletonList(record.getImagePath())),
                        new InsertParam.Field(FEATURE_VECTOR_FIELD,
                                Collections.singletonList(floatArrayToList(record.getVector()))),
                        new InsertParam.Field(CREATED_AT_FIELD, Collections.singletonList(record.getCreatedAt()))
                ))
                .build();

        R<MutationResult> response = milvusClient.insert(insertParam);
        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Failed to insert vector: " + response.getMessage());
        }
        ensureCollectionLoaded();
        log.info("[单条入库] id={} | 耗时 {}ms", record.getId(), System.currentTimeMillis() - t0);
        return record.getId();
    }

    /**
     * 批量插入向量记录。
     *
     * <p>职责：本方法只负责把外部准备好的 {@link VectorRecord} 列表写入 Milvus。
     * <ul>
     *   <li>不调特征提取——record.vector 由编排层填好</li>
     *   <li>不生成 ID / imagePath / createdAt——这些由编排层填好</li>
     *   <li>不解析业务字段——record 是什么就存什么</li>
     *   <li>默认按 {@link MilvusProperties#getInsertBatchSize()} 拆批打 RPC</li>
     * </ul>
     */
    public List<String> insertVectors(List<VectorRecord> records, int batchSize) throws Exception {
        if (records == null) {
            throw new IllegalArgumentException("records 不可为 null");
        }
        if (records.isEmpty()) {
            log.info("[批量入库] 0 条，跳过");
            return Collections.emptyList();
        }
        int batchSizeEff = batchSize <= 0 ? milvusProps.getInsertBatchSize() : batchSize;

        long t0 = System.currentTimeMillis();
        int total = records.size();
        List<String> insertedIds = new ArrayList<>(total);
        long sumRpcMs = 0;
        int lastPct = -1;

        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        log.info("[批量入库] 开始: 总 {} 条 / 批 {} 条 | 模型: {}",
                total, batchSizeEff, featureExtractor.getModelName());

        for (int start = 0; start < total; start += batchSizeEff) {
            int end = Math.min(start + batchSizeEff, total);
            int subSize = end - start;

            // 1) 组装本批的 fields
            List<String> subIds = new ArrayList<>(subSize);
            List<String> subPaths = new ArrayList<>(subSize);
            List<List<Float>> subVecs = new ArrayList<>(subSize);
            List<Long> subCreatedAts = new ArrayList<>(subSize);
            for (int k = start; k < end; k++) {
                VectorRecord r = records.get(k);
                if (r == null || r.getVector() == null || r.getVector().length == 0) continue;
                subIds.add(r.getId());
                subPaths.add(r.getImagePath());
                subVecs.add(floatArrayToList(r.getVector()));
                subCreatedAts.add(r.getCreatedAt());
            }
            if (subIds.isEmpty()) continue;

            // 2) 调 Milvus insert
            List<InsertParam.Field> fields = new ArrayList<>();
            fields.add(new InsertParam.Field(PRIMARY_KEY_FIELD, subIds));
            fields.add(new InsertParam.Field(IMAGE_PATH_FIELD, subPaths));
            fields.add(new InsertParam.Field(FEATURE_VECTOR_FIELD, subVecs));
            fields.add(new InsertParam.Field(CREATED_AT_FIELD, subCreatedAts));

            InsertParam insertParam = InsertParam.newBuilder()
                    .withCollectionName(collectionName)
                    .withFields(fields)
                    .build();

            long rpcT0 = System.currentTimeMillis();
            R<MutationResult> resp = milvusClient.insert(insertParam);
            long rpcMs = System.currentTimeMillis() - rpcT0;
            sumRpcMs += rpcMs;
            log.info("[批量入库] Milvus insert RPC: 本批 {} 条 | 耗时 {}ms ({}ms/条)",
                    subIds.size(), rpcMs, rpcMs / subIds.size());
            if (resp.getStatus() != R.Status.Success.getCode()) {
                throw new RuntimeException("Milvus insert 失败 (本批 " + subIds.size() + " 条): " + resp.getMessage());
            }
            insertedIds.addAll(subIds);

            // 3) 进度日志：每 10% 一行
            int pct = (int) ((end * 100L) / total);
            if (pct / 10 != lastPct / 10) {
                long elapsed = System.currentTimeMillis() - t0;
                long etaMs = pct == 0 ? 0 : (long) ((double) elapsed / pct * (100 - pct));
                log.info("[批量入库] 进度 {}/{} ({}%) | 已用 {}s | 预计剩余 ~{}s",
                        end, total, pct, elapsed / 1000, etaMs / 1000);
            }
            lastPct = pct;
        }

        long totalMs = System.currentTimeMillis() - t0;
        double throughput = totalMs == 0 ? 0.0 : (insertedIds.size() * 1000.0 / totalMs);
        log.info("[批量入库] 完成: 成功 {} / {} 条，耗时 {}ms (平均 {}ms/条, 吞吐 {} img/s)",
                insertedIds.size(), total, totalMs, totalMs / Math.max(1, total),
                String.format("%.2f", throughput));
        log.info("[批量入库] insert RPC 累计 {}ms ({:.0f}%)",
                sumRpcMs, 100.0 * sumRpcMs / Math.max(1, totalMs));
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        return insertedIds;
    }

    /**
     * 批量插入向量记录（默认批次大小）。
     */
    public List<String> insertVectors(List<VectorRecord> records) throws Exception {
        return insertVectors(records, 0);
    }

    /**
     * 以图搜图（直接传向量，跳过特征提取步骤）
     * 用于查询图只活在内存里（上传的 MultipartFile 等）不想落盘的场景
     *
     * <p>返回结果包含 schema 字段（id / imagePath / createdAt）+ 相似度。
     * 不返回 vector（搜索结果不需要向量本身，避免冗余传输）。
     */
    public List<SearchResult> searchByVector(float[] queryVector, int topK) throws Exception {
        if (queryVector == null || queryVector.length == 0) {
            throw new IllegalArgumentException("查询向量为空");
        }
        List<List<Float>> searchVectors = Collections.singletonList(floatArrayToList(queryVector));

        // 1.5 确保集合已加载（如果索引丢了，自动重建）
        ensureCollectionLoaded();

        // 2. 构建搜索参数：把 schema 字段一起带回来供编排层用
        SearchParam searchParam = SearchParam.newBuilder()
                .withCollectionName(collectionName)
                .withVectorFieldName(FEATURE_VECTOR_FIELD)
                .withFloatVectors(searchVectors)
                .withTopK(topK)
                .withParams("{\"ef\":256}")
                .withConsistencyLevel(ConsistencyLevelEnum.BOUNDED)
                .withOutFields(Lists.newArrayList(IMAGE_PATH_FIELD, CREATED_AT_FIELD))
                .build();

        // 3. 执行搜索
        R<io.milvus.grpc.SearchResults> response = milvusClient.search(searchParam);
        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Failed to search: " + response.getMessage());
        }

        // 4. 解析结果
        SearchResultsWrapper wrapper = new SearchResultsWrapper(response.getData().getResults());
        List<SearchResultsWrapper.IDScore> scores = wrapper.getIDScore(0);

        List<SearchResult> results = new ArrayList<>();
        for (SearchResultsWrapper.IDScore score : scores) {
            SearchResult result = new SearchResult();
            result.setId(score.getStrID());
            float rawScore = score.getScore();
            result.setScore(rawScore);
            // COSINE ∈ [-1, 1] → 0~100%；归一化：(score + 1) / 2 ∈ [0, 1]
            float similarity = Math.clamp((rawScore + 1f) / 2f, 0f, 1f) * 100f;
            result.setSimilarity(String.format("%.2f%%", similarity));
            Object imagePath = score.get(IMAGE_PATH_FIELD);
            if (imagePath != null) result.setImagePath(imagePath.toString());
            Object ts = score.get(CREATED_AT_FIELD);
            if (ts instanceof Number) result.setCreatedAt(((Number) ts).longValue());
            results.add(result);
        }

        return results;
    }

    /**
     * 删除集合
     */
    public void dropCollection() {
        milvusClient.dropCollection(
                DropCollectionParam.newBuilder()
                        .withCollectionName(collectionName)
                        .build()
        );
        log.info("Collection '{}' dropped", collectionName);
    }

    /**
     * 按 id 列表查询完整的 {@link VectorRecord} 实体（含 imagePath、vector、createdAt）。
     *
     * <p>典型用法：搜索后编排层拿 id 列表 → 调这个方法拿到 imagePath → 拼装业务结果回显。
     *
     * @param ids 要查询的主键列表
     * @return 与 ids 对齐的 record 列表；找不到的 id 对应 record 为 null
     */
    public List<VectorRecord> queryByIds(List<String> ids) {
        if (ids == null || ids.isEmpty()) {
            return Collections.emptyList();
        }
        // 用 expr 表达 "id in [...]"——Milvus Java SDK 的 QueryParam 没有 withIds
        // 字符串值需双引号包起来，内部双引号需转义
        StringBuilder expr = new StringBuilder(PRIMARY_KEY_FIELD + " in [");
        for (int i = 0; i < ids.size(); i++) {
            if (i > 0) expr.append(",");
            expr.append("\"").append(ids.get(i).replace("\"", "\\\"")).append("\"");
        }
        expr.append("]");

        try {
            R<io.milvus.grpc.QueryResults> qr = milvusClient.query(
                    QueryParam.newBuilder()
                            .withCollectionName(collectionName)
                            .withOutFields(Arrays.asList(
                                    PRIMARY_KEY_FIELD,
                                    IMAGE_PATH_FIELD,
                                    FEATURE_VECTOR_FIELD,
                                    CREATED_AT_FIELD))
                            .withExpr(expr.toString())
                            .build()
            );
            if (qr.getStatus() != R.Status.Success.getCode()) {
                log.warn("[queryByIds] 失败: {}", qr.getMessage());
                return Collections.emptyList();
            }
            QueryResultsWrapper qw = new QueryResultsWrapper(qr.getData());
            Map<String, VectorRecord> byId = new HashMap<>();
            for (QueryResultsWrapper.RowRecord row : qw.getRowRecords()) {
                Object idObj = row.get(PRIMARY_KEY_FIELD);
                if (idObj == null) continue;
                String id = idObj.toString();
                VectorRecord rec = new VectorRecord();
                rec.setId(id);
                Object pathObj = row.get(IMAGE_PATH_FIELD);
                rec.setImagePath(pathObj == null ? null : pathObj.toString());
                Object vecObj = row.get(FEATURE_VECTOR_FIELD);
                if (vecObj instanceof List<?> vecList) {
                    float[] arr = new float[vecList.size()];
                    for (int i = 0; i < vecList.size(); i++) {
                        Object v = vecList.get(i);
                        arr[i] = (v instanceof Number) ? ((Number) v).floatValue() : 0f;
                    }
                    rec.setVector(arr);
                }
                Object tsObj = row.get(CREATED_AT_FIELD);
                if (tsObj instanceof Number) {
                    rec.setCreatedAt(((Number) tsObj).longValue());
                }
                byId.put(id, rec);
            }
            // 按输入 ids 顺序对齐返回
            List<VectorRecord> out = new ArrayList<>(ids.size());
            for (String id : ids) out.add(byId.get(id));
            return out;
        } catch (Exception e) {
            log.warn("[queryByIds] 异常: {}", e.getMessage());
            return Collections.emptyList();
        }
    }

    /**
     * 获取集合信息
     */
    public Map<String, Object> getCollectionInfo() {
        Map<String, Object> info = new HashMap<>();
        info.put("collectionName", collectionName);
        info.put("dimension", featureExtractor.getFeatureDim());
        info.put("exists", collectionExists());
        return info;
    }

    /**
     * 获取集合统计信息（行数 + 样本图片）
     */
    public Map<String, Object> getCollectionStats(int sampleSize) {
        Map<String, Object> result = new HashMap<>();
        result.put("collectionName", collectionName);
        result.put("exists", collectionExists());

        if (!collectionExists()) {
            result.put("rowCount", 0L);
            result.put("samples", Collections.emptyList());
            return result;
        }

        // 先保证集合被加载，避免 cold collection 时 query 失败
        try {
            ensureCollectionLoaded();
        } catch (Exception e) {
            log.warn("ensureCollectionLoaded failed: {}", e.getMessage());
            result.put("samples", Collections.emptyList());
            result.putIfAbsent("rowCount", 0L);
            return result;
        }

        long rowCount = 0;
        try {
            R<io.milvus.grpc.QueryResults> qr = milvusClient.query(
                    QueryParam.newBuilder()
                            .withCollectionName(collectionName)
                            .withOutFields(Lists.newArrayList(PRIMARY_KEY_FIELD, IMAGE_PATH_FIELD, CREATED_AT_FIELD))
                            .withLimit((long)sampleSize)
                            .build()
            );
            if (qr.getStatus() == R.Status.Success.getCode()) {
                QueryResultsWrapper qw = new QueryResultsWrapper(qr.getData());
                List<QueryResultsWrapper.RowRecord> rows = qw.getRowRecords();
                List<Map<String, Object>> samples = new ArrayList<>();
                for (QueryResultsWrapper.RowRecord row : rows) {
                    Map<String, Object> item = new HashMap<>();
                    Object idObj = row.get(PRIMARY_KEY_FIELD);
                    Object pathObj = row.get(IMAGE_PATH_FIELD);
                    Object tsObj = row.get(CREATED_AT_FIELD);
                    item.put("id", idObj == null ? "" : idObj.toString());
                    item.put("imagePath", pathObj == null ? null : pathObj.toString());
                    if (tsObj instanceof Number) {
                        item.put("createdAt", ((Number) tsObj).longValue());
                    }
                    samples.add(item);
                }
                result.put("samples", samples);
            } else {
                result.put("samples", Collections.emptyList());
            }
        } catch (Exception e) {
            log.warn("Failed to fetch samples: {}", e.getMessage());
            result.put("samples", Collections.emptyList());
        }

        // 行数：尝试拿统计
        try {
            R<io.milvus.grpc.GetCollectionStatisticsResponse> stats = milvusClient.getCollectionStatistics(
                    GetCollectionStatisticsParam.newBuilder()
                            .withCollectionName(collectionName)
                            .build()
            );
            if (stats.getStatus() == R.Status.Success.getCode()) {
                for (io.milvus.grpc.KeyValuePair kv : stats.getData().getStatsList()) {
                    if ("row_count".equals(kv.getKey())) {
                        rowCount = Long.parseLong(kv.getValue());
                        break;
                    }
                }
            }
        } catch (Exception e) {
            log.warn("Failed to fetch row count: {}", e.getMessage());
        }
        result.put("rowCount", rowCount);
        return result;
    }

    /**
     * 确保集合已被加载到内存。如果遇到索引丢失，会自动重建索引并重新加载。
     * 供编排层（如 ImageIndexService）在批量入库前显式调用。
     */
    public void ensureCollectionLoaded() {
        try {
            R<RpcStatus> r = milvusClient.loadCollection(
                    LoadCollectionParam.newBuilder()
                            .withCollectionName(collectionName)
                            .build()
            );
            if (r.getStatus() == R.Status.Success.getCode()) {
                return;
            }
            log.warn("loadCollection returned non-success: {} — will rebuild index", r.getMessage());
        } catch (Exception e) {
            log.warn("loadCollection failed ({}), attempting to rebuild index", e.getMessage());
        }

        // 索引可能丢了 → 重建索引并加载
        try {
            milvusClient.dropIndex(
                    io.milvus.param.index.DropIndexParam.newBuilder()
                            .withCollectionName(collectionName)
                            .withIndexName(FEATURE_VECTOR_FIELD)
                            .build()
            );
        } catch (Exception ignore) {
        }
        try {
            createIndex();
        } catch (Exception e) {
            log.error("Failed to rebuild index: {}", e.getMessage());
            throw new RuntimeException("Failed to rebuild Milvus index: " + e.getMessage(), e);
        }
        // createIndex 是异步的，需要 flush 等索引构建完才能 load
        try {
            milvusClient.flush(
                    FlushParam.newBuilder()
                            .withCollectionNames(Collections.singletonList(collectionName))
                            .build()
            );
        } catch (Exception e) {
            log.warn("Flush after rebuild index failed (non-fatal): {}", e.getMessage());
        }
        try {
            milvusClient.loadCollection(
                    LoadCollectionParam.newBuilder()
                            .withCollectionName(collectionName)
                            .build()
            );
            log.info("Collection '{}' rebuilt index and reloaded", collectionName);
        } catch (Exception e) {
            log.error("Reload after rebuild failed: {}", e.getMessage());
            throw new RuntimeException("Failed to reload Milvus collection: " + e.getMessage(), e);
        }
    }

    // ============ 辅助方法 ============

    private List<Float> floatArrayToList(float[] array) {
        List<Float> list = new ArrayList<>(array.length);
        for (float v : array) {
            list.add(v);
        }
        return list;
    }
}
