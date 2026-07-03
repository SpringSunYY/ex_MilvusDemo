package com.yy.milvus.domain;

import lombok.Data;

/**
 * 业务层搜索结果（含 imagePath）。
 *
 * <p>由编排层 {@link ImageIndexService} 在 Milvus 纯向量搜索结果之上反查 id→imagePath 拼装而成。
 * 不要把它当成 MilvusService 的返回类型——它属于业务编排层的概念。
 */
@Data
public class SearchHit {
    private String id;
    private String imagePath;
    /**
     * Milvus COSINE 距离 ∈ [-1, 1]，越大越像
     */
    private float score;
    /**
     * 相似度百分比 ∈ [0, 100]
     */
    private String similarity;
}
