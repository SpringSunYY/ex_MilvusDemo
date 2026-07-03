package com.yy.milvus.domain;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 向量记录实体（Milvus collection 中的一行）。
 *
 * <p>作为服务层与 Milvus 持久层之间的<b>传输对象</b>。
 * <ul>
 *   <li>{@link MilvusService} 不关心此实体，只知道要把行写入 Milvus 集合，
 *       对<b>业务层</b> imagePath 字段。至于 id、何种 vector 算法，服务层只负责 CRUD。</li>
 *   <li>业务层（如 {@link ImageIndexService}）将"业务要装到 VectorRecord"加工出来，
 *       写什么字段是业务层自己的事。</li>
 * </ul>
 *
 * <p>字段对应 Milvus schema：
 * <pre>
 *   id              VarChar        主键
 *   image_path      VarChar        业务字段，只存字符串，不存文件
 *   feature_vector  FloatVector    归一化向量
 *   created_at      Int64          业务时间（秒级时间戳）
 * </pre>
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class VectorRecord {

    /** 主键（由调用方生成） */
    private String id;

    /** 图片路径（业务字段），仅作为字符串持久化，MilvusService 不关心文件 */
    private String imagePath;

    /** 已归一化的特征向量 */
    private float[] vector;

    /** 业务时间（秒级时间戳） */
    private long createdAt;

    /** 两字段构造：只需要 id + vector，用于异步构建索引 */
    public VectorRecord(String id, float[] vector) {
        this.id = id;
        this.vector = vector;
    }
}
