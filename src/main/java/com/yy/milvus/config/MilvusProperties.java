package com.yy.milvus.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Milvus 配置类。
 *
 * <p>存放 Milvus 相关的可调整参数，在 yml 的 {@code milvus} 节点下，
 * 覆盖 {@link com.yy.milvus.service.MilvusService} 中硬编码常量。
 *
 * <p>字段含义：
 * <ul>
 *   <li>{@code host / port}                  … Milvus 服务连接地址</li>
 *   <li>{@code collection-name}              … 集合名前缀，dynamic-collection=true 时拼入模块名后缀</li>
 *   <li>{@code dynamic-collection}           … 是否按模块自动拼接集合名</li>
 *   <li>{@code recreate-on-schema-change}    … 启动时检测到 schema 变更时是否重建集合</li>
 *   <li>{@code insert-batch-size}            … 单次 insert RPC 的批处理最大行数</li>
 * </ul>
 */
@Data
@ConfigurationProperties(prefix = "milvus")
public class MilvusProperties {

    /** Milvus 服务地址 */
    private String host = "localhost";

    /** Milvus 服务端口 */
    private int port = 19530;

    /** 集合名前缀，dynamic-collection=true 时自动拼接当前模块名 */
    private String collectionName = "image_search";

    /** 是否启用动态集合名（自动拼接模块名后缀） */
    private boolean dynamicCollection = true;

    /**
     * 将 ID 类型从 Int64 改成 VarChar 时，旧 collection 的 schema 已不兼容。
     * 设为 true 则启动时强制 drop 旧 collection 并用新 schema 重建（数据会丢失）。
     */
    private boolean recreateOnSchemaChange = true;

    /** 单次 Milvus insert RPC 的批处理最大行数 */
    private int insertBatchSize = 64;
}
