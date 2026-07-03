package com.yy.milvus.config;

import io.milvus.client.MilvusServiceClient;
import io.milvus.param.ConnectParam;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Milvus 客户端 Bean 定义。
 *
 * <p>注意：Milvus 的<b>配置字段</b>（host / port / collection-name / batch-size 等）
 * 全部在 {@link MilvusProperties}；本类只负责用这些配置构造 {@link MilvusServiceClient}。
 *
 * <p>这样拆的好处：
 * <ul>
 *   <li>配置归 {@link MilvusProperties}（yml 映射），无侵入</li>
 *   <li>Bean 装配归 {@code MilvusConfig}（@Configuration），逻辑清晰</li>
 *   <li>{@link com.yy.milvus.service.MilvusService} 只注入需要的 Properties，不持有连接逻辑</li>
 * </ul>
 */
@Configuration
public class MilvusConfig {

    @Bean(destroyMethod = "close")
    public MilvusServiceClient milvusServiceClient(MilvusProperties props) {
        ConnectParam connectParam = ConnectParam.newBuilder()
                .withHost(props.getHost())
                .withPort(props.getPort())
                .build();
        return new MilvusServiceClient(connectParam);
    }
}