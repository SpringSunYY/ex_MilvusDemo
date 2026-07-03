package com.yy.milvus;

import com.yy.milvus.config.EmbeddingProperties;
import com.yy.milvus.config.MilvusProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

@SpringBootApplication
@EnableConfigurationProperties({EmbeddingProperties.class, MilvusProperties.class})
public class MilvusDemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(MilvusDemoApplication.class, args);
    }
}
