package com.yy.milvus.config;

import org.springframework.boot.web.embedded.tomcat.TomcatServletWebServerFactory;
import org.springframework.boot.web.server.WebServerFactoryCustomizer;
import org.springframework.context.annotation.Configuration;

@Configuration
public class TomcatConfig implements WebServerFactoryCustomizer<TomcatServletWebServerFactory> {

    @Override
    public void customize(TomcatServletWebServerFactory factory) {
        factory.addConnectorCustomizers(connector -> {
            // -1 means unlimited; otherwise Tomcat rejects the request with 413
            // before it ever reaches Spring's multipart parser.
            connector.setProperty("maxPostSize", "-1");
            connector.setProperty("maxSavePostSize", "-1");
            connector.setProperty("maxSwallowSize", "-1");
        });
    }
}