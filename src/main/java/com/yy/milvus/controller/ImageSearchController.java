package com.yy.milvus.controller;

import com.yy.milvus.domain.QueryCondition;
import com.yy.milvus.domain.QueryResult;
import com.yy.milvus.domain.SearchResult;
import com.yy.milvus.service.ImageIndexService;
import com.yy.milvus.service.MilvusService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.util.*;

/**
 * 图片搜索 REST API 控制器
 */
@RestController
@RequestMapping("/api/image-search")
@RequiredArgsConstructor
@Slf4j
@CrossOrigin(origins = "*")
public class ImageSearchController {

    private final ImageIndexService imageIndexService;
    private final MilvusService milvusService;

    /**
     * 上传并索引单张图片
     */
    @PostMapping("/upload")
    public ResponseEntity<Map<String, Object>> uploadImage(@RequestParam("file") MultipartFile file) {
        try {
            String id = imageIndexService.index(file);
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("id", id);
            response.put("imagePath", toUrl("images/" + id + guessExt(file)));
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            log.error("Failed to upload image", e);
            return badRequest(e.getMessage());
        }
    }

    /**
     * 批量上传并索引
     */
    @PostMapping("/upload/batch")
    public ResponseEntity<Map<String, Object>> uploadImages(@RequestParam("files") MultipartFile[] files) {
        try {
            List<String> imagePaths = new ArrayList<>();
            for (MultipartFile file : files) {
                String path = imageIndexService.index(file);
                imagePaths.add(path);
            }
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("inserted", imagePaths.size());
            response.put("paths", imagePaths);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            log.error("Failed to upload images", e);
            return badRequest(e.getMessage());
        }
    }

    /**
     * 以图搜图 - 上传图片
     */
    @PostMapping("/search")
    public ResponseEntity<Map<String, Object>> searchByImage(
            @RequestParam("file") MultipartFile queryFile,
            @RequestParam(value = "topK", defaultValue = "10") int topK) {
        try {
            List<SearchResult> results = imageIndexService.searchByStream(
                    queryFile.getInputStream(), topK);
            results.forEach(r -> r.setImagePath(toUrl(r.getImagePath())));

            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("queryImageName", queryFile.getOriginalFilename());
            response.put("totalResults", results.size());
            response.put("results", results);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            log.error("Failed to search", e);
            return badRequest(e.getMessage() == null ? e.toString() : e.getMessage());
        }
    }

    /**
     * 以图搜图 - 指定已有图片路径
     */
    @GetMapping("/search")
    public ResponseEntity<Map<String, Object>> searchByPath(
            @RequestParam("path") String imagePath,
            @RequestParam(value = "topK", defaultValue = "10") int topK) {
        try {
            List<SearchResult> results = imageIndexService.searchByPath(imagePath, topK);
            results.forEach(r -> r.setImagePath(toUrl(r.getImagePath())));

            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("queryImage", toUrl(imagePath));
            response.put("totalResults", results.size());
            response.put("results", results);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            log.error("Failed to search", e);
            return badRequest(e.getMessage());
        }
    }

    /**
     * 获取集合信息
     */
    @GetMapping("/info")
    public ResponseEntity<Map<String, Object>> getInfo() {
        try {
            return ResponseEntity.ok(milvusService.getCollectionInfo());
        } catch (Exception e) {
            log.error("Failed to get info", e);
            return badRequest(e.getMessage());
        }
    }

    /**
     * 获取集合统计
     */
    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> getStats(
            @RequestParam(value = "sampleSize", defaultValue = "50") int sampleSize) {
        try {
            Map<String, Object> stats = milvusService.getCollectionStats(sampleSize);
            if (stats.containsKey("samples")) {
                List<Map<String, Object>> samples = (List<Map<String, Object>>) stats.get("samples");
                samples.forEach(s -> s.put("imagePath", toUrl((String) s.get("imagePath"))));
            }
            return ResponseEntity.ok(stats);
        } catch (Exception e) {
            log.error("Failed to get stats", e);
            return badRequest(e.getMessage());
        }
    }

    // ==================== 字段查询 / 条件查询 ====================

    /**
     * 按主键 id 查询单条记录。
     */
    @GetMapping("/query/id/{id}")
    public ResponseEntity<Map<String, Object>> queryById(@PathVariable("id") String id) {
        try {
            QueryResult r = imageIndexService.queryById(id);
            if (r == null) {
                return ResponseEntity.ok(Map.of("success", true, "found", false, "id", id));
            }
            r.setImagePath(toUrl(r.getImagePath()));
            return ResponseEntity.ok(Map.of("success", true, "found", true, "result", r));
        } catch (Exception e) {
            log.error("Failed to query by id", e);
            return badRequest(e.getMessage());
        }
    }

    /**
     * 按图片路径精确查询。
     */
    @GetMapping("/query/path")
    public ResponseEntity<Map<String, Object>> queryByPath(@RequestParam("path") String imagePath) {
        try {
            List<QueryResult> results = imageIndexService.queryByImagePath(imagePath);
            results.forEach(r -> r.setImagePath(toUrl(r.getImagePath())));
            Map<String, Object> resp = new HashMap<>();
            resp.put("success", true);
            resp.put("queryPath", toUrl(imagePath));
            resp.put("totalResults", results.size());
            resp.put("results", results);
            return ResponseEntity.ok(resp);
        } catch (Exception e) {
            log.error("Failed to query by path", e);
            return badRequest(e.getMessage());
        }
    }

    /**
     * 按图片路径模糊查询（LIKE '%path%'）。
     */
    @GetMapping("/query/path/like")
    public ResponseEntity<Map<String, Object>> queryByPathLike(
            @RequestParam("keyword") String keyword,
            @RequestParam(value = "limit", defaultValue = "100") int limit) {
        try {
            int lim = limit <= 0 ? 100 : Math.min(limit, 1000);
            QueryCondition cond = QueryCondition.builder()
                    .contains("image_path", keyword)
                    .build();
            List<QueryResult> results = imageIndexService.queryByCondition(cond, false, lim);
            results.forEach(r -> r.setImagePath(toUrl(r.getImagePath())));
            Map<String, Object> resp = new HashMap<>();
            resp.put("success", true);
            resp.put("keyword", keyword);
            resp.put("totalResults", results.size());
            resp.put("results", results);
            return ResponseEntity.ok(resp);
        } catch (Exception e) {
            log.error("Failed to query by path like", e);
            return badRequest(e.getMessage());
        }
    }

    /**
     * 按入库时间范围查询。
     */
    @GetMapping("/query/time-range")
    public ResponseEntity<Map<String, Object>> queryByTimeRange(
            @RequestParam(value = "from", required = false) Long from,
            @RequestParam(value = "to", required = false) Long to,
            @RequestParam(value = "limit", defaultValue = "100") int limit) {
        try {
            int lim = limit <= 0 ? 100 : Math.min(limit, 1000);
            List<QueryResult> results = imageIndexService.queryByCreatedAtRange(from, to);
            // 业务层不限 limit，这里手工截
            if (results.size() > lim) results = results.subList(0, lim);
            results.forEach(r -> r.setImagePath(toUrl(r.getImagePath())));
            Map<String, Object> resp = new HashMap<>();
            resp.put("success", true);
            resp.put("from", from);
            resp.put("to", to);
            resp.put("totalResults", results.size());
            resp.put("results", results);
            return ResponseEntity.ok(resp);
        } catch (Exception e) {
            log.error("Failed to query by time range", e);
            return badRequest(e.getMessage());
        }
    }

    /**
     * 通用条件查询（POST + JSON body）。
     *
     * <p>请求体格式：
     * <pre>{@code
     * {
     *   "imagePath": "images/cat.jpg",       // 可选：image_path 等值匹配
     *   "pathPrefix": "images/",              // 可选：image_path 前缀
     *   "pathContains": "cat",                // 可选：image_path 包含
     *   "idIn": ["a", "b"],                   // 可选：id IN [...]
     *   "from": 1700000000000,                // 可选：created_at 下界（毫秒，含）
     *   "to":   1800000000000,                // 可选：created_at 上界（毫秒，含）
     *   "withVector": false,                  // 可选：是否返回向量（默认 false）
     *   "limit": 100                          // 可选：返回条数上限（默认 100，最大 1000）
     * }
     * }</pre>
     */
    @PostMapping("/query")
    public ResponseEntity<Map<String, Object>> queryByCondition(@RequestBody Map<String, Object> body) {
        try {
            QueryCondition.Builder b = QueryCondition.builder();

            String id = strOrNull(body, "id");
            if (id != null) b.eq("id", id);

            String imagePath = strOrNull(body, "imagePath");
            if (imagePath != null) b.eq("image_path", imagePath);

            String pathPrefix = strOrNull(body, "pathPrefix");
            if (pathPrefix != null) b.startsWith("image_path", pathPrefix);

            String pathContains = strOrNull(body, "pathContains");
            if (pathContains != null) b.contains("image_path", pathContains);

            Object idInRaw = body.get("idIn");
            if (idInRaw instanceof List<?> idInList) {
                List<String> ids = new ArrayList<>();
                for (Object o : idInList) if (o != null) ids.add(o.toString());
                if (!ids.isEmpty()) b.in("id", ids);
            }

            Long from = longOrNull(body, "from");
            if (from != null) b.gte("created_at", from);

            Long to = longOrNull(body, "to");
            if (to != null) b.lte("created_at", to);

            boolean withVector = Boolean.TRUE.equals(body.get("withVector"));
            int limit = intOrDefault(body, "limit", 100);
            if (limit <= 0) limit = 100;
            limit = Math.min(limit, 1000);

            List<QueryResult> results = imageIndexService.queryByCondition(b.build(), withVector, limit);
            results.forEach(r -> r.setImagePath(toUrl(r.getImagePath())));

            Map<String, Object> resp = new HashMap<>();
            resp.put("success", true);
            resp.put("expr", b.build().toExpr());
            resp.put("withVector", withVector);
            resp.put("totalResults", results.size());
            resp.put("results", results);
            return ResponseEntity.ok(resp);
        } catch (Exception e) {
            log.error("Failed to query by condition", e);
            return badRequest(e.getMessage());
        }
    }

    /**
     * 高级：直接传 Milvus expr 字符串（不走白名单）。
     * <b>注意</b>：此接口无字段名白名单校验，调用方需自行保证 expr 安全。
     */
    @PostMapping("/query/raw")
    public ResponseEntity<Map<String, Object>> queryByRawExpr(@RequestBody Map<String, Object> body) {
        try {
            String expr = strOrNull(body, "expr");
            if (expr == null) throw new IllegalArgumentException("expr 不可为空");
            boolean withVector = Boolean.TRUE.equals(body.get("withVector"));
            int limit = intOrDefault(body, "limit", 100);
            if (limit <= 0) limit = 100;
            limit = Math.min(limit, 1000);

            List<QueryResult> results = imageIndexService.queryByRawExpr(expr, withVector, limit);
            results.forEach(r -> r.setImagePath(toUrl(r.getImagePath())));

            Map<String, Object> resp = new HashMap<>();
            resp.put("success", true);
            resp.put("expr", expr);
            resp.put("withVector", withVector);
            resp.put("totalResults", results.size());
            resp.put("results", results);
            return ResponseEntity.ok(resp);
        } catch (Exception e) {
            log.error("Failed to query by raw expr", e);
            return badRequest(e.getMessage());
        }
    }

    private static String strOrNull(Map<String, Object> body, String key) {
        Object v = body.get(key);
        return v == null ? null : v.toString();
    }

    private static Long longOrNull(Map<String, Object> body, String key) {
        Object v = body.get(key);
        if (v == null) return null;
        if (v instanceof Number n) return n.longValue();
        try {
            return Long.parseLong(v.toString());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("字段 " + key + " 不是合法 Long: " + v);
        }
    }

    private static int intOrDefault(Map<String, Object> body, String key, int def) {
        Object v = body.get(key);
        if (v == null) return def;
        if (v instanceof Number n) return n.intValue();
        try {
            return Integer.parseInt(v.toString());
        } catch (NumberFormatException e) {
            return def;
        }
    }

    /**
     * 删除并重建集合
     */
    @DeleteMapping("/reset")
    public ResponseEntity<Map<String, Object>> resetCollection() {
        try {
            milvusService.dropCollection();
            milvusService.initCollection();
            return ResponseEntity.ok(Map.of("success", true, "message", "Collection reset successfully"));
        } catch (Exception e) {
            log.error("Failed to reset collection", e);
            return badRequest(e.getMessage());
        }
    }

    /**
     * 从本地目录批量导入
     */
    @GetMapping("/import")
    public ResponseEntity<Map<String, Object>> importFromDir(
            @RequestParam(value = "dir", defaultValue = "E:/Desktop/images") String dir,
            @RequestParam(value = "recursive", defaultValue = "true") boolean recursive,
            @RequestParam(value = "batchSize", defaultValue = "16") int batchSize) {
        log.info("[Import] dir={}, recursive={}, batchSize={}", dir, recursive, batchSize);
        try {
            Map<String, Object> result = imageIndexService.importFromDirectory(dir, recursive, batchSize);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            log.error("Failed to import from directory", e);
            return badRequest(e.getMessage());
        }
    }

    /**
     * 转换磁盘路径为 HTTP URL
     */
    private String toUrl(String diskPath) {
        if (diskPath == null || diskPath.isEmpty()) return diskPath;
        String normalized = diskPath.replace(File.separatorChar, '/');
        int idx = normalized.lastIndexOf("images/");
        return idx >= 0 ? "/" + normalized.substring(idx) : diskPath;
    }

    private String guessExt(MultipartFile file) {
        String name = file.getOriginalFilename();
        if (name != null) {
            int dot = name.lastIndexOf('.');
            if (dot > 0) return name.substring(dot);
        }
        return ".jpg";
    }

    private ResponseEntity<Map<String, Object>> badRequest(String msg) {
        return ResponseEntity.badRequest().body(Map.of("success", false, "error", msg));
    }
}
