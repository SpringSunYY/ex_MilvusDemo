package com.yy.milvus.controller;

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
