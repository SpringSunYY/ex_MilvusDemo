package com.yy.milvus.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.UUID;

/**
 * 文件存储服务：只负责文件的保存、读取、删除
 * 与业务逻辑解耦，方便后续替换存储后端（如 OSS、MinIO）
 */
@Service
@Slf4j
public class FileStorageService {

    private static final String UPLOAD_DIR = "images/";
    private static final String[] IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"};

    public FileStorageService() {
        ensureUploadDir();
    }

    private void ensureUploadDir() {
        File dir = new File(UPLOAD_DIR);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }

    /**
     * 保存上传的文件，返回存储路径
     *
     * @param file     MultipartFile
     * @param uniqueId 可选的唯一标识（null 则自动生成 UUID）
     * @return 保存后的相对路径，如 "images/xxx.jpg"
     */
    public String save(MultipartFile file, String uniqueId) throws IOException {
        validateImage(file);

        String originalName = file.getOriginalFilename();
        String ext = getExtension(originalName);
        String savedName;

        if (uniqueId != null && !uniqueId.isBlank()) {
            savedName = sanitizeFileName(uniqueId) + ext;
        } else {
            savedName = UUID.randomUUID().toString() + ext;
        }

        File destFile = resolveUniqueFile(new File(UPLOAD_DIR), savedName);
        Files.copy(file.getInputStream(), destFile.toPath(), StandardCopyOption.REPLACE_EXISTING);

        log.info("[文件存储] 保存成功: {} -> {}", originalName, destFile.getName());
        return UPLOAD_DIR + destFile.getName();
    }

    /**
     * 保存上传文件，自动生成唯一文件名（保留原扩展名）
     */
    public String save(MultipartFile file) throws IOException {
        return save(file, null);
    }

    /**
     * 保存上传文件，<b>优先复用</b> uploads 目录下与上传文件<b>原始文件名</b>相同的已有非空文件。
     * 命中时直接返回已存在文件的路径，跳过落盘 —— 避免重复上传同名图片导致磁盘堆积。
     * 命中失败（文件不存在 / 已被占用为空文件）才走正常 save() 流程。
     * <p>
     * 注意：复用只复用磁盘文件，Milvus 入库仍由调用方按原逻辑处理（每次上传都会插入新记录，
     * 与 {@link #copyFromPath(String)} 的语义保持一致 —— 文件层幂等，索引层不强求幂等）。
     */
    public String saveOrReuse(MultipartFile file) throws IOException {
        validateImage(file);
        String originalName = file.getOriginalFilename();
        File existing = findByName(originalName);
        if (existing != null) {
            log.info("[文件存储] 上传命中复用（同名已存在）: {} -> {}", originalName, existing.getName());
            return UPLOAD_DIR + existing.getName();
        }
        return save(file, null);
    }

    /**
     * 从源路径复制文件到存储目录。
     * <p>
     * 命中规则：images/ 目录下已存在与源文件<b>文件名（含扩展名）</b>相同的非空文件时，
     * 直接复用，不重复落盘 —— 避免反复导入同名图导致磁盘堆积。
     * 注意：这里只复用磁盘文件，Milvus 入库仍由调用方（{@code importFromDirectory}）
     * 按原逻辑新增记录。
     */
    public String copyFromPath(String sourcePath) throws IOException {
        File src = new File(sourcePath);
        if (!src.exists()) {
            throw new IOException("源文件不存在: " + sourcePath);
        }

        String srcName = src.getName();
        File existing = findByName(srcName);
        if (existing != null) {
            log.info("[文件存储] 命中复用（同名已存在）: {} -> {}", srcName, existing.getName());
            return UPLOAD_DIR + existing.getName();
        }

        File destFile = resolveUniqueFile(new File(UPLOAD_DIR), srcName);
        Files.copy(src.toPath(), destFile.toPath(), StandardCopyOption.REPLACE_EXISTING);

        log.info("[文件存储] 复制: {} -> {}", sourcePath, destFile.getName());
        return UPLOAD_DIR + destFile.getName();
    }

    /**
     * 在 uploads 目录下按文件名查找已有文件。
     * 只比对文件名（含扩展名），忽略大小写；不递归子目录。
     *
     * @return 命中的文件；不存在或文件为空时返回 null
     */
    private File findByName(String fileName) {
        File dir = new File(UPLOAD_DIR);
        File[] matches = dir.listFiles((d, name) -> name.equalsIgnoreCase(fileName));
        if (matches == null) return null;
        for (File f : matches) {
            if (f.isFile() && f.length() > 0) {
                return f;
            }
        }
        return null;
    }

    /**
     * 删除文件
     */
    public boolean delete(String relativePath) {
        File file = toAbsolute(relativePath);
        if (file.exists()) {
            boolean deleted = file.delete();
            if (deleted) {
                log.info("[文件存储] 删除: {}", relativePath);
            }
            return deleted;
        }
        return false;
    }

    /**
     * 检查文件是否存在
     */
    public boolean exists(String relativePath) {
        return toAbsolute(relativePath).exists();
    }

    /**
     * 获取文件绝对路径
     */
    public String getAbsolutePath(String relativePath) {
        return toAbsolute(relativePath).getAbsolutePath();
    }

    private File toAbsolute(String relativePath) {
        String clean = relativePath.startsWith("/") ? relativePath.substring(1) : relativePath;
        return new File(clean);
    }

    private void validateImage(MultipartFile file) throws IOException {
        if (file == null || file.isEmpty()) {
            throw new IOException("文件为空");
        }
        String name = file.getOriginalFilename();
        if (name == null || !isImageExtension(name)) {
            throw new IOException("不支持的图片格式: " + name);
        }
    }

    private boolean isImageExtension(String fileName) {
        String lower = fileName.toLowerCase();
        for (String ext : IMAGE_EXTENSIONS) {
            if (lower.endsWith(ext)) {
                return true;
            }
        }
        return false;
    }

    private String getExtension(String fileName) {
        if (fileName == null) return ".jpg";
        int dot = fileName.lastIndexOf('.');
        return dot > 0 ? fileName.substring(dot) : ".jpg";
    }

    private String sanitizeFileName(String name) {
        return name.replaceAll("[^a-zA-Z0-9_\\-]", "_");
    }

    private File resolveUniqueFile(File dir, String baseName) {
        int dot = baseName.lastIndexOf('.');
        String name = dot > 0 ? baseName.substring(0, dot) : baseName;
        String ext = dot > 0 ? baseName.substring(dot) : "";

        File f = new File(dir, baseName);
        int counter = 1;
        while (f.exists() && f.length() > 0) {
            f = new File(dir, name + "_" + counter + ext);
            counter++;
        }
        return f;
    }
}
