# 下载 CLIP ViT-B/32 ONNX 模型（视觉塔 + 文本塔）到 resources/models/
# 模型源：HuggingFace Xenova/clip-vit-base-patch32 (Apache-2.0)
#
# 关键点：完整多模态 model.onnx 在 ONNX Runtime sequential executor 下
# 跑子图时仍会触发未提供的输入（如 text_model 的 input_ids）报错，
# 所以这里下拆好塔的 vision_model.onnx 和 text_model.onnx。

$ErrorActionPreference = "Stop"
$targetDir = Join-Path $PSScriptRoot "..\src\main\resources\models\clip-vit-base-patch32"
$targetDir = [System.IO.Path]::GetFullPath($targetDir)

Write-Host "目标目录: $targetDir" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

$base = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx"

# vision_model_fp16.onnx（半精度，单视觉塔）+ text_model_fp16.onnx（半精度，单文本塔）+ tokenizer 与预处理配置
$files = @(
    @{ url = "$base/vision_model_fp16.onnx";  out = "vision_model.onnx" }
    @{ url = "$base/text_model_fp16.onnx";    out = "text_model.onnx" }
    @{ url = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json";               out = "tokenizer.json" }
    @{ url = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer_config.json";        out = "tokenizer_config.json" }
    @{ url = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/preprocessor_config.json";     out = "preprocessor_config.json" }
)

foreach ($f in $files) {
    $dest = Join-Path $targetDir $f.out
    if (Test-Path $dest) {
        Write-Host "  跳过 $($f.out) (已存在)" -ForegroundColor Yellow
        continue
    }
    Write-Host "  下载 $($f.out) ..." -ForegroundColor Green
    Invoke-WebRequest -Uri $f.url -OutFile $dest -UseBasicParsing
}

Write-Host ""
Write-Host "完成！" -ForegroundColor Green
Get-ChildItem $targetDir | Format-Table Name, @{N="Size(MB)";E={[math]::Round($_.Length/1MB,1)}} -AutoSize
