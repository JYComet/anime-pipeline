#!/bin/bash

# 每半小时将 _separated.wav 文件移动到目标 NAS 路径
# 用法: ./sync_separated_wav.sh &
# 日志: 写入同目录下的 sync_separated_wav.log

SOURCE_DIR="E:/ComicCut/anime-pipeline/data/temp/pipeline_video"
DEST_DIR="//RS3621/CompanyShare-Confidential/Persons/jiangyichen/Enhanced/529"
LOG_FILE="$(dirname "$0")/sync_separated_wav.log"
INTERVAL_SECONDS=$((30 * 60))  # 30分钟

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "======== 开始同步任务 ========"
log "源: $SOURCE_DIR"
log "目标: $DEST_DIR"
log "间隔: 每 30 分钟"

while true; do
    log "--- 扫描中 ---"

    if [ ! -d "$SOURCE_DIR" ]; then
        log "错误: 源目录不存在 - $SOURCE_DIR"
        sleep "$INTERVAL_SECONDS"
        continue
    fi

    if [ ! -d "$DEST_DIR" ]; then
        log "错误: 目标目录不存在 - $DEST_DIR"
        sleep "$INTERVAL_SECONDS"
        continue
    fi

    moved=0
    while IFS= read -r -d '' src_file; do
        filename=$(basename "$src_file")
        dest_file="$DEST_DIR/$filename"

        # 如果目标已存在且源没有更新，跳过
        if [ -f "$dest_file" ] && [ ! "$src_file" -nt "$dest_file" ]; then
            continue
        fi

        # 先复制到目标
        cp "$src_file" "$dest_file" 2>&1
        if [ $? -ne 0 ]; then
            log "复制失败: $filename"
            continue
        fi

        # 校验目标文件大小与源一致
        src_size=$(stat -c%s "$src_file")
        dest_size=$(stat -c%s "$dest_file")
        if [ "$src_size" -ne "$dest_size" ]; then
            log "校验失败(大小不一致): $filename (源=$src_size, 目标=$dest_size)"
            continue
        fi

        # 校验通过，删除源文件
        rm "$src_file" 2>&1
        if [ $? -eq 0 ]; then
            log "已移动: $filename"
            ((moved++))
        else
            log "删除源文件失败: $filename (目标已保留)"
        fi
    done < <(find "$SOURCE_DIR" -name "*_separated.wav" -type f -print0)

    if [ "$moved" -eq 0 ]; then
        log "无新文件需要移动"
    else
        log "本轮共移动 $moved 个文件"
    fi

    log "下次扫描: $(date -d "+$INTERVAL_SECONDS seconds" '+%Y-%m-%d %H:%M:%S')"
    sleep "$INTERVAL_SECONDS"
done
