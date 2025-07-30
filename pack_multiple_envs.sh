#!/bin/bash
# 打包多个Conda环境的脚本

BACKUP_DIR="${1:-/linzhihang/huaiwenpang/legal_LLM/env_backup}"
#默认地址
shift
ENVIRONMENTS=("$@")

# 如果没有指定环境，使用所有非base环境
if [ ${#ENVIRONMENTS[@]} -eq 0 ]; then
    echo "未指定环境，将打包所有非base环境..."
    # 获取所有环境（排除base和行首有#的行）
    mapfile -t ENVIRONMENTS < <(conda env list | awk 'NR>3 && !/#/ {print $1}')
fi

mkdir -p "$BACKUP_DIR"

for env in "${ENVIRONMENTS[@]}"; do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 开始打包环境: $env"
    
    if conda info --envs | grep -q " $env "; then
        # 打包环境，启用压缩（-z）并显示进度（-v）
        conda pack -n "$env" -o "$BACKUP_DIR/${env}.tar.gz" -z -v
        
        # 检查打包结果
        if [ $? -eq 0 ]; then
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✓ 打包成功: $BACKUP_DIR/${env}.tar.gz ($(du -sh "$BACKUP_DIR/${env}.tar.gz" | cut -f1))"
        else
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✗ 打包失败: $env"
        fi
    else
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✗ 环境不存在: $env"
    fi
done

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 所有环境打包完成！备份目录: $BACKUP_DIR"
echo "总计打包: ${#ENVIRONMENTS[@]} 个环境"