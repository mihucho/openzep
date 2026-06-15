#!/usr/bin/env bash
# setup_docker.sh — 交互式 Docker 安装脚本

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR ]${NC} $*"; exit 1; }

read_env_value() {
    local key="$1"
    local file="${2:-.env}"
    if [[ -f "$file" ]]; then
        grep -E "^${key}=" "$file" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '\r'
    fi
}

prompt_value() {
    local var_name="$1"
    local prompt_text="$2"
    local default_value="${3:-}"
    local secret="${4:-0}"
    local required="${5:-1}"
    local current_value="${!var_name:-}"

    if [ -n "$current_value" ]; then
        printf -v "$var_name" '%s' "$current_value"
        return 0
    fi

    if [ ! -t 0 ]; then
        if [ "$required" = "1" ] && [ -z "$default_value" ]; then
            error "${var_name} 未提供，且当前为非交互模式"
        fi
        printf -v "$var_name" '%s' "$default_value"
        return 0
    fi

    if [ -n "$default_value" ]; then
        printf "%b" "${BOLD}${prompt_text} [默认: ${default_value}]: ${NC}"
    else
        printf "%b" "${BOLD}${prompt_text}: ${NC}"
    fi

    if [ "$secret" = "1" ]; then
        read -rs current_value
        echo
    else
        read -r current_value
    fi

    if [ -z "$current_value" ]; then
        current_value="$default_value"
    fi

    if [ "$required" = "1" ] && [ -z "$current_value" ]; then
        error "${var_name} 不能为空"
    fi

    printf -v "$var_name" '%s' "$current_value"
}

normalize_docker_url() {
    local raw_url="$1"
    local label="$2"
    if [[ "$raw_url" =~ ^http://localhost([/:]|$) ]]; then
        local fixed="${raw_url/http:\/\/localhost/http:\/\/host.docker.internal}"
        warn "${label} 使用了 localhost，Docker 内将自动改为: ${fixed}" >&2
        printf '%s\n' "$fixed"
        return 0
    fi
    if [[ "$raw_url" =~ ^http://127\.0\.0\.1([/:]|$) ]]; then
        local fixed="${raw_url/http:\/\/127.0.0.1/http:\/\/host.docker.internal}"
        warn "${label} 使用了 127.0.0.1，Docker 内将自动改为: ${fixed}" >&2
        printf '%s\n' "$fixed"
        return 0
    fi
    printf '%s\n' "$raw_url"
}

echo
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║     OpenZep Docker 安装向导          ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════╝${NC}"
echo

command -v docker >/dev/null 2>&1 || error "未找到 docker，请先安装 Docker"
docker compose version >/dev/null 2>&1 || error "未找到 docker compose，请先安装 Docker Compose"
success "Docker 环境检查通过"
echo

ENV_FILE=".env"
if [[ -f "$ENV_FILE" ]]; then
    cp "$ENV_FILE" "${ENV_FILE}.bak"
    warn ".env 已存在，已备份到 .env.bak"
fi

prompt_value "LLM_BASE_URL" "LLM Base URL（如 https://api.openai.com/v1）" "$(read_env_value LLM_BASE_URL "$ENV_FILE")"
prompt_value "LLM_API_KEY" "LLM API Key" "$(read_env_value LLM_API_KEY "$ENV_FILE")" 1
prompt_value "LLM_MODEL" "LLM 模型名称" "$(read_env_value LLM_MODEL "$ENV_FILE")"
prompt_value "LLM_SMALL_MODEL" "LLM 小模型名称" "$(read_env_value LLM_SMALL_MODEL "$ENV_FILE" || true)" 0 0

echo
prompt_value "SEPARATE_EMBEDDER" "是否单独配置 Embedder？[y/N]" "N" 0 0
if [[ "$SEPARATE_EMBEDDER" =~ ^[Yy]$ ]]; then
    prompt_value "EMBEDDER_BASE_URL" "Embedder Base URL" "$(read_env_value EMBEDDER_BASE_URL "$ENV_FILE")"
    prompt_value "EMBEDDER_API_KEY" "Embedder API Key" "$(read_env_value EMBEDDER_API_KEY "$ENV_FILE")" 1
    prompt_value "EMBEDDER_MODEL" "Embedder 模型名称" "$(read_env_value EMBEDDER_MODEL "$ENV_FILE")" 0 0
else
    EMBEDDER_BASE_URL=""
    EMBEDDER_API_KEY=""
    prompt_value "EMBEDDER_MODEL" "Embedder 模型名称" "$(read_env_value EMBEDDER_MODEL "$ENV_FILE")" 0 0
    EMBEDDER_MODEL=${EMBEDDER_MODEL:-text-embedding-3-small}
fi

echo
prompt_value "API_KEY" "OpenZep API Key（留空自动生成）" "$(read_env_value API_KEY "$ENV_FILE")" 1 0
if [[ -z "$API_KEY" ]]; then
    API_KEY="$(od -An -N 6 -tx1 /dev/urandom)"
    API_KEY="openzep-${API_KEY//[[:space:]]/}"
    info "已生成随机 API Key: ${BOLD}${API_KEY}${NC}"
fi
prompt_value "NEO4J_PASSWORD" "Neo4j 密码" "$(read_env_value NEO4J_PASSWORD "$ENV_FILE")" 1 0
NEO4J_PASSWORD=${NEO4J_PASSWORD:-password123}

LLM_BASE_URL="$(normalize_docker_url "$LLM_BASE_URL" "LLM_BASE_URL")"
if [[ -n "$EMBEDDER_BASE_URL" ]]; then
    EMBEDDER_BASE_URL="$(normalize_docker_url "$EMBEDDER_BASE_URL" "EMBEDDER_BASE_URL")"
fi

cat > "$ENV_FILE" <<EOF
# LLM
LLM_API_KEY=${LLM_API_KEY}
LLM_BASE_URL=${LLM_BASE_URL}
LLM_MODEL=${LLM_MODEL}
LLM_SMALL_MODEL=${LLM_SMALL_MODEL:-$LLM_MODEL}

# Embedder
EMBEDDER_API_KEY=${EMBEDDER_API_KEY}
EMBEDDER_BASE_URL=${EMBEDDER_BASE_URL}
EMBEDDER_MODEL=${EMBEDDER_MODEL}

GRAPH_DB=neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=${NEO4J_PASSWORD}

SQLITE_PATH=openzep.db

API_KEY=${API_KEY}
EOF
success ".env 已写入 Docker 版本配置"
echo

info "启动 Docker Compose..."
docker compose up -d --build

info "等待 OpenZep 健康检查..."
for i in $(seq 1 20); do
    if curl -sf http://localhost:8000/healthz >/dev/null 2>&1; then
        success "OpenZep Docker 服务已启动"
        break
    fi
    sleep 2
    if [[ "$i" -eq 20 ]]; then
        warn "健康检查超时，请查看日志: docker compose logs --tail=100 openzep"
    fi
done

echo
echo -e "  服务地址:  ${BOLD}http://localhost:8000${NC}"
echo -e "  API Key:   ${BOLD}${API_KEY}${NC}"
echo -e "  文档地址:  ${BOLD}http://localhost:8000/docs${NC}"
echo -e "  排障日志:  ${BOLD}docker compose logs -f openzep${NC}"
echo
