#!/usr/bin/env bash
set -euo pipefail

# Setup WorkOS Case harness for aic development
# Usage: bash scripts/setup_case.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AIC_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Installing Bun ==="
if ! command -v bun &>/dev/null; then
  curl -fsSL https://bun.sh/install | bash
  export PATH="$HOME/.bun/bin:$PATH"
fi
echo "bun $(bun --version)"

echo "=== Installing nvm + Node 20 ==="
if ! command -v nvm &>/dev/null; then
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
  export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
fi
nvm install 20 2>/dev/null || true
nvm use --delete-prefix 20 2>/dev/null || nvm use 20
echo "node $(node --version)"

echo "=== Cloning Case ==="
CASE_DIR="$(dirname "$AIC_DIR")/case"
if [ ! -d "$CASE_DIR" ]; then
  git clone git@github.com:vincewu51/case.git "$CASE_DIR"
  cd "$CASE_DIR"
  git remote add upstream https://github.com/workos/case.git
else
  cd "$CASE_DIR"
fi
bun install

echo "=== Registering aic in projects.json ==="
if ! grep -q '"aic"' "$CASE_DIR/projects.json"; then
  # Add aic entry before the closing ] in repos array
  python3 -c "
import json, sys
with open('$CASE_DIR/projects.json') as f:
    data = json.load(f)
data['repos'].append({
    'name': 'aic',
    'type': 'library',
    'path': '../aic',
    'remote': 'git@github.com:sl628/aic.git',
    'description': 'AI for Industry Challenge - robotics competition toolkit with RL training pipeline',
    'language': 'python',
    'packageManager': 'pip',
    'commands': {
        'setup': 'pixi install',
        'test': 'pixi run python -m pytest',
        'lint': 'pixi run black --check . && pixi run isort --check-only .',
        'typecheck': 'pixi run pyright',
        'format': 'pixi run black . && pixi run isort .'
    }
})
with open('$CASE_DIR/projects.json', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
"
  echo "aic added to projects.json"
else
  echo "aic already in projects.json"
fi

echo "=== Creating Case config ==="
mkdir -p ~/.config/case
cat > ~/.config/case/config.json << 'CONF'
{
  "models": {
    "default": { "provider": "anthropic", "model": "claude-sonnet-4-20250514" }
  }
}
CONF

echo "=== Setting up API key ==="
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "Set your API key:"
  echo '  read -s ANTHROPIC_API_KEY && echo "export ANTHROPIC_API_KEY=\"$ANTHROPIC_API_KEY\"" >> ~/.bashrc && source ~/.bashrc'
else
  echo "ANTHROPIC_API_KEY already set"
fi

echo ""
echo "=== Done ==="
echo "Run 'source ~/.bashrc' then use 'ca' from ~/case"
