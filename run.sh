#!/bin/bash
# ─────────────────────────────────────────────
#   MediFusion AI — Quick Start Script
# ─────────────────────────────────────────────

PROJECT_DIR="/Users/harshit/Desktop/MEDIFUSION_AI/MediFusionAi"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║        MediFusion AI  🧬  v1.0           ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Kill any existing server on port 8000
if lsof -ti :8000 &>/dev/null; then
    echo "⚠️  Port 8000 in use — stopping old server..."
    lsof -ti :8000 | xargs kill -9 2>/dev/null
    sleep 1
fi

# Set required environment flags
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PYTHONPATH="$PROJECT_DIR"

echo "📂  Project: $PROJECT_DIR"
echo "🚀  Starting server at http://localhost:8000"
echo "📖  API Docs: http://localhost:8000/docs"
echo "🛑  Press Ctrl+C to stop"
echo ""

# Auto-open browser when ready
(sleep 2 && open "http://localhost:8000") &

cd "$PROJECT_DIR"
/opt/anaconda3/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
