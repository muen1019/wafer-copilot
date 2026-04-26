#!/bin/bash
# Wafer Copilot 測試執行腳本
# 
# 使用方式:
#   cd wafer_copilot
#   bash tests/run_all_tests.sh

set -e

echo "=============================================="
echo "🧪 Wafer Copilot 測試套件"
echo "=============================================="
echo ""

# 切換到專案根目錄
cd "$(dirname "$0")/.."

# 建立輸出目錄
mkdir -p tests/outputs

echo "📋 測試項目:"
echo "   1. 視覺辨識分類測試"
echo "   2. Grad-CAM 熱力圖展示"
echo "   3. 知識庫檢索測試"
echo "   4. Agent 整合測試"
echo "   5. 數位孿生模組測試"
echo ""

# 詢問要執行哪些測試
read -p "執行全部測試? (y/n): " run_all

if [ "$run_all" = "y" ] || [ "$run_all" = "Y" ]; then
    echo ""
    echo "▶ 執行分類測試..."
    python -m tests.test_classifier
    
    echo ""
    echo "▶ 執行 Grad-CAM 測試..."
    python -m tests.test_gradcam
    
    echo ""
    echo "▶ 執行知識檢索測試..."
    python -m tests.test_knowledge_retrieval
    
    echo ""
    echo "▶ 執行 Agent 整合測試..."
    python -m tests.test_agent_integration
    
    echo ""
    echo "▶ 執行數位孿生模組測試..."
    python -m tests.test_digital_twin
else
    echo ""
    echo "選擇要執行的測試:"
    echo "   1) 視覺辨識分類"
    echo "   2) Grad-CAM 展示"
    echo "   3) 知識庫檢索"
    echo "   4) Agent 整合"
    echo "   5) 數位孿生模組"
    echo ""
    read -p "輸入編號 (可多選，如 1,3,5): " choices
    
    IFS=',' read -ra ADDR <<< "$choices"
    for i in "${ADDR[@]}"; do
        case $i in
            1)
                echo "▶ 執行分類測試..."
                python -m tests.test_classifier
                ;;
            2)
                echo "▶ 執行 Grad-CAM 測試..."
                python -m tests.test_gradcam
                ;;
            3)
                echo "▶ 執行知識檢索測試..."
                python -m tests.test_knowledge_retrieval
                ;;
            4)
                echo "▶ 執行 Agent 整合測試..."
                python -m tests.test_agent_integration
                ;;
            5)
                echo "▶ 執行數位孿生模組測試..."
                python -m tests.test_digital_twin
                ;;
        esac
    done
fi

echo ""
echo "=============================================="
echo "✅ 測試完成！"
echo "=============================================="
echo "輸出目錄: tests/outputs/"
ls -la tests/outputs/
