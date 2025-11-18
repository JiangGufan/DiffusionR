#!/bin/bash
# 快速命令参考 - diffuR Package 测试

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PKG_DIR="/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR"

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  diffuR Package - Quick Command Guide${NC}"
echo -e "${BLUE}=========================================${NC}\n"

# 显示可用的命令
echo -e "${GREEN}可用命令:${NC}\n"
echo "1. compile     - 编译 Package"
echo "2. test        - 运行完整单元测试"
echo "3. dist        - 运行分布演示"
echo "4. image       - 运行图像演示"
echo "5. all         - 运行所有测试"
echo "6. clean       - 清理输出文件"
echo "7. docs        - 查看文档"
echo ""

# 如果提供了参数
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}用法: ./commands.sh [command]${NC}"
    echo ""
    echo "例子:"
    echo "  ./commands.sh compile"
    echo "  ./commands.sh test"
    echo "  ./commands.sh all"
    exit 0
fi

case "$1" in
    compile)
        echo -e "${BLUE}编译 Package...${NC}\n"
        cd "$PKG_DIR"
        R CMD INSTALL --no-multiarch --with-keep.source .
        ;;
    test)
        echo -e "${BLUE}运行完整单元测试...${NC}\n"
        cd "$PKG_DIR"
        Rscript inst/test_demo.R
        ;;
    dist)
        echo -e "${BLUE}运行分布演示...${NC}\n"
        cd "$PKG_DIR"
        Rscript inst/benchmarks/distribution_demo.R
        ;;
    image)
        echo -e "${BLUE}运行图像演示...${NC}\n"
        cd "$PKG_DIR"
        Rscript inst/benchmarks/image_demo.R
        ;;
    all)
        echo -e "${BLUE}运行所有测试...${NC}\n"
        cd "$PKG_DIR"
        echo -e "${GREEN}1/3 运行单元测试${NC}"
        Rscript inst/test_demo.R
        echo ""
        echo -e "${GREEN}2/3 运行分布演示${NC}"
        Rscript inst/benchmarks/distribution_demo.R
        echo ""
        echo -e "${GREEN}3/3 运行图像演示${NC}"
        Rscript inst/benchmarks/image_demo.R
        ;;
    clean)
        echo -e "${BLUE}清理输出文件...${NC}\n"
        cd "$PKG_DIR"
        rm -f test_plot.png dist_demo.png mnist_samples.png
        echo -e "${GREEN}清理完成${NC}"
        ;;
    docs)
        echo -e "${BLUE}可用的文档:${NC}\n"
        echo "1. QUICK_START.md  - 快速参考指南"
        echo "2. TEST_GUIDE.md   - 详细测试指南"
        echo "3. README_TESTS.md - 测试总结"
        echo ""
        echo -e "${BLUE}打开文档:${NC}"
        echo "  cat $PKG_DIR/QUICK_START.md"
        echo "  cat $PKG_DIR/TEST_GUIDE.md"
        echo "  cat $PKG_DIR/README_TESTS.md"
        ;;
    *)
        echo -e "${YELLOW}未知命令: $1${NC}"
        echo "可用命令: compile, test, dist, image, all, clean, docs"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}完成!${NC}\n"
