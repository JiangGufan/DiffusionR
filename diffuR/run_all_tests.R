#!/usr/bin/env Rscript
# Quick Start Runner - Run all tests sequentially
# Usage: Rscript run_all_tests.R

# ============================================================================
# CONFIGURATION
# ============================================================================

PKG_ROOT <- dirname(dirname(getwd()))
TEST_SCRIPTS <- list(
  test_demo = file.path(PKG_ROOT, "inst", "test_demo.R"),
  dist_demo = file.path(PKG_ROOT, "inst", "benchmarks", "distribution_demo.R"),
  img_demo = file.path(PKG_ROOT, "inst", "benchmarks", "image_demo.R")
)

# ============================================================================
# MAIN
# ============================================================================

cat("\n")
cat(paste0("╔", paste(rep("═", 78), collapse = ""), "╗\n"))
cat(paste0("║ diffuR Package - Complete Test Suite", paste(rep(" ", 42), collapse = ""), "║\n"))
cat(paste0("╚", paste(rep("═", 78), collapse = ""), "╝\n\n"))

# Choose which tests to run
cat("Available tests:\n")
cat("  [1] Test Demo (完整单元测试) - RECOMMENDED for first-time\n")
cat("  [2] Distribution Demo (分布演示)\n")
cat("  [3] Image Demo (图像演示)\n")
cat("  [4] Run all tests (运行所有)\n")
cat("  [0] Exit\n\n")

response <- readline(prompt = "Choose option (0-4): ")

run_test <- function(script_path, name) {
  if (!file.exists(script_path)) {
    cat("\n⚠️  Script not found:", script_path, "\n")
    return(FALSE)
  }
  
  cat("\n")
  cat(paste0(paste(rep("─", 80), collapse = ""), "\n"))
  cat("Running:", name, "\n")
  cat(paste0(paste(rep("─", 80), collapse = ""), "\n"))
  
  source(script_path, echo = FALSE)
  return(TRUE)
}

switch(response,
  "1" = {
    cat("Running Test Demo...\n")
    run_test(TEST_SCRIPTS$test_demo, "Test Demo")
  },
  "2" = {
    cat("Running Distribution Demo...\n")
    run_test(TEST_SCRIPTS$dist_demo, "Distribution Demo")
  },
  "3" = {
    cat("Running Image Demo...\n")
    run_test(TEST_SCRIPTS$img_demo, "Image Demo")
  },
  "4" = {
    cat("Running all tests...\n")
    run_test(TEST_SCRIPTS$test_demo, "Test Demo")
    run_test(TEST_SCRIPTS$dist_demo, "Distribution Demo")
    run_test(TEST_SCRIPTS$img_demo, "Image Demo")
  },
  "0" = {
    cat("Exiting...\n")
  },
  {
    cat("Invalid option. Exiting...\n")
  }
)

cat("\n")
cat(paste0(paste(rep("═", 80), collapse = ""), "\n\n"))
