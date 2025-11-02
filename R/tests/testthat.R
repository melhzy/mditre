# MDITRE R Package Tests
# 
# This file runs all tests using testthat framework.
# Tests are organized by module/layer for clarity.

library(testthat)
library(torch)

# Source all R files
source("R/math_utils.R")
source("R/base_layer.R")
source("R/seeding.R")
source("R/layer1_phylogenetic_focus.R")
source("R/layer2_temporal_focus.R")
source("R/layer3_detector.R")
source("R/layer4_rule.R")
source("R/layer5_classification.R")
source("R/models.R")
source("R/phyloseq_loader.R")
source("R/trainer.R")
source("R/evaluation.R")
source("R/visualize.R")

# Run all tests
test_check("mditre")
