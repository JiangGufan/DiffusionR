library(testthat)
library(diffuR)

test_that("schedules produce sensible values", {
  sch <- beta_linear(10)
  expect_equal(length(sch$beta), 10)
  expect_true(all(sch$alpha_bar > 0 & sch$alpha_bar <= 1))
})
