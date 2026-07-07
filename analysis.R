d <- read.csv("/tmp/eval_comparison.csv")
nearest <- d[d$group == "nearest",]
batch5 <- d[d$group == "batch5_rl",]

cat(sprintf("Nearest: n=%d | Batch 5 RL: n=%d\n\n", nrow(nearest), nrow(batch5)))

kpis <- c("paper_cycle_time", "completed_cycles", "completion_rate",
          "orders_completed", "order_cycle_time", "total_energy",
          "energy_per_order", "congestion")

labels <- c("Paper Cycle Time (s)", "Completed Cycles", "Completion Rate",
            "Orders Completed", "Order Cycle Time (s)", "Total Energy",
            "Energy per Order", "Congestion (stop&go)")

# --- Normality check ---
cat("=== NORMALITY CHECK (Shapiro-Wilk) ===\n")
cat(sprintf("%-25s %12s %12s\n", "KPI", "Nearest p", "Batch5 p"))
cat(paste(rep("-", 52), collapse=""), "\n")
for (i in seq_along(kpis)) {
  k <- kpis[i]
  sw_nr <- shapiro.test(nearest[[k]])
  sw_b5 <- shapiro.test(batch5[[k]])
  nr_flag <- ifelse(sw_nr$p.value >= 0.05, "NORMAL", "NON-NORMAL")
  b5_flag <- ifelse(sw_b5$p.value >= 0.05, "NORMAL", "NON-NORMAL")
  cat(sprintf("%-25s %5.4f %-10s %5.4f %-10s\n",
              labels[i], sw_nr$p.value, nr_flag, sw_b5$p.value, b5_flag))
}

cat("\nMost KPIs fail normality -> applying Welch's ANOVA (robust to unequal variances and normality departures with equal/moderate sample sizes).\n\n")

# --- Variance Homogeneity check ---
cat("=== VARIANCE HOMOGENEITY CHECK ===\n")
cat("Note: Classical F-test assumes normality and is sensitive to departures from it.\n")
cat("Fligner-Killeen is a robust non-parametric test for homogeneity of variance.\n\n")
cat(sprintf("%-25s %12s %12s %12s %12s\n", "KPI", "F-test p", "F-test Sig", "Fligner p", "Fligner Sig"))
cat(paste(rep("-", 80), collapse=""), "\n")
for (i in seq_along(kpis)) {
  k <- kpis[i]
  vt <- var.test(d[[k]] ~ d$group, data = d)
  ft <- fligner.test(d[[k]] ~ d$group, data = d)
  
  vt_sig <- ifelse(vt$p.value < 0.05, "*", "")
  ft_sig <- ifelse(ft$p.value < 0.05, "*", "")
  
  cat(sprintf("%-25s %12.4e %-12s %12.4f %-12s\n",
              labels[i], vt$p.value, vt_sig, ft$p.value, ft_sig))
}
cat("\n")

# --- Main results: Welch's ANOVA + Bootstrap CI ---
set.seed(42)
n_boot <- 10000

cat("=== PARAMETRIC RESULTS (Welch's ANOVA) ===\n")
cat("Welch's One-Way ANOVA (not assuming equal variances) + Bootstrap 95% CI of mean difference\n\n")
cat(sprintf("%-25s %12s %12s %8s %12s %16s %10s %6s %24s\n",
            "KPI", "Nearest", "Batch5 RL", "Delta%", "Welch F", "df (num, den)", "p-val", "Sig", "Boot 95% CI (diff)"))
cat(paste(rep("=", 128), collapse=""), "\n")

for (i in seq_along(kpis)) {
  k <- kpis[i]
  nr_vals <- nearest[[k]]
  b5_vals <- batch5[[k]]

  nr_mean <- mean(nr_vals)
  b5_mean <- mean(b5_vals)
  delta_pct <- (b5_mean - nr_mean) / nr_mean * 100

  ow <- oneway.test(d[[k]] ~ d$group, data = d, var.equal = FALSE)
  pval <- ow$p.value
  fval <- ow$statistic
  df1 <- ow$parameter[1]
  df2 <- ow$parameter[2]

  sig <- ""
  if (pval < 0.001) { sig <- "***" }
  else if (pval < 0.01) { sig <- "**" }
  else if (pval < 0.05) { sig <- "*" }
  else if (pval < 0.1) { sig <- "." }

  boot_diffs <- replicate(n_boot, {
    nr_boot <- sample(nr_vals, replace = TRUE)
    b5_boot <- sample(b5_vals, replace = TRUE)
    mean(b5_boot) - mean(nr_boot)
  })
  ci_lo <- quantile(boot_diffs, 0.025)
  ci_hi <- quantile(boot_diffs, 0.975)

  df_str <- sprintf("%d, %.1f", df1, df2)
  cat(sprintf("%-25s %12.2f %12.2f %+7.2f%% %12.4f %-16s %10.4f %-6s [%+10.2f, %+10.2f]\n",
              labels[i], nr_mean, b5_mean, delta_pct, fval, df_str, pval, sig, ci_lo, ci_hi))
}

cat("\nSig codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1\n")
cat("Bootstrap: 10,000 resamples, percentile method\n")

# --- Effect size: Cohen's d ---
cat("\n\n=== EFFECT SIZES (Cohen's d) ===\n")
cat("Interpretation: |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, else large\n\n")

cohens_d <- function(x, y) {
  n1 <- length(x)
  n2 <- length(y)
  v1 <- var(x)
  v2 <- var(y)
  pooled_sd <- sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
  d <- (mean(x) - mean(y)) / pooled_sd
  return(d)
}

for (i in seq_along(kpis)) {
  k <- kpis[i]
  nr_vals <- nearest[[k]]
  b5_vals <- batch5[[k]]

  # We compute Cohen's d (Batch5 vs Nearest)
  d_val <- cohens_d(b5_vals, nr_vals)

  magnitude <- "negligible"
  if (abs(d_val) >= 0.8) { magnitude <- "LARGE" }
  else if (abs(d_val) >= 0.5) { magnitude <- "MEDIUM" }
  else if (abs(d_val) >= 0.2) { magnitude <- "SMALL" }

  cat(sprintf("  %-25s Cohen's d = %+.4f  (%s)\n", labels[i], d_val, magnitude))
}

# --- Detailed per-KPI with means and Welch's confidence intervals ---
cat("\n\n=== DETAILED PER-KPI (mean, SD, Welch's ANOVA) ===\n")
for (i in seq_along(kpis)) {
  k <- kpis[i]
  nr_vals <- nearest[[k]]
  b5_vals <- batch5[[k]]

  cat(sprintf("\n--- %s ---\n", labels[i]))
  cat(sprintf("  Nearest:   mean=%.4f, SD=%.4f\n", mean(nr_vals), sd(nr_vals)))
  cat(sprintf("  Batch5 RL: mean=%.4f, SD=%.4f\n", mean(b5_vals), sd(b5_vals)))

  ow <- oneway.test(d[[k]] ~ d$group, data = d, var.equal = FALSE)
  tt <- t.test(b5_vals, nr_vals, var.equal = FALSE)
  
  cat(sprintf("  Welch ANOVA: F(%.0f, %.2f) = %.4f, p = %.6f\n",
              ow$parameter[1], ow$parameter[2], ow$statistic, ow$p.value))
  cat(sprintf("  Welch t-test: t(%.2f) = %.4f, p = %.6f\n",
              tt$parameter, tt$statistic, tt$p.value))
  cat(sprintf("  Mean difference (Batch5 - Nearest): %.4f\n", tt$estimate[1] - tt$estimate[2]))
  cat(sprintf("  95%% CI of difference (Welch): [%.4f, %.4f]\n", tt$conf.int[1], tt$conf.int[2]))
}
