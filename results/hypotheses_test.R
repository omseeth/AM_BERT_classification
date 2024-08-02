# Script for testing three hypotheses with respect to the Argument Mining tasks
#
# 28.07.2024
# OMS


################################################################################


# H1: Model size) The classification of nodes, that is propositions, can
# be improved with training and testing on the larger BERT model compared to 
# DistilBERT.

# (H1) for the CDCP corpus

distr_cdcp_F1_DistilBERT <- c(0.8065758889269157, 0.8234033989588201, 0.810304122883002)
distr_cdcp_F1_BERT <- c(0.8130671231603914, 0.8231033432437441, 0.8141430584660462)

# Perform the one-tailed t-test
h1_cdcp_t_test_result <- t.test(distr_cdcp_F1_BERT, distr_cdcp_F1_DistilBERT, alternative = "greater")
print("(H1) for the CDCP corpus")
print(h1_cdcp_t_test_result)

# (H1) for the Microtext corpus

distr_micro_F1_DistilBERT <- c(0.8303742670831279, 0.7668112029814157, 0.8149035488140257)
distr_micro_F1_BERT <- c(0.7590966555568326, 0.7590966555568326, 0.8810710577065718)

# Perform the one-tailed t-test
h1_micro_t_test_result <- t.test(distr_micro_F1_BERT, distr_micro_F1_DistilBERT, alternative = "greater")
print("(H1) for the Microtext corpus")
print(h1_micro_t_test_result)


################################################################################


# H2: Window clipping) The classification of directed and typed edges, that is, 
# argument relations that are not None, can be improved when training and 
# testing are done with a window for neighboring nodes with a window-size of 1.

# (H2) for support relations in the CDCP corpus

distr_cdcp_sup_F1_no_window <- c(0.1325478645066274, 0.12736660929432014, 0.110912343470483)
distr_cdcp_sup_F1_window_1 <- c(0.3614457831325301, 0.3444730077120822, 0.17272727272727276)

# Perform the one-tailed t-test
h2_cdcp_t_test_result <- t.test(distr_cdcp_sup_F1_window_1, distr_cdcp_sup_F1_no_window, alternative = "greater")
print("(H2) for the support relations in the CDCP corpus")
print(h2_cdcp_t_test_result)

# (H2) for support relations in the Microtext corpus

distr_micro_sup_F1_no_window <- c(0.37500000000000006, 0.4351145038167939, 0.31896551724137934)
distr_micro_sup_F1_window_1 <- c(0, 0, 0)

# Perform the one-tailed t-test
h2_micro_sup_t_test_result <- t.test(distr_micro_sup_F1_window_1, distr_micro_sup_F1_no_window, alternative = "greater")
print("(H2) for the support relations in the Microtext corpus")
print(h2_micro_sup_t_test_result)

# (H2) for example relations in the Microtext corpus

distr_micro_exa_F1_no_window <- c(0, 0 , 0)
distr_micro_exa_F1_window_1 <- c(0, 0, 0)

# Perform the one-tailed t-test
h2_micro_exa_t_test_result <- t.test(distr_micro_exa_F1_window_1, distr_micro_exa_F1_no_window, alternative = "greater")
print("(H2) for the example relations in the Microtext corpus")
print(h2_micro_exa_t_test_result)

# (H2) for rebuttal relations in the Microtext corpus

distr_micro_reb_F1_no_window <- c(0, 0, 0)
distr_micro_reb_F1_window_1 <- c(0, 0, 0)

# Perform the one-tailed t-test
h2_micro_reb_t_test_result <- t.test(distr_micro_reb_F1_window_1, distr_micro_reb_F1_no_window, alternative = "greater")
print("(H2) for the rebuttal relations in the Microtext corpus")
print(h2_micro_reb_t_test_result)


################################################################################


# H3 Data augmentation) The classification of directed and typed edges, that is,
# argument relations that are not None, can be improved when training and 
# testing are done with a window for neighboring nodes with a window-size of 1 
# AND when the data of the corpus is augmented.

# (H3) for support relations from the augmented CDCP corpus

distr_cdcp_sup_F1_window_1 <- c(0.3614457831325301, 0.3444730077120822, 0.17272727272727276)
distr_aug_sup_F1_window_1 <- c(0.2292490118577075, 0.30618892508143325, 0.3082437275985663)

# Perform the one-tailed t-test
h3_aug_t_test_result <- t.test(distr_aug_sup_F1_window_1, distr_cdcp_sup_F1_window_1, alternative = "greater")
print("(H3) for the augmented CDCP corpus")
print(h3_aug_t_test_result)

