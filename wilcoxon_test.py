#non-hyperparameter t-test
#conda install scipy
import numpy as np
from scipy.stats import wilcoxon
#generate images from each trained model ten times. set different seeds to get different images from each trained model.
#/proj/afraid/users/x_wayan/Results/stylegan3/training-runs/00041-stylegan3-r-AFF_train_color_intermediate-gpus8-batch32-gamma2
#/proj/afraid/users/x_wayan/Results/stylegan3/training-runs/00040-stylegan3-r-NFF_train_color_intermediate-gpus8-batch32-gamma2
gan_00041_AFF_FID = np.array([53.25,55.64,56.34,57.74,56.91,55.63,57.45,55.87,57.32,56.62])
gan_00040_NFF_FID = np.array([21.01,21.21,21.08,21.40,21.30,20.63,21.03,21.11,20.96,20.94])
#/proj/afraid/users/x_wayan/Data/generate_image/stylegan3/AFFNFF256_00022
gan_AFFNFF256_00022_AFF_FID = np.array([46.65,47.33,46.57,46.14,46.92,46.26,46.47,47.91,47.00,47.24])
gan_AFFNFF256_00022_NFF_FID = np.array([20.08,19.92,20.25,20.16,19.62,19.75,20.35,20.20,19.65,20.29])
#/proj/afraid/users/x_wayan/Results/diffusion/AFF/AFF64_run2
#/proj/afraid/users/x_wayan/Results/diffusion/NFF/NFF64_run2
diff_AFF64run2_FID = np.array([25.34,23.03,24.39,24.44,24.31,23.54,23.81,22.84,24.28,24.18])
diff_NFF64run2_FID = np.array([16.38,16.77,15.99,16.07,16.45,16.44,16.13,16.15,16.67,15.99])
#/proj/afraid/users/x_wayan/Results/diffusion/stack_AFF_NFF2
diff_stacked_AFF_FID=np.array([45.48,46.02,45.38,45.68,46.01,46.08,46.39,45.33,44.76,45.05])
diff_stacked_NFF_FID=np.array([26.50,26.78,26.69,25.92,27.12,26.43,26.88,26.50,26.89,27.32])

gan_gan_AFF=gan_00041_AFF_FID-gan_AFFNFF256_00022_AFF_FID
# Perform Wilcoxon signed-rank test
print(wilcoxon(gan_gan_AFF, alternative='two-sided'))
print(wilcoxon(gan_00041_AFF_FID,gan_AFFNFF256_00022_AFF_FID, alternative='two-sided'))

gan_gan_NFF=gan_00040_NFF_FID-gan_AFFNFF256_00022_NFF_FID
# Perform Wilcoxon signed-rank test
print(wilcoxon(gan_gan_NFF, alternative='two-sided'))
print(wilcoxon(gan_00040_NFF_FID,gan_AFFNFF256_00022_NFF_FID, alternative='two-sided'))

diff_diff_AFF=diff_AFF64run2_FID-diff_stacked_AFF_FID
# Perform Wilcoxon signed-rank test
print(wilcoxon(diff_diff_AFF, alternative='two-sided'))
print(wilcoxon(diff_AFF64run2_FID,diff_stacked_AFF_FID, alternative='two-sided'))

diff_diff_NFF=diff_NFF64run2_FID-diff_stacked_NFF_FID
# Perform Wilcoxon signed-rank test
print(wilcoxon(diff_diff_NFF, alternative='two-sided'))
print(wilcoxon(diff_NFF64run2_FID,diff_stacked_NFF_FID, alternative='two-sided'))

gan_diff_AFF=gan_AFFNFF256_00022_AFF_FID-diff_AFF64run2_FID
# Perform Wilcoxon signed-rank test
print(wilcoxon(gan_diff_AFF, alternative='two-sided'))
print(wilcoxon(gan_AFFNFF256_00022_AFF_FID,diff_AFF64run2_FID, alternative='two-sided'))

gan_diff_NFF=gan_AFFNFF256_00022_NFF_FID-diff_NFF64run2_FID
# Perform Wilcoxon signed-rank test
print(wilcoxon(gan_diff_NFF, alternative='two-sided'))
print(wilcoxon(gan_AFFNFF256_00022_NFF_FID,diff_NFF64run2_FID, alternative='two-sided'))
