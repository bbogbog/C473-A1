import numpy as np
import math

# ---------------------------------------------------------
# 1. DATA ENTRY
# ---------------------------------------------------------
# Data for Class w1 (rows = samples, cols = x1, x2, x3)
w1_data = np.array([
    [-5.01, -8.12, -3.68],
    [-5.43, -3.48, -3.54],
    [ 1.08, -5.52,  1.66],
    [ 0.86, -3.78, -4.11],
    [-2.67,  0.63,  7.39],
    [ 4.94,  3.29,  2.08],
    [-2.51,  2.09, -2.59],
    [-2.25, -2.13, -6.94],
    [ 5.56,  2.86, -2.26],
    [ 1.03, -3.33,  4.33]
])

# Data for Class w2
w2_data = np.array([
    [-0.91, -0.18, -0.05],
    [ 1.30, -2.06, -3.53],
    [-7.75, -4.54, -0.95],
    [-5.47,  0.50,  3.92],
    [ 6.14,  5.72, -4.85],
    [ 3.60,  1.26,  4.36],
    [ 5.37, -4.63, -3.65],
    [ 7.18,  1.46, -6.66],
    [-7.39,  1.17,  6.30],
    [-7.50, -6.32, -0.31]
])

# (Class w3 is provided in the table but P(w3)=0, 
# so we ignore it for classification design as per instructions).

# Prior Probabilities
P_w1 = 0.5
P_w2 = 0.5

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def estimate_params(data):
    """
    Estimates Mean (mu) and Covariance (sigma) for a dataset.
    Handles 1D and nD cases.
    """
    # Calculate mean vector
    mu = np.mean(data, axis=0)
    
    # Calculate covariance matrix
    # rowvar=False because rows are samples in our data
    sigma = np.cov(data, rowvar=False)
    
    # Handle the 1D case where numpy returns a 0-d array (scalar)
    if data.shape[1] == 1:
        sigma = np.array([[sigma]])
        
    return mu, sigma

def discriminant_function(x, mu, sigma, prior):
    """
    Calculates g_i(x) based on Eq. 49
    g_i(x) = -1/2(x-u).T * inv(Sigma) * (x-u) - d/2 ln(2pi) - 1/2 ln|Sigma| + ln(P(w))
    """
    d = len(x)
    sigma_inv = np.linalg.inv(sigma)
    sigma_det = np.linalg.det(sigma)
    
    # Term 1: Mahalanobis distance part
    diff = x - mu
    # Reshape diff to be a column vector for matrix math consistency
    # specific trick to handle 1D vs nD dot products cleanly in numpy
    term1 = -0.5 * np.dot(np.dot(diff.T, sigma_inv), diff)
    
    # Term 2: - d/2 ln(2pi)
    term2 = -(d / 2) * np.log(2 * np.pi)
    
    # Term 3: - 1/2 ln |Sigma|
    term3 = -0.5 * np.log(sigma_det)
    
    # Term 4: + ln P(w)
    term4 = np.log(prior)
    
    return term1 + term2 + term3 + term4

def calc_bhattacharyya_bound(mu1, cov1, mu2, cov2, p1, p2):
    """
    Calculates the Bhattacharyya error bound.
    P(error) <= sqrt(P(w1)P(w2)) * e^(-k(0.5))
    """
    # Average Covariance
    cov_avg = (cov1 + cov2) / 2
    cov_avg_inv = np.linalg.inv(cov_avg)
    
    # Difference in means
    mu_diff = mu2 - mu1
    
    # Bhattacharyya distance k(1/2)
    # Part 1: 1/8 (u2-u1)^T * inv(avg_cov) * (u2-u1)
    term1 = (1/8) * np.dot(np.dot(mu_diff.T, cov_avg_inv), mu_diff)
    
    # Part 2: 1/2 ln( det(avg_cov) / sqrt(det(cov1)*det(cov2)) )
    det_avg = np.linalg.det(cov_avg)
    det_1 = np.linalg.det(cov1)
    det_2 = np.linalg.det(cov2)
    
    term2 = 0.5 * np.log(det_avg / math.sqrt(det_1 * det_2))
    
    k_half = term1 + term2
    
    # The Bound
    bound = math.sqrt(p1 * p2) * math.exp(-k_half)
    return bound

def perform_experiment(features_indices, description):
    print(f"--- {description} ---")
    
    # 1. Select Features
    d1 = w1_data[:, features_indices]
    d2 = w2_data[:, features_indices]
    
    # 2. Estimate Parameters
    mu1, cov1 = estimate_params(d1)
    mu2, cov2 = estimate_params(d2)
    
    print(f"Parameters Estimated.")
    
    # 3. Calculate Empirical Training Error
    # We test the classifier on the exact same data we trained on
    errors = 0
    total_samples = len(d1) + len(d2)
    
    # Test on class 1 data (should be class 1)
    for x in d1:
        g1 = discriminant_function(x, mu1, cov1, P_w1)
        g2 = discriminant_function(x, mu2, cov2, P_w2)
        if g2 > g1: # Misclassified as class 2
            errors += 1
            
    # Test on class 2 data (should be class 2)
    for x in d2:
        g1 = discriminant_function(x, mu1, cov1, P_w1)
        g2 = discriminant_function(x, mu2, cov2, P_w2)
        if g1 >= g2: # Misclassified as class 1
            errors += 1
            
    empirical_error = errors / total_samples
    print(f"Empirical Training Error: {empirical_error * 100:.2f}% ({errors}/{total_samples})")
    
    # 4. Calculate Bhattacharyya Bound
    b_bound = calc_bhattacharyya_bound(mu1, cov1, mu2, cov2, P_w1, P_w2)
    print(f"Bhattacharyya Error Bound:  {b_bound:.4f}")
    print("")

# ---------------------------------------------------------
# 3. EXECUTION
# ---------------------------------------------------------

# (a), (b), (c): Use only feature x1 (index 0)
perform_experiment([0], "1D Case (Feature x1)")

# (d): Use features x1, x2 (indices 0, 1)
perform_experiment([0, 1], "2D Case (Features x1, x2)")

# (e): Use features x1, x2, x3 (indices 0, 1, 2)
perform_experiment([0, 1, 2], "3D Case (Features x1, x2, x3)")