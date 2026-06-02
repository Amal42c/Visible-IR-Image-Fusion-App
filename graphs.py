import numpy as np
import cv2
def entropy(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sample = image.flatten()
    hist, bin = np.histogram(sample, bins=256, range = (0,257))
    probability = hist / np.sum(hist)# his into probability
    probability = probability[probability > 0] #log=0 is crash
    joint_entropy = -np.sum(probability * np.log2(probability))  
    return joint_entropy

def spatial_Freq(result):
    if len(result.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result = result.astype(np.float32)
    res_rf = np.sqrt(np.mean(np.diff(result, axis=0)**2))
    res_cf = np.sqrt(np.mean(np.diff(result, axis=1)**2))
    res_sf = np.sqrt(res_rf**2 + res_cf**2)
    return res_sf

def Standard_Deviation(result):
    if len(result.shape) == 3: 
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    return np.std(result)

def getSSISM(I1, I2):
    '''
    :param i1: represents original image matrix
    :param i2: represents degraded image matrix
    :return: SSIM score
    '''
    
    if len(I1.shape) == 3: I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    if len(I2.shape) == 3: I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # Constants for luminance and contrast
    C1 = 6.5025
    C2 = 58.5225
    # C3=C2/2

    # converting to float for squaring
    I1 = np.float32(I1)
    I2 = np.float32(I2)
    I2_2 = I2 * I2
    I1_2 = I1 * I1
    I1_I2 = I1 * I2

    # applying GaussianBlur with (11,11) kernel where mean=st_dev=1.5
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2

    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2  # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2  # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    ssim_map = cv2.divide(t3, t1)
    ssim = cv2.mean(ssim_map)[0]
    return ssim

import cv2
import numpy as np

def FQIFSM(vis, ir, result):
    # 1. Grab target dimensions cleanly at the very top of the scope
    target_h, target_w = result.shape[0], result.shape[1]
    
    def process_input_matrix(img):
        # Handle color conversion safely
        if len(img.shape) == 3: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # FIX: Explicitly resize using local variables, completely ignoring global scope
        img_resized = cv2.resize(img, (target_w, target_h))
        
        if img_resized.dtype == np.uint8:
            return img_resized.astype(np.float32) / 255.0
        return img_resized.astype(np.float32)

    # 2. Process inputs into matching normalized arrays
    I1 = process_input_matrix(vis)
    I2 = process_input_matrix(ir)
    If = process_input_matrix(result)
    
    win_size = 8
    C1 = 1e-4
    C2 = 1e-4

    def get_local_stats(X, Y):
        muX = cv2.blur(X, (win_size, win_size))
        muY = cv2.blur(Y, (win_size, win_size))
        
        varX = np.maximum(cv2.blur(X*X, (win_size, win_size)) - muX**2, 0.0)
        varY = np.maximum(cv2.blur(Y*Y, (win_size, win_size)) - muY**2, 0.0)
        covXY = cv2.blur(X*Y, (win_size, win_size)) - muX*muY
        
        num = 4 * covXY * muX * muY + C1
        den = (varX + varY) * (muX**2 + muY**2) + C2
        return num / den, varX, covXY

    # 3. Structural mapping calculations
    qi1_map, s1_sq, s1f = get_local_stats(I1, If)
    qi2_map, s2_sq, s2f = get_local_stats(I2, If)

    # 4. Piella local and global saliency weighting
    denom_variance = s1_sq + s2_sq + 1e-8
    lamb_map = s1_sq / denom_variance

    cw_map = np.maximum(s1_sq, s2_sq)
    total_cw = np.sum(cw_map)

    if total_cw < 1e-7:
        fqi = float(np.mean(lamb_map * qi1_map + (1.0 - lamb_map) * qi2_map))
    else:
        fqi_map = (cw_map / total_cw) * (lamb_map * qi1_map + (1.0 - lamb_map) * qi2_map)
        fqi = float(np.sum(fqi_map))

    # 5. Fusion Similarity Metric
    s1f_abs = np.abs(s1f)
    s2f_abs = np.abs(s2f)
    sim_map = s1f_abs / (s1f_abs + s2f_abs + 1e-8)
    
    fsm_map = sim_map * (qi1_map - qi2_map) + qi2_map
    fsm = float(np.mean(fsm_map))

    return fqi, fsm



def graph_entropy(img_vis, img_IR, img_fused):
    histv, binv = np.histogram(img_vis.flatten(), bins=256, range=(0, 255))
    histi, bini = np.histogram(img_IR.flatten(), bins=256, range=(0, 255))
    histr, binr = np.histogram(img_fused.flatten(), bins=256, range=(0, 255))

    probsr = histr / histr.sum()
    probsr = probsr[probsr > 0] # Remove zeros to avoid log error
    entropyf = -np.sum(probsr * np.log2(probsr))

    probsi = histi / histi.sum()
    probsi = probsi[probsi > 0] # Remove zeros to avoid log error
    entropy_IR = -np.sum(probsi * np.log2(probsi))

    probsv = histv / histv.sum()
    probsv = probsv[probsv > 0] # Remove zeros to avoid log error
    entropy_vis = -np.sum(probsv * np.log2(probsv))
    
    return histv, histi, histr, (entropy_vis, entropy_IR, entropyf)

def mutual_information(img_vis, img_fused, bins=256):
    x = img_vis.ravel()
    y = img_fused.ravel()
    h_xy, _, _ = np.histogram2d(x, y, bins=bins)
    
    n_total = np.sum(h_xy)
    p_xy = h_xy / n_total

    p_x = np.sum(p_xy, axis=1, keepdims=True)
    p_y = np.sum(p_xy, axis=0, keepdims=True)
    
    nonzero_mask = p_xy > 0 #log(0) errors
    
    mi = np.sum(
        p_xy[nonzero_mask] * 
        np.log2(p_xy[nonzero_mask] / (p_x * p_y)[nonzero_mask])
    )
    return mi



