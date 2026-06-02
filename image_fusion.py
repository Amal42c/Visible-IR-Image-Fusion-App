import cv2
import numpy as np
import pywt
from PIL import Image


#(FIRST MODEL) PATCH BASED FUSION +OVERLAPPING+(CORNER DETECTION+ LINEAR WEIGHTING)
def overlapping_patch_based_image_fusion(vis_img, IR_img):
    patch_size = 16
    shift = 8
    vis_img_height = vis_img.shape[0]
    vis_img_width = vis_img.shape[1]
    vis_img_dimensions = (vis_img_width, vis_img_height)
    IR_img = cv2.resize(IR_img, vis_img_dimensions)

    IR_img_grayscale = cv2.cvtColor(IR_img, cv2.COLOR_BGR2GRAY)#convert IR image to greyscale

    IR_img_color = cv2.applyColorMap(IR_img_grayscale, cv2.COLORMAP_JET)#apply a color map
    
    #Accumulates the sum of fused pixel values and tracks the count per pixel due to overlap
    fused_accumulator = np.zeros((vis_img_height, vis_img_width, 3), dtype=np.float32)
    weight_accumulator = np.zeros((vis_img_height, vis_img_width, 3), dtype=np.float32)

    mask = cv2.cornerHarris(np.float32(IR_img_grayscale), 2, 3, 0.04)#corner detection
    #saliency_map = cv2.dilate(saliency_map, None,  iterations = 3)
    mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)# detected range between 0 to 1

    for i in range(0, vis_img_height - patch_size, shift):
        for j in range(0, vis_img_width - patch_size, shift):
                patch_saliency = mask[i:i+patch_size, j:j+patch_size]
                e1 = np.mean(patch_saliency)
    
                #linear slope
                wt1 = e1 * 0.4  
    
                
                wt1 = np.clip(wt1, 0, 1) 
                wt2 = 1.0 - wt1

                #Patch Extraction
                patch_IR = IR_img_color[i:i+patch_size, j:j+patch_size].astype(np.float32)
                patch_vis = vis_img[i:i+patch_size, j:j+patch_size].astype(np.float32)

                #fusion here
                fused_accumulator[i:i+patch_size, j:j+patch_size] += (wt1 * patch_IR + wt2 * patch_vis)
                weight_accumulator[i:i+patch_size, j:j+patch_size] += 1.0

    weight_accumulator[weight_accumulator == 0] = 1.0
    fused_final = fused_accumulator / weight_accumulator
    fused_final = np.clip(fused_final, 0, 255).astype('uint8')

    return fused_final



#MULTILEVEL WAVELET on both vis and IR +max of intensity ir and vis taken+WEIGHTED+ HSI
def Multilevel_Wavelet_fusion(vis_img, IR_img, colour, level = 1):
    H, S, I_vis = get_Split(vis_img, colour)
    I_vis = I_vis.astype(np.float32)

    I_IR = cv2.cvtColor(IR_img, cv2.COLOR_BGR2GRAY)#converting Intensity of IR to grayscale
    I_IR = cv2.resize(I_IR, (I_vis.shape[1], I_vis.shape[0]))#resize to fit Intensity of Visible image

    #Wavelet decomposition on IR intensity
    wave_family = 'haar' #also known as Daubechies 1
    max_lvl = pywt.dwtn_max_level(I_IR.shape, wave_family)
    if level > max_lvl: 
        print("Error") #warning user of an invalid decomposition request
    level = min(level, max_lvl)

    #below periodization makes the image less boxy, when the dimension is not clear power of 2
    coeffs_vis = pywt.wavedec2(I_vis, wave_family, level=level, mode='periodization')
    coeffs_IR = pywt.wavedec2(I_IR, wave_family, level=level, mode='periodization')

    fused = [(0.35 * coeffs_vis[0]) + (0.65 * coeffs_IR[0])] #weighted average fusion

    for i in range(1, level + 1):
        (LH_vis, HL_vis, HH_vis) = coeffs_vis[i]
        (LH_IR, HL_IR, HH_IR) = coeffs_IR[i]

        #select max between visible and IR
        LH_fused = np.maximum(LH_vis, LH_IR)
        HL_fused = np.maximum(HL_vis, HL_IR)
        HH_fused = np.maximum(HH_vis, HH_IR)

        fused.append((LH_fused, HL_fused, HH_fused))

    I_fused = pywt.waverec2(fused, wave_family, mode='periodization')#reconstruction
    I_fused_resized = cv2.resize(I_fused, (H.shape[1], H.shape[0]))

    I_final = np.clip(I_fused_resized, 0, 255).astype(np.uint8)
    result_hsv = merge_channel(H, S, I_final, colour)
    return colour_Transform(result_hsv, colour, "BGR")


#MULTILEVEL WAVELET on both images +  SALIENCY USING(HARRIS+OPEN)AND OWN WEIGHT+IR 0.8 cap+ CLAHE
def wavelet_CLAHE_sal(vis_img, IR_img, colour, level = 1):
    H, S, I_vis = get_Split(vis_img, colour)

    I_IR = cv2.cvtColor(IR_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    I_IR = cv2.resize(I_IR, (I_vis.shape[1], I_vis.shape[0]))

    # Harris Corner + open mask
    harris = cv2.cornerHarris(I_vis.astype(np.uint8), 2, 3, 0.04)
    harris_open = cv2.morphologyEx(harris, cv2.MORPH_OPEN, None)
    mask = cv2.normalize(harris_open, None, 0, 1, cv2.NORM_MINMAX)

    # Wavelet Decomposition (db2)
    wave_family = 'db4' 
    max_lvl = pywt.dwtn_max_level(I_IR.shape, wave_family)
    if level > max_lvl:
        print("Error") #warning user of an invalid decomposition request
    level = min(level, max_lvl)
    #below periodization makes the image less boxy, when the dimension is not clear power of 2
    coeffs_vis = pywt.wavedec2(I_vis, wave_family, level=level, mode='periodization')
    coeffs_IR = pywt.wavedec2(I_IR, wave_family, level=level, mode='periodization')

    # Resize mask wrt LL visible
    resized_mask = cv2.resize(mask, (coeffs_vis[0].shape[1], coeffs_vis[0].shape[0]))

    #saliency weighted fusion
    fused = [(resized_mask * coeffs_vis[0]) + ((1 - resized_mask) * (coeffs_vis[0] * 0.6 + coeffs_IR[0] * 0.4))]

    for i in range(1, level + 1):
            (LH_vis, HL_vis, HH_vis) = coeffs_vis[i]
            (LH_IR, HL_IR, HH_IR) = coeffs_IR[i]
    
            #select absolute between visible and IR
            LH_fused = np.where(np.abs(LH_IR) > np.abs(LH_vis), LH_IR, LH_vis)
            HL_fused = np.where(np.abs(HL_IR) > np.abs(HL_vis), HL_IR, HL_vis)
            HH_fused = np.where(np.abs(HH_IR) > np.abs(HH_vis), HH_IR, HH_vis)

            fused.append((LH_fused, HL_fused, HH_fused))

    #Wavelet Reconstruction
    fused_i = pywt.waverec2(fused, wave_family, mode='periodization')
    fused_i = cv2.resize(fused_i, (H.shape[1], H.shape[0]))

    # Final Enhancement (CLAHE)
    fused_i_u8 = np.clip(fused_i, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(9, 9))
    final_i = clahe.apply(fused_i_u8)

    # Merge back to HSV and return as BGR
    result_hsv = merge_channel(H, S, final_i, colour)
    return colour_Transform(result_hsv, colour, "BGR")


#MULTILEVEL WAVELET ON BOTH IR, VIS Intensity + (Harris+OPEN) mask + PCA fusion +  CLAHE 
def Multilevel_Wavelet_PCA_CLAHE(vis_img, img_ir, colour,level = 1):
    H, S, I_vis = get_Split(vis_img, colour)

    I_IR = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY).astype(np.float32)
    I_IR = cv2.resize(I_IR, (I_vis.shape[1], I_vis.shape[0]))

    # Computes raw corner response scores, suppresses noice, normalizes them to form a mask in range 0 to 11
    harris_vis = cv2.cornerHarris(I_vis.astype(np.uint8), 2, 3, 0.04)
    open_vis = cv2.morphologyEx(harris_vis, cv2.MORPH_OPEN, None)
    mask = cv2.normalize(open_vis, None, 0, 1, cv2.NORM_MINMAX)

    # Wavelet Decomposition (db2)
    wave_family = 'db4' 
    max_lvl = pywt.dwtn_max_level(I_IR.shape, wave_family)
    if level > max_lvl:
        print("Error") #warning user of an invalid decomposition request
    level = min(level, max_lvl)
    #below periodization makes the image less boxy
    coeffs_vis = pywt.wavedec2(I_vis, wave_family, level=level, mode='periodization')
    coeffs_IR = pywt.wavedec2(I_IR, wave_family, level=level, mode='periodization')

    # Resize mask wrt LL 
    mask_resized = cv2.resize(mask, (coeffs_vis[0].shape[1], coeffs_vis[0].shape[0]))

    #PCA fusion using covariance, Eigen decomposition
    v1 = coeffs_vis[0].flatten()
    v2 = coeffs_IR[0].flatten()
    cov_matrix = np.cov(np.stack((v1, v2), axis=0))
    
    values, vectors = np.linalg.eigh(cov_matrix)
    best_vector = vectors[:, np.argmax(values)]
    
    # PCA weights
    wt_vis, wt_IR = best_vector / np.sum(best_vector)

    # LL PCA weights + Harris Mask is added for details
    LL_PCA = (wt_vis * coeffs_vis[0]) + (wt_IR * coeffs_IR[0])
    fused = [(mask_resized * coeffs_vis[0]) + ((1 - mask_resized) * LL_PCA)]

    # for better details select highest detailed among the visible and IR bands
    for i in range(1, level + 1):
            (LH_vis, HL_vis, HH_vis) = coeffs_vis[i]
            (LH_IR, HL_IR, HH_IR) = coeffs_IR[i]
    
            #select max between visible and IR
            LH_fused = np.where(np.abs(LH_IR) > np.abs(LH_vis), LH_IR, LH_vis)
            HL_fused = np.where(np.abs(HL_IR) > np.abs(HL_vis), HL_IR, HL_vis)
            HH_fused = np.where(np.abs(HH_IR) > np.abs(HH_vis), HH_IR, HH_vis)

            fused.append((LH_fused, HL_fused, HH_fused))

    # wavelet Reconstruction
    fused_i = pywt.waverec2(fused, wave_family, mode='periodization')
    fused_i = cv2.resize(fused_i, (H.shape[1], H.shape[0]))#resize to original size

    #applying CLAHE for contrast enhancement
    i_final = np.clip(fused_i, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    i_final = clahe.apply(i_final)

    #Merge the H, S, and new found I for result
    result_hsv = merge_channel(H, S, i_final, colour)
    return colour_Transform(result_hsv, colour, "BGR")


def resizing(image):
    if isinstance(image, str):
        img = cv2.imread(image)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
    else:
        img_pil = Image.open(image)
        img_rgb = np.array(img_pil)
        
    resized_img = cv2.resize(img_rgb, (350, 200))

    return resized_img

def colour_Transform(img, source_space="BGR", target_space="HSV"):
    if img is None:
        raise ValueError("The input image matrix is empty. Fix your pipeline load step.")

    src = source_space
    tgt = target_space
    if src == tgt or tgt == "HSI":
        tgt = "HSV"
    if src == "HSI":
        src = "HSV"

    conversions = {
        ("BGR", "HSV"): cv2.COLOR_BGR2HSV,
        ("HSV", "BGR"): cv2.COLOR_HSV2BGR,

        ("BGR", "CIE XYZ"): cv2.COLOR_BGR2XYZ,
        ("CIE XYZ", "BGR"): cv2.COLOR_XYZ2BGR,

        ("BGR", "LAB"): cv2.COLOR_BGR2Lab,
        ("LAB", "BGR"): cv2.COLOR_Lab2BGR,

        ("BGR", "YCrCb"): cv2.COLOR_BGR2YCrCb,
        ("YCrCb", "BGR"): cv2.COLOR_YCrCb2BGR,
        
        ("BGR", "YUV"): cv2.COLOR_BGR2YUV,
        ("YUV", "BGR"): cv2.COLOR_YUV2BGR,

        ("BGR", "CIELUV"): cv2.COLOR_BGR2LUV,
        ("CIELUV", "BGR"): cv2.COLOR_LUV2BGR,

        ("BGR", "HLS"): cv2.COLOR_BGR2HLS,
        ("HLS", "BGR"): cv2.COLOR_HLS2BGR,

        ("BGR", "GRAY"): cv2.COLOR_BGR2GRAY,
        ("GRAY", "BGR"): cv2.COLOR_GRAY2BGR
    }

    if (src, tgt) in conversions:
        return cv2.cvtColor(img, conversions[(src, tgt)])
    #return image if none of above
    return img.copy()


def get_Split(img, colour):
    img_transformed = colour_Transform(img, "BGR", colour)
    #3rd position is the intensity
    if colour == "YCrCb":
        #y is intensity
        y, cr, cb = img_transformed[:,:,0], img_transformed[:,:,1], img_transformed[:,:,2]
        return cr, cb, y
    
    elif colour == "LAB":
        #l is intensity
        l, a, b = img_transformed[:,:,0], img_transformed[:,:,1], img_transformed[:,:,2]
        return a, b, l 
    
    elif colour == "HSI":
        #here i is intensity
        h, s, i = img_transformed[:,:,0], img_transformed[:,:,1], img_transformed[:,:,2]
        return h, s, i
    
    elif colour == "CIE XYZ":
        #y is intensity 
        x, y, z = img_transformed[:,:,0], img_transformed[:,:,1], img_transformed[:,:,2]
        return x, z, y
    
    elif colour == "YUV":
        # y is intensity (luma)
        y, u, v = img_transformed[:,:,0], img_transformed[:,:,1], img_transformed[:,:,2]
        return u, v, y

    elif colour == "CIELUV":
        # l is intensity (perceptual lightness)
        l, u, v = img_transformed[:,:,0], img_transformed[:,:,1], img_transformed[:,:,2]
        return u, v, l

    elif colour == "HLS":
        # L is intensity (lightness)
        h, l, s = img_transformed[:,:,0], img_transformed[:,:,1], img_transformed[:,:,2]
        return h, s, l
    

def merge_channel(X, Y, Z, colour):
    if colour == "YCrCb":
        #cr, cb, y
        result_hsv = cv2.merge([Z, X.astype(np.uint8), Y.astype(np.uint8)])
        return result_hsv
    
    elif colour == "LAB":
        #a, b, l 
        result_hsv = cv2.merge([Z, X.astype(np.uint8), Y.astype(np.uint8)])
        return result_hsv 
    
    elif colour == "HSI":
        #h, s, i
        result_hsv = cv2.merge([X.astype(np.uint8), Y.astype(np.uint8), Z])
        return result_hsv 
    
    elif colour == "CIE XYZ":
        #x, z, y
        result_hsv = cv2.merge([X.astype(np.uint8), Z, Y.astype(np.uint8)])
        return result_hsv
    
    elif colour == "YUV":
        #u, v, y
        result_hsv = cv2.merge([Z, X.astype(np.uint8), Y.astype(np.uint8)])
        return result_hsv

    elif colour == "CIELUV":
        # u, v, l
        result_hsv = cv2.merge([Z, X.astype(np.uint8), Y.astype(np.uint8)])
        return result_hsv

    elif colour == "HLS":
        # h, s, l
        result_hsv = cv2.merge([X.astype(np.uint8), Z,  Y.astype(np.uint8)])
        return result_hsv