 import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.signal import find_peaks

# --- Image Preprocessing Section ---
# Load and process original image
image = cv2.imread('/content/image_bend.jpg')

# Convert to HSV and create mask
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_bound = np.array([70, 0, 112])
upper_bound = np.array([115, 112, 255])
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Apply mask and convert to grayscale
result = cv2.bitwise_and(image, image, mask=mask)
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Threshold and skeletonize
gray = cv2.erode(gray, (5,5), iterations=5)
gray = cv2.dilate(gray, (3,3), iterations=2)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# --- Line Analysis Section ---
# Find all white pixels in skeletonized image
white_pixels = np.column_stack(np.where(thresh > 150))  # (y, x) coordinates

# K-means clustering for line separation
kmeans = KMeans(n_clusters=2, random_state=42).fit(white_pixels[:, 1].reshape(-1, 1))
labels = kmeans.labels_

# Create separate arrays for each line
line1_points = white_pixels[labels == 0]
line2_points = white_pixels[labels == 1]

def fourier_fit(x, y, num_terms=8):
    """Fit a Fourier series to x,y data points"""
    # Normalize to [0, 2π] range for Fourier
    t = np.linspace(0, 2*np.pi, len(x))
    
    # Define Fourier basis functions
    A = np.ones((len(t), 2*num_terms+1))
    for i in range(1, num_terms+1):
        A[:, 2*i-1] = np.cos(i*t)
        A[:, 2*i] = np.sin(i*t)
    
    # Solve for coefficients
    coeffs = np.linalg.lstsq(A, x, rcond=None)[0]
    
    # Function to evaluate the fit at any point
    def fourier_func(t_new):
        A_new = np.ones((len(t_new), 2*num_terms+1))
        for i in range(1, num_terms+1):
            A_new[:, 2*i-1] = np.cos(i*t_new)
            A_new[:, 2*i] = np.sin(i*t_new)
        return A_new @ coeffs
    
    return fourier_func

def find_regions_of_interest(y_sorted, deviations, window_size=50, threshold_factor=1.5):
    """
    Dynamically find regions with significant deviations
    
    Parameters:
    - y_sorted: y-coordinates of points, sorted
    - deviations: deviation values at each point
    - window_size: size of the sliding window for analysis
    - threshold_factor: multiplier for deviation threshold (higher = fewer ROIs)
    
    Returns:
    - List of tuples (y_min, y_max) for each ROI
    """
    # Calculate moving average of deviations
    avg_deviation = np.mean(deviations)
    threshold = avg_deviation * threshold_factor
    
    # Find continuous regions where deviation exceeds threshold
    regions = []
    start_idx = None
    
    for i in range(len(deviations)):
        if deviations[i] > threshold:
            if start_idx is None:
                start_idx = i
        elif start_idx is not None:
            # Found the end of a region
            if i - start_idx > 10:  # Minimum region size
                y_min = max(0, y_sorted[start_idx] - window_size//2)
                y_max = min(y_sorted[i-1] + window_size//2, y_sorted[-1])
                regions.append((y_min, y_max))
            start_idx = None
    
    # Handle case where region extends to the end
    if start_idx is not None:
        y_min = max(0, y_sorted[start_idx] - window_size//2)
        y_max = min(y_sorted[-1] + window_size//2, y_sorted[-1])
        regions.append((y_min, y_max))
    
    # Merge overlapping regions
    if regions:
        merged_regions = [regions[0]]
        for current in regions[1:]:
            prev = merged_regions[-1]
            if current[0] <= prev[1]:
                # Regions overlap, merge them
                merged_regions[-1] = (prev[0], max(prev[1], current[1]))
            else:
                # Non-overlapping region
                merged_regions.append(current)
        return merged_regions
    return []

def analyze_line(points, line_num):
    """Analyze deviations for a single line"""
    if len(points) == 0: return

    x = points[:, 1]  # x-coordinates
    y = points[:, 0]  # y-coordinates

    # Keep only points with x > 10 and y >= 10 (excluding top 10 pixels)
    valid_indices = (x > 0) & (y >= 10)
    x = x[valid_indices]
    y = y[valid_indices]
    
    # Sort points by y-coordinate
    sort_idx = np.argsort(y)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Perfect straight line (mean x value)
    straight_x = np.mean(x_sorted)

    # Polynomial fit
    poly_coeffs = np.polyfit(y_sorted, x_sorted, 6)
    poly_eq = np.poly1d(poly_coeffs)
    
    # Fourier fit
    t = np.linspace(0, 2*np.pi, len(y_sorted))
    fourier_func = fourier_fit(x_sorted, y_sorted, num_terms=8)
    
    # Generate points for plotting
    y_range = np.linspace(min(y_sorted), max(y_sorted), 1000)
    t_range = np.linspace(0, 2*np.pi, 1000)
    x_poly = poly_eq(y_range)
    x_fourier = fourier_func(t_range)

    # Calculate deviations
    straight_deviations = np.abs(x_sorted - straight_x)
    fourier_predictions = fourier_func(t)
    fourier_deviations = np.abs(x_sorted - fourier_predictions)
    
    # Calculate RMSE
    poly_rmse = np.sqrt(np.mean(np.abs(x_sorted - poly_eq(y_sorted))**2))
    
    # Find regions of interest based on deviations from both fits
    # Use difference between straight line and Fourier fit to find regions
    fit_difference = np.abs(fourier_predictions - straight_x)
    roi_regions = find_regions_of_interest(y_sorted, fit_difference, window_size=100, threshold_factor=1.3)

    # Create plot
    plt.figure(figsize=(10, 12))
    plt.imshow(thresh, cmap='gray')

    # Enhanced deviation visualization with better colormap
    sc = plt.scatter(x_sorted, y_sorted, c=straight_deviations,
                    cmap='plasma', s=4, vmin=0, vmax=np.max(straight_deviations),
                    alpha=0.8)
    
    # Find and highlight max deviation point
    max_dev_idx = np.argmax(straight_deviations)
    plt.scatter(x_sorted[max_dev_idx], y_sorted[max_dev_idx],
               s=100, edgecolor='red', facecolor='none',
               linewidth=2, label='Max Deviation')
    
    # # Plot polynomial fit
    # plt.plot(x_poly, y_range, 'g-', linewidth=2,
    #         label=f'Curve Fit (RMSE: {poly_rmse:.2f}px)')
    
    # Plot Fourier fit
    plt.plot(x_fourier, y_range, 'y-', linewidth=2,
            label=f'Fourier Fit')
    
    # Plot perfect straight line
    plt.axvline(straight_x, color='cyan', linestyle=':',
               label=f'Ideal Vertical (x = {straight_x:.2f})')
    
    # Highlight dynamically detected ROIs
    for i, (roi_ymin, roi_ymax) in enumerate(roi_regions):
        # Filter points within ROI
        roi_mask = (y_sorted >= roi_ymin) & (y_sorted <= roi_ymax)
        roi_x = x_sorted[roi_mask]
        roi_y = y_sorted[roi_mask]
        
        # Calculate ROI-specific deviations
        if len(roi_x) > 0:
            roi_straight_dev = np.abs(roi_x - straight_x)
            roi_max_dev = np.max(roi_straight_dev)
            roi_mean_dev = np.mean(roi_straight_dev)
            
            # Highlight ROI with rectangle
            plt.gca().add_patch(plt.Rectangle((straight_x-50, roi_ymin), 
                                             100, roi_ymax-roi_ymin, 
                                             fill=False, color='white', linestyle='--'))
            
            # Add text annotation for the region
            mid_y = (roi_ymin + roi_ymax) // 2
            plt.text(straight_x + 50, mid_y, 
                     f"ROI {i+1} Max dev: {roi_max_dev:.2f}px", 
                     color='white', fontweight='bold', 
                     bbox=dict(facecolor='black', alpha=0.7))

    plt.colorbar(sc, label='Horizontal Deviation (pixels)')
    plt.gca().invert_yaxis()
    plt.legend(loc='lower right')
    plt.title(f'Line {line_num} Deviation Analysis\n'
             f'Mean: {np.mean(straight_deviations):.2f}px | '
             f'Max: {np.max(straight_deviations):.2f}px')
    plt.show()

# --- Execute Analysis ---
print("\n=== Processing Results ===")
analyze_line(line1_points, 1)
analyze_line(line2_points, 2)

# --- Display Preprocessing Steps ---
plt.figure(figsize=(15, 10))
plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(232), plt.imshow(mask, cmap='gray'), plt.title('HSV Mask')
plt.subplot(233), plt.imshow(thresh, cmap='gray'), plt.title('Thresh Image')
plt.tight_layout()
plt.show()