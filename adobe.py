import cv2
import numpy as np
from svgpathtools import Path, CubicBezier
from skimage.measure import approximate_polygon

def load_image(image_path):
    # Load image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img

def detect_edges(img):
    # Use Canny edge detection to find edges
    edges = cv2.Canny(img, 100, 200)
    return edges

def find_contours_and_approximate(edges):
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Approximate each contour with Bézier curves
    beziers = []
    for contour in contours:
        # Reshape the contour to remove the redundant dimension
        contour = contour.reshape(-1, 2)
        
        # Simplify the contour to reduce the number of points
        approx = approximate_polygon(contour, tolerance=2.0)
        
        # Convert the simplified contour to cubic Bézier curves
        path = Path()
        for i in range(len(approx) - 1):
            p0 = np.array(approx[i])
            p3 = np.array(approx[i + 1])
            # Control points are approximated; you can refine this further
            p1 = p0 + 0.33 * (p3 - p0)
            p2 = p0 + 0.66 * (p3 - p0)
            bezier = CubicBezier(complex(p0[0], p0[1]), complex(p1[0], p1[1]), complex(p2[0], p2[1]), complex(p3[0], p3[1]))
            path.append(bezier)
        
        beziers.append(path)
    
    return beziers

def main(image_path):
    img = load_image(image_path)
    edges = detect_edges(img)
    beziers = find_contours_and_approximate(edges)
    
    for path in beziers:
        print(path)  # Outputs the Bezier curves

if __name__ == "__main__":
    image_path = r"C:\Users\peehu\OneDrive\Desktop\adobe.png"
    main(image_path)
