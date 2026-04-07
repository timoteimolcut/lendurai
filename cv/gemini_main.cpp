#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <ctime>
#include <sstream>

#define IMAGE_PATH "/home/timct/github/lendurai/images/src/processed/image_fpv_drone.png"
#define OUTPUT_PATH "/home/timct/github/lendurai/images/dst/sweep/birds_eye_map"

struct GroundPoint {
    double x; 
    double y; 
};

// Standard Filename Generator
std::string timestampedFilename(const std::string& base, const std::vector<double>& info, const std::string& ext) {
    std::time_t now = std::time(nullptr);
    char buf[20];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&now));
    std::stringstream res;
    res << base << "_" << buf;
    for (double val : info) res << "_" << val;
    res << "." << ext;
    return res.str();
}

/**
 * THE FIXED GEOMETRY ENGINE (Aviation Standard)
 */
GroundPoint project_to_ground(double alt, double pitch_rad, double h_fov_rad, double v_fov_rad, double img_x, double img_y) {
    // 1. Ray in Camera Space
    double vx = tan(h_fov_rad / 2.0) * img_x;
    double vy = tan(v_fov_rad / 2.0) * img_y;
    double vz = 1.0; 

    // 2. Rotate to World Space (Drone pitched DOWN is a negative angle)
    // Z_world is Up, Y_world is Forward, X_world is Right
    double ray_x = vx;
    double ray_y = vz * cos(pitch_rad) + vy * sin(pitch_rad);
    double ray_z = vz * sin(pitch_rad) - vy * cos(pitch_rad);

    // 3. Prevent Homography Collapse
    // If ray_z is positive or 0, it's pointing at the sky/horizon.
    // We clamp it to a tiny negative number so it hits the ground incredibly far away,
    // preserving the 4-sided trapezoid shape for the perspective warp.
    if (ray_z > -1e-4) {
        ray_z = -1e-4; 
    }

    // 4. Ground Intersection
    double S = -alt / ray_z;

    GroundPoint res;
    res.x = ray_x * S; 
    res.y = ray_y * S;
    return res;
}

cv::Mat generate_birdseye_view(cv::Mat fpv_img, double alt, double pitch_rad, double h_fov_rad, double v_fov_rad) {
    std::vector<cv::Point2f> src_pixels = {
        {0, 0}, {(float)fpv_img.cols, 0}, 
        {(float)fpv_img.cols, (float)fpv_img.rows}, {0, (float)fpv_img.rows}
    };

    std::vector<cv::Point2f> dst_pixels;
    double norm_coords[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};

    std::cout << "\n--- Corner Mapping ---" << std::endl;
    for (int i = 0; i < 4; i++) {
        GroundPoint m = project_to_ground(alt, pitch_rad, h_fov_rad, v_fov_rad, norm_coords[i][0], norm_coords[i][1]);
        
        // Map to 800x800. Drone at center (400, 400).
        float px = 400.0f + (float)m.x;
        float py = 400.0f - (float)m.y;
        
        std::cout << "Corner " << i << ": Meters(" << m.x << ", " << m.y << ") -> Pixel(" << px << ", " << py << ")" << std::endl;
        dst_pixels.push_back(cv::Point2f(px, py));
    }

    cv::Mat H = cv::getPerspectiveTransform(src_pixels, dst_pixels);
    cv::Mat result;
    cv::warpPerspective(fpv_img, result, H, cv::Size(800, 800), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return result;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <alt_m> <pitch_deg> <hfov_deg> <vfov_deg>" << std::endl;
        return -1;
    }

    double altitude = std::atof(argv[1]);
    double pitch_deg = std::atof(argv[2]);
    double h_fov_deg = std::atof(argv[3]);
    double v_fov_deg = std::atof(argv[4]);

    // Convert to Radians
    double pitch_rad = pitch_deg * (M_PI / 180.0);
    double h_fov_rad = h_fov_deg * (M_PI / 180.0);
    double v_fov_rad = v_fov_deg * (M_PI / 180.0);

    cv::Mat src = cv::imread(IMAGE_PATH);
    if (src.empty()) {
        std::cerr << "ERROR: Cannot find image at " << IMAGE_PATH << std::endl;
        return -1;
    }

    cv::Mat result = generate_birdseye_view(src, altitude, pitch_rad, h_fov_rad, v_fov_rad);

    std::vector<double> info = {altitude, pitch_deg, h_fov_deg, v_fov_deg};
    std::string output_file_name = timestampedFilename(OUTPUT_PATH, info, "png");

    std::cout << "Saving: " << output_file_name << std::endl;
    cv::imwrite(output_file_name, result);

    cv::imshow("Bird's Eye View", result);
    // CHANGED THIS TO 0: The window will now stay open until you press a key!
    cv::waitKey(0);
    return 0;
}