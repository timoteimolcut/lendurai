#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <ctime>
#include <sstream>
#include <algorithm>

#define IMAGE_PATH "/home/timct/github/lendurai/images/src/processed/image_fpv_drone.png"
#define OUTPUT_PATH "/home/timct/github/lendurai/images/dst/sweep/birds_eye_map"

static const int OUTPUT_SIZE   = 800;
static const int OUTPUT_CENTER = OUTPUT_SIZE / 2;

struct GroundPoint {
    double x;
    double y;
};

// Standard Filename Generator (unchanged)
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

// ============================================================
//  GEOMETRY ENGINE
//
//  Camera convention: optical axis = +Z, image right = +X, image down = +Y.
//  World frame:       forward = +Y, right = +X, up = +Z.
//
//  Rotations are applied in order: pitch → roll → yaw.
//  Each is a simple axis rotation so the convention stays explicit
//  and easy to verify. When roll = yaw = 0 the result is identical
//  to the original pitch-only code.
// ============================================================
GroundPoint project_to_ground(double alt,
                               double pitch_rad, double roll_rad, double yaw_rad,
                               double h_fov_rad, double v_fov_rad,
                               double img_x, double img_y)
{
    // 1. Ray in Camera Space
    double vx = tan(h_fov_rad / 2.0) * img_x;
    double vy = tan(v_fov_rad / 2.0) * img_y;
    double vz = 1.0;

    // 2. Pitch: rotate around camera X axis (negative pitch = nose down).
    //    This is your original, proven-correct rotation.
    //    optical axis (vz=1) maps to world +Y (forward) when pitch=0.
    double rx1 = vx;
    double ry1 = vz * cos(pitch_rad) + vy * sin(pitch_rad);
    double rz1 = vz * sin(pitch_rad) - vy * cos(pitch_rad);

    // 3. Roll: rotate around the forward axis (world Y after pitch).
    //    Positive roll = right wing down.
    double rx2 =  rx1 * cos(roll_rad) + rz1 * sin(roll_rad);
    double ry2 =  ry1;
    double rz2 = -rx1 * sin(roll_rad) + rz1 * cos(roll_rad);

    // 4. Yaw: rotate around the world Z (up) axis.
    //    Positive yaw = nose turns right.
    double ray_x =  rx2 * cos(yaw_rad) - ry2 * sin(yaw_rad);
    double ray_y =  rx2 * sin(yaw_rad) + ry2 * cos(yaw_rad);
    double ray_z =  rz2;

    // 5. Prevent Homography Collapse.
    //    Ray must point downward (ray_z < 0) to hit the ground.
    //    Clamp rays at or above the horizon so the 4-corner trapezoid
    //    stays well-defined at the crop boundary.
    if (ray_z > -1e-6) ray_z = -1e-6;

    // 6. Ground Intersection: solve  altitude + t * ray_z = 0
    double t = -alt / ray_z;

    return { ray_x * t, ray_y * t };
}

// ============================================================
//  HORIZON ESTIMATION  (used when pitch is unknown / AUTO mode)
//
//  Sky rows have low horizontal-edge energy; ground rows have high
//  energy. Finds the row with the steepest upward jump in per-row
//  mean Sobel-X magnitude.
// ============================================================
int estimate_horizon_row(const cv::Mat& img) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);

    cv::Mat sx;
    cv::Sobel(gray, sx, CV_32F, 1, 0, 3);
    sx = cv::abs(sx);

    std::vector<float> energy(img.rows);
    for (int r = 0; r < img.rows; r++)
        energy[r] = static_cast<float>(cv::mean(sx.row(r))[0]);

    // 5-tap box smooth
    std::vector<float> smooth(img.rows, 0.f);
    for (int r = 2; r < img.rows - 2; r++)
        smooth[r] = (energy[r-2]+energy[r-1]+energy[r]+energy[r+1]+energy[r+2]) / 5.f;

    int   search_start = img.rows / 6;
    int   search_end   = img.rows * 2 / 3;
    int   horizon_row  = img.rows / 3;
    float max_delta    = 0.f;
    for (int r = search_start; r < search_end - 1; r++) {
        float delta = smooth[r+1] - smooth[r];
        if (delta > max_delta) { max_delta = delta; horizon_row = r; }
    }
    return horizon_row;
}

double pitch_from_horizon(int horizon_row, int img_height, double v_fov_rad) {
    double norm_y = 2.0 * horizon_row / img_height - 1.0;
    return atan(norm_y * tan(v_fov_rad / 2.0));
}

// ============================================================
//  BIRD'S-EYE PROJECTION
// ============================================================
cv::Mat generate_birdseye_view(const cv::Mat& fpv_img,
                                double alt,
                                double pitch_rad, double roll_rad, double yaw_rad,
                                double h_fov_rad, double v_fov_rad)
{
    const int W = fpv_img.cols;
    const int H = fpv_img.rows;

    // ----------------------------------------------------------
    // 1. Compute horizon row and crop to the ground-only strip.
    //    Everything at or above the horizon projects to infinity
    //    and would blow up the homography — discard it.
    // ----------------------------------------------------------
    double horizon_norm_y = tan(pitch_rad) / tan(v_fov_rad / 2.0);
    int    horizon_px     = static_cast<int>((horizon_norm_y + 1.0) / 2.0 * H);
    int    margin         = std::max(1, static_cast<int>(H * 0.05));
    int    crop_start     = std::clamp(horizon_px + margin, 0, H - 2);

    std::cout << "  Horizon at row " << horizon_px << " / " << H << std::endl;
    std::cout << "  Ground strip: rows [" << crop_start << ", " << H << ")" << std::endl;

    cv::Mat strip = fpv_img(cv::Rect(0, crop_start, W, H - crop_start));

    // ----------------------------------------------------------
    // 2. Map the 4 corners of the strip to ground metres,
    //    then to output canvas pixels.
    //    Normalised coords are in the original full image frame.
    // ----------------------------------------------------------
    double ny_top    = 2.0 * crop_start / H - 1.0;
    double ny_bottom = 1.0;

    double norm_coords[4][2] = {
        { -1.0, ny_top    },   // top-left  of strip
        {  1.0, ny_top    },   // top-right
        {  1.0, ny_bottom },   // bottom-right
        { -1.0, ny_bottom }    // bottom-left
    };

    std::vector<cv::Point2f> src_pixels = {
        { 0.f,      0.f                },
        { (float)W, 0.f                },
        { (float)W, (float)strip.rows  },
        { 0.f,      (float)strip.rows  }
    };

    std::vector<cv::Point2f> dst_pixels;

    std::cout << "\n--- Corner Mapping ---" << std::endl;
    for (int i = 0; i < 4; i++) {
        GroundPoint m = project_to_ground(alt,
                                          pitch_rad, roll_rad, yaw_rad,
                                          h_fov_rad, v_fov_rad,
                                          norm_coords[i][0], norm_coords[i][1]);
        float px = OUTPUT_CENTER + (float)m.x;
        float py = OUTPUT_CENTER - (float)m.y;   // screen Y flipped vs world Y
        std::cout << "Corner " << i
                  << ": Metres(" << m.x << ", " << m.y << ")"
                  << " -> Pixel(" << px  << ", " << py  << ")" << std::endl;
        dst_pixels.push_back(cv::Point2f(px, py));
    }

    // ----------------------------------------------------------
    // 3. Compute homography and warp the ground strip
    // ----------------------------------------------------------
    cv::Mat Hmat = cv::getPerspectiveTransform(src_pixels, dst_pixels);
    cv::Mat result;
    cv::warpPerspective(strip, result, Hmat,
                        cv::Size(OUTPUT_SIZE, OUTPUT_SIZE),
                        cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT,
                        cv::Scalar(0, 0, 0));

    // ----------------------------------------------------------
    // 4. Mark drone position (centre of output canvas)
    // ----------------------------------------------------------
    cv::circle(result, { OUTPUT_CENTER, OUTPUT_CENTER },  5, cv::Scalar(0, 0, 255), -1);
    cv::circle(result, { OUTPUT_CENTER, OUTPUT_CENTER }, 10, cv::Scalar(0, 0, 255),  2);

    return result;
}

// ============================================================
//  MAIN
// ============================================================
int main(int argc, char** argv) {
    if (argc < 7) {
        std::cout << "Usage: " << argv[0]
                  << " <alt_m> <pitch_deg|AUTO> <roll_deg> <yaw_deg> <hfov_deg> <vfov_deg>"
                  << std::endl;
        std::cout << "  pitch_deg: negative = nose down. Pass AUTO to estimate from image."
                  << std::endl;
        std::cout << "Example: " << argv[0] << " 60 -25 0 0 110 80" << std::endl;
        std::cout << "Example: " << argv[0] << " 60 AUTO 0 0 110 80" << std::endl;
        return -1;
    }

    double      altitude  = std::atof(argv[1]);
    std::string pitch_arg = argv[2];
    double      roll_deg  = std::atof(argv[3]);
    double      yaw_deg   = std::atof(argv[4]);
    double      h_fov_deg = std::atof(argv[5]);
    double      v_fov_deg = std::atof(argv[6]);

    // Guard: altitude must be positive (zero causes division by zero in ground projection)
    if (altitude <= 0.0) {
        std::cerr << "ERROR: altitude must be > 0 m, got " << altitude << std::endl;
        return -1;
    }
    if (h_fov_deg <= 0.0 || h_fov_deg >= 180.0 || v_fov_deg <= 0.0 || v_fov_deg >= 180.0) {
        std::cerr << "ERROR: FOV values must be in (0, 180) degrees." << std::endl;
        return -1;
    }

    double h_fov_rad = h_fov_deg * (M_PI / 180.0);
    double v_fov_rad = v_fov_deg * (M_PI / 180.0);
    double roll_rad  = roll_deg  * (M_PI / 180.0);
    double yaw_rad   = yaw_deg   * (M_PI / 180.0);

    cv::Mat src = cv::imread(IMAGE_PATH);
    if (src.empty()) {
        std::cerr << "ERROR: Cannot find image at " << IMAGE_PATH << std::endl;
        return -1;
    }

    double pitch_rad;
    if (pitch_arg == "AUTO") {
        int horizon_row = estimate_horizon_row(src);
        pitch_rad       = pitch_from_horizon(horizon_row, src.rows, v_fov_rad);
        std::cout << "AUTO pitch estimation:\n"
                  << "  horizon row : " << horizon_row << " / " << src.rows << "\n"
                  << "  pitch       : " << pitch_rad * 180.0 / M_PI << " deg\n";
    } else {
        pitch_rad = std::atof(pitch_arg.c_str()) * (M_PI / 180.0);
    }

    std::cout << "\nParameters:\n"
              << "  image    : " << src.cols << "x" << src.rows << "\n"
              << "  altitude : " << altitude  << " m\n"
              << "  pitch    : " << pitch_rad * 180.0 / M_PI << " deg\n"
              << "  roll     : " << roll_deg  << " deg\n"
              << "  yaw      : " << yaw_deg   << " deg\n"
              << "  HFOV     : " << h_fov_deg << " deg\n"
              << "  VFOV     : " << v_fov_deg << " deg\n\n";

    cv::Mat result = generate_birdseye_view(src, altitude,
                                             pitch_rad, roll_rad, yaw_rad,
                                             h_fov_rad, v_fov_rad);

    std::vector<double> info = { altitude,
                                  pitch_rad * 180.0 / M_PI,
                                  roll_deg, yaw_deg,
                                  h_fov_deg, v_fov_deg };
    std::string output_file_name = timestampedFilename(OUTPUT_PATH, info, "png");

    std::cout << "Saving: " << output_file_name << std::endl;
    cv::imwrite(output_file_name, result);

    cv::imshow("Bird's Eye View", result);
    cv::waitKey(0);

    return 0;
}