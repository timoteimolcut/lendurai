#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

// #define IMAGE_PATH "/home/timct/github/lendurai/images/src/image_fpv_drone.png"
#define IMAGE_PATH "/home/timct/github/lendurai/images/src/processed/image_fpv_drone.png"
#define MAP_PATH "/home/timct/github/lendurai/images/src/satellite_map.png"
#define OUTPUT_PATH "/home/timct/github/lendurai/images/dst/birds_eye_map_high_res_500m2_map.png"


// =========================================================
// GLOBAL VARIABLES (Needed for the mouse to talk to main)
// =========================================================
std::vector<cv::Point2f> src_pts; // This will store our 4 clicks
cv::Mat display_image;            // A copy of the image to draw green dots on

// =========================================================
// THE MOUSE LISTENER FUNCTION
// =========================================================
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        if (src_pts.size() < 4) {
            // Save the X, Y coordinate to our list
            src_pts.push_back(cv::Point2f(x, y));
            
            // Draw a tiny green circle where the user clicked so they can see it!
            cv::circle(display_image, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);
            
            // Update the window to show the new green circle
            cv::imshow("Click the 4 corners (Top-L, Top-R, Bot-R, Bot-L)", display_image);
            
            std::cout << "Point " << src_pts.size() << " saved at: X=" << x << ", Y=" << y << std::endl;
        }
    }
}

int main() {
    // =========================================================
    // STEP 1: LOAD THE IMAGE
    // =========================================================
    cv::Mat input_image = cv::imread(IMAGE_PATH, cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Error: Could not find 'drone_photo.jpg'." << std::endl;
        return 1;
    }

    // Make a copy for the UI so we don't warp the green dots later
    display_image = input_image.clone(); 

    // =========================================================
    // STEP 2: THE INTERACTIVE UI (Click your points)
    // =========================================================
    std::cout << "Opening UI..." << std::endl;
    std::cout << "Please click the 4 corners in this order:" << std::endl;
    std::cout << "1. Top-Left  2. Top-Right  3. Bottom-Right  4. Bottom-Left" << std::endl;

    // Create a named window
    cv::namedWindow("Click the 4 corners (Top-L, Top-R, Bot-R, Bot-L)");
    
    // Tell OpenCV: "When the mouse moves in this window, run the onMouse function"
    cv::setMouseCallback("Click the 4 corners (Top-L, Top-R, Bot-R, Bot-L)", onMouse, nullptr);
    
    // Show the image
    cv::imshow("Click the 4 corners (Top-L, Top-R, Bot-R, Bot-L)", display_image);

    // PAUSE THE PROGRAM: Wait here in a loop until the user clicks 4 times
    while (src_pts.size() < 4) {
        cv::waitKey(10); // Wait 10 milliseconds, then check again
    }

    std::cout << "4 points collected! Closing UI and crunching the math..." << std::endl;
    cv::destroyAllWindows(); // Close the picture window

    // =========================================================
    // STEP 3: BUILD THE BLANK CANVAS (Destination Points)
    // =========================================================
    double area_width_m  = 25.0;
    double area_height_m = 20.0;
    double pixels_per_meter = 100.0; 
    int output_width_px  = area_width_m * pixels_per_meter;
    int output_height_px = area_height_m * pixels_per_meter;

    std::vector<cv::Point2f> dst_pts = {
        cv::Point2f(0, 0),
        cv::Point2f(output_width_px, 0),
        cv::Point2f(output_width_px, output_height_px),
        cv::Point2f(0, output_height_px)
    };

    cv::Mat sharpened;
    cv::GaussianBlur(input_image, sharpened, cv::Size(0, 0), 3);
    cv::addWeighted(input_image, 1.5, sharpened, -0.5, 0, sharpened);
    // Now warp 'sharpened' instead of 'input_image'

    // =========================================================
    // STEP 4: CONNECT THE DOTS (The Math & Warp)
    // =========================================================
    cv::Mat H = cv::getPerspectiveTransform(src_pts, dst_pts);
    cv::Mat map_image;
    // cv::warpPerspective(input_image, map_image, H, cv::Size(output_width_px, output_height_px));
    // Use LANCZOS4 for the highest possible sharpness on the RPi5
    cv::warpPerspective(sharpened, map_image, H, cv::Size(output_width_px, output_height_px), cv::INTER_LANCZOS4);

    // =========================================================
    // STEP 5: SAVE THE MAP
    // =========================================================
    cv::imwrite(OUTPUT_PATH, map_image);
    std::cout << "Success! Saved mapped image to SD card." << std::endl;

    cv::imshow("Resulted map", map_image);
    cv::waitKey(0);
    return 0;
}