#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

// #define IMAGE_PATH "/home/timct/github/lendurai/images/src/image_fpv_drone.png"
#define IMAGE_PATH "/home/timct/github/lendurai/images/src/processed/image_fpv_drone.png"


// Global variables for the "Knobs"
int i_pitch = 45, i_roll = 50, i_yaw = 50, i_alt = 50;
Mat src;

void update(int, void*) {
    // 1. Convert trackbar (0-100) to real angles/height
    double pitch = (i_pitch - 90) * CV_PI / 180.0; // -90 to 0 degrees
    double roll  = (i_roll - 50)  * CV_PI / 180.0; // -50 to 50
    double alt   = i_alt + 10;                     // 10 to 110 meters

    // 2. Define the output canvas (800x800, 1px = 1m)
    int W = 800, H = 800;
    Point2f dst_pts[4] = { {0,0}, {(float)W,0}, {(float)W,(float)H}, {0,(float)H} };

    // 3. The Math: Create a basic rotation matrix based on your knobs
    // (Simplified for this exercise to show the "stretch")
    Mat r_pitch = (Mat_<double>(3,3) << 1,0,0, 0,cos(pitch),-sin(pitch), 0,sin(pitch),cos(pitch));
    Mat r_roll  = (Mat_<double>(3,3) << cos(roll),0,sin(roll), 0,1,0, -sin(roll),0,cos(roll));
    
    // 4. Calculate Perspective Transform
    // We'll use a simpler 'Source' rectangle for now to show the effect
    Point2f src_pts[4] = {
        Point2f(200 - i_roll, 200 + i_pitch), 
        Point2f(400 + i_roll, 200 + i_pitch),
        Point2f(500 + i_roll, 400 - i_pitch),
        Point2f(100 - i_roll, 400 - i_pitch)
    };

    Mat M = getPerspectiveTransform(src_pts, dst_pts);
    Mat warped;
    warpPerspective(src, warped, M, Size(W, H), INTER_LINEAR);

    imshow("Virtual Drone Map", warped);
}

int main() {
    src = imread(IMAGE_PATH);
    if(src.empty()) return -1;

    namedWindow("Virtual Drone Map");
    createTrackbar("Pitch", "Virtual Drone Map", &i_pitch, 90, update);
    createTrackbar("Roll",  "Virtual Drone Map", &i_roll,  100, update);
    createTrackbar("Alt",   "Virtual Drone Map", &i_alt,   200, update);

    update(0, 0); // Run once to start
    waitKey(0);
    return 0;
}