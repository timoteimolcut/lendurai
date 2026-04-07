#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>


// #define IMAGE_PATH "/home/timct/github/lendurai/images/image_fpv_drone.png"
#define IMAGE_PATH "/home/timct/github/lendurai/images/src/processed/image_fpv_drone.png"
#define MAP_PATH "/home/timct/github/lendurai/images/satellite_map.png"
#define OUTPUT_PATH "/home/timct/github/lendurai/images/birds_eye_map_high_res_500m2_map.png"

int read_image() {
    std::string image_path = IMAGE_PATH;
    // std::string image_path = path_satellite_map;

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Error: Could not read the image at " << image_path << std::endl;
        return 1; // Exit the program with an error code
    }
    cv::imshow("Loaded Image", image);
    cv::waitKey(0);
    return 0;
}

int pin_hole_camera_model() {
    // ---------------------------------------------------------
    // 1. DEFINE CAMERA PARAMETERS (The Drone's State)
    // ---------------------------------------------------------
    
    // Intrinsic Matrix (K): Focal length 800px, Image center at (640, 360)
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
                 800.0, 0.0, 640.0,
                 0.0, 800.0, 360.0,
                 0.0, 0.0, 1.0);

    // Extrinsic Rotation (R): Let's assume the drone is pitched forward by 30 degrees
    // (A 30-degree rotation around the X-axis)
    double pitch = 30.0 * CV_PI / 180.0; 
    cv::Mat R = (cv::Mat_<double>(3, 3) << 
                 1.0, 0.0, 0.0,
                 0.0, cos(pitch), -sin(pitch),
                 0.0, sin(pitch), cos(pitch));

    // Extrinsic Translation (t): Drone is at Altitude (Z) = 50 meters
    cv::Mat t = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 50.0);

    // ---------------------------------------------------------
    // 2. BUILD THE HOMOGRAPHY MATRIX (H)
    // ---------------------------------------------------------
    
    // Because Z=0, we combine Column 1 of R, Column 2 of R, and the Translation vector (t)
    cv::Mat H = cv::Mat::zeros(3, 3, CV_64F);
    R.col(0).copyTo(H.col(0)); // r1
    R.col(1).copyTo(H.col(1)); // r2
    t.copyTo(H.col(2));        // t vector

    // Multiply by Intrinsics to get the final Homography matrix: H = K * [r1, r2, t]
    H = K * H;

    // ---------------------------------------------------------
    // 3. THE REVERSE CALCULATION (Pixel to World)
    // ---------------------------------------------------------
    
    // Let's say we detect a car at pixel coordinate (u=640, v=500)
    cv::Mat pixel_coord = (cv::Mat_<double>(3, 1) << 640.0, 500.0, 1.0);

    // To go backward, we multiply the pixel by the INVERSE of the Homography matrix
    cv::Mat H_inv = H.inv();
    cv::Mat world_coord = H_inv * pixel_coord;

    // ---------------------------------------------------------
    // 4. THE PERSPECTIVE DIVIDE
    // ---------------------------------------------------------
    
    // The result is a Homogeneous coordinate [X', Y', W']. 
    // We MUST divide by the 3rd element (W') to get the true physical X, Y in meters.
    double W = world_coord.at<double>(2, 0);
    double ground_X = world_coord.at<double>(0, 0) / W;
    double ground_Y = world_coord.at<double>(1, 0) / W;

    // Output the result!
    std::cout << "Target found at Pixel (640, 500)" << std::endl;
    std::cout << "Physical Location relative to drone: " << std::endl;
    std::cout << "X (Right/Left): " << ground_X << " meters" << std::endl;
    std::cout << "Y (Forward): " << ground_Y << " meters" << std::endl;

    return 0;
}

int fun(){
    // 1. Print the OpenCV version to the terminal
    std::cout << "Successfully loaded OpenCV Version: " << CV_VERSION << std::endl;

    // 2. Create a blank image (Height: 400, Width: 600, 8-bit 3-channel color)
    // The cv::Scalar(50, 50, 50) fills it with a dark gray color (Blue, Green, Red)
    cv::Mat image(400, 600, CV_8UC3, cv::Scalar(50, 50, 50));

    // 3. Draw text on the image
    cv::putText(image,                     // Target image
                "Hello, Ubuntu 24 LTS!",   // Text to display
                cv::Point(75, 200),        // Starting position (x, y)
                cv::FONT_HERSHEY_SIMPLEX,  // Font style
                1.0,                       // Font scale
                cv::Scalar(0, 255, 0),     // Color: Green (B:0, G:255, R:0)
                2);                        // Thickness

    // 4. Create a window and show the image
    cv::imshow("My First OpenCV App", image);

    // 5. Pause the program until the user presses any key
    // If you don't do this, the window will close instantly!
    cv::waitKey(0);
    return 0;
}

int camera2map(){
    // =========================================================================
    // STEP 1: LOAD THE ORIGINAL DRONE IMAGE
    // =========================================================================
    std::string image_path = IMAGE_PATH;
    cv::Mat input_image = cv::imread(image_path, cv::IMREAD_COLOR);

    if (input_image.empty()) {
        std::cerr << "CRITICAL ERROR: Could not find 'drone_photo.jpg'. Make sure it is in the build folder!" << std::endl;
        return 1;
    }

    std::cout << "Successfully loaded image: " << input_image.cols << "x" << input_image.rows << std::endl;

    // =========================================================================
    // STEP 2: DEFINE INTRINSIC PARAMETERS (The Lens)
    // =========================================================================
    // Let's assume a 1920x1080 camera with a focal length of 1200 pixels
    double f = 1200.0;
    double cx = input_image.cols / 2.0; // Center X (960)
    double cy = input_image.rows / 2.0; // Center Y (540)

    cv::Mat K = (cv::Mat_<double>(3, 3) << 
                 f,   0.0, cx,
                 0.0, f,   cy,
                 0.0, 0.0, 1.0);

    // =========================================================================
    // STEP 3: DEFINE EXTRINSIC PARAMETERS (The Drone's Position)
    // =========================================================================
    // Altitude in meters
    double altitude = 30.0; 
    cv::Mat t = (cv::Mat_<double>(3, 1) << 0.0, 0.0, altitude);

    // Pitch: Let's assume the drone camera is tilted downward by 45 degrees.
    // We convert degrees to radians for the C++ math functions.
    double pitch_deg = 45.0; 
    double pitch_rad = pitch_deg * CV_PI / 180.0;

    // Rotation Matrix around the X-axis (Pitch)
    cv::Mat R = (cv::Mat_<double>(3, 3) << 
                 1.0, 0.0,            0.0,
                 0.0, cos(pitch_rad), -sin(pitch_rad),
                 0.0, sin(pitch_rad), cos(pitch_rad));

    // =========================================================================
    // STEP 4: BUILD THE HOMOGRAPHY MATRIX (H)
    // =========================================================================
    // We assume the ground is flat (Z=0). This deletes the 3rd column of R.
    // We build H by combining Column 1 of R, Column 2 of R, and Translation t.
    cv::Mat H_extrinsic = cv::Mat::zeros(3, 3, CV_64F);
    R.col(0).copyTo(H_extrinsic.col(0)); // r1
    R.col(1).copyTo(H_extrinsic.col(1)); // r2
    t.copyTo(H_extrinsic.col(2));        // t

    // The final Homography is Intrinsic * Extrinsic
    cv::Mat H = K * H_extrinsic;

    // Optional but recommended: Normalize the matrix so the bottom right value is 1.0
    H = H / H.at<double>(2, 2);

    // =========================================================================
    // STEP 5: WARP THE IMAGE (Inverse Perspective Mapping)
    // =========================================================================
    std::cout << "Calculating Bird's Eye View... (This happens on the CPU)" << std::endl;
    cv::Mat output_image;
    
    // We define how big we want the final map to be. 
    // Let's make it the same resolution as the input for now.
    cv::Size output_size(input_image.cols, input_image.rows);

    // Run the massive SIMD math operation to warp every pixel
    cv::warpPerspective(input_image, output_image, H, output_size, cv::INTER_LINEAR);

    // =========================================================================
    // STEP 6: SAVE THE RESULT TO THE RASPBERRY PI'S SD CARD
    // =========================================================================
    std::string output_path = OUTPUT_PATH;
    cv::imwrite(output_path, output_image);

    std::cout << "Success! Saved mapped image to: " << OUTPUT_PATH << std::endl;

    return 0;
}

int camera2map_simple(){
    // =========================================================
    // 1. LOAD THE DRONE PHOTO
    // =========================================================
    cv::Mat input_image = cv::imread(IMAGE_PATH);

    if (input_image.empty()) {
        std::cerr << "Error: Could not find 'drone_photo.jpg'." << std::endl;
        return 1;
    }

    // =========================================================
    // 2. DEFINE THE PHYSICAL SCALE (500 m^2 Area)
    // =========================================================
    // Let's assume the area we want to map is 25m wide and 20m tall (500 m^2)
    double area_width_m  = 25.0;
    double area_height_m = 20.0;

    // We have plenty of RAM, so let's use an ultra-sharp resolution:
    // 1 real-world centimeter = 1 pixel (which means 100 pixels per meter)
    double pixels_per_meter = 100.0; 

    // Calculate the final image resolution in pixels
    // Width: 25m * 100 = 2500 pixels
    // Height: 20m * 100 = 2000 pixels
    int output_width_px  = area_width_m * pixels_per_meter;
    int output_height_px = area_height_m * pixels_per_meter;

    // =========================================================
    // 3. DEFINE DESTINATION POINTS (The Perfect Rectangle)
    // =========================================================
    // We map the 4 corners of our new 2500x2000 map.
    std::vector<cv::Point2f> dst_pts = {
        cv::Point2f(0, 0),                               // Top Left
        cv::Point2f(output_width_px, 0),                 // Top Right
        cv::Point2f(output_width_px, output_height_px),  // Bottom Right
        cv::Point2f(0, output_height_px)                 // Bottom Left
    };

    // =========================================================
    // 4. DEFINE SOURCE POINTS (From your distorted photo)
    // =========================================================
    // IMPORTANT: You must open your image and find the X,Y pixel coordinates 
    // of the 4 corners that represent that 25x20m area in real life.
    // Replace these dummy numbers with the actual coordinates from your photo!
    std::vector<cv::Point2f> src_pts = {
        cv::Point2f(400, 500),   // Top Left corner of area in photo
        cv::Point2f(1500, 500),  // Top Right corner of area in photo
        cv::Point2f(1800, 1000), // Bottom Right corner of area in photo
        cv::Point2f(100, 1000)   // Bottom Left corner of area in photo
    };

    // =========================================================
    // 5. CALCULATE HOMOGRAPHY AND WARP
    // =========================================================
    std::cout << "Calculating matrix..." << std::endl;
    cv::Mat H = cv::getPerspectiveTransform(src_pts, dst_pts);

    std::cout << "Warping image to " << output_width_px << "x" << output_height_px << " pixels..." << std::endl;
    cv::Mat map_image;
    cv::warpPerspective(input_image, map_image, H, cv::Size(output_width_px, output_height_px));

    // =========================================================
    // 6. SAVE THE BIRD'S EYE MAP
    // =========================================================
    cv::imwrite(OUTPUT_PATH, map_image);
    std::cout << "Success! Saved ultra-high-resolution map to SD card." << std::endl;

    return 0;
}

int main() {
    // read_image();
    // pin_hole_camera_model();
    // camera2map();
    // camera2map_simple();

    std::string path = IMAGE_PATH;
    cv::Mat input_image = cv::imread(path, cv::IMREAD_COLOR);
    if (input_image.empty()){
        std::cerr << "Image not found: " << path << std::endl;
        return 1;
    }
    std::cout << "Image loaded" << std::endl;
    std::cout << "Height: " << input_image.rows << std::endl;
    std::cout << "Width: " << input_image.cols << std::endl;
    // =========================================================
    // STEP 2: BUILD THE BLANK CANVAS (Destination Points)
    // =========================================================
    
    // 1. Define the physical size of the dirt we are mapping (25m x 20m)
    double area_width_m  = 25.0;
    double area_height_m = 20.0;

    // 2. Define our scale: 1 real-world centimeter = 1 pixel
    // This means there are 100 pixels in every 1 physical meter.
    double pixels_per_meter = 100.0; 

    // 3. Calculate how massive our new image needs to be in pixels
    // Width: 25 * 100 = 2500 pixels
    // Height: 20 * 100 = 2000 pixels
    int output_width_px  = area_width_m * pixels_per_meter;
    int output_height_px = area_height_m * pixels_per_meter;

    // 4. Define the 4 corners of this perfect new canvas
    // We store these inside a "vector" (which is just a C++ list).
    // cv::Point2f is just a tiny container holding an X and Y coordinate.
    std::vector<cv::Point2f> dst_pts = {
        cv::Point2f(0, 0),                               // Top-Left corner
        cv::Point2f(output_width_px, 0),                 // Top-Right corner
        cv::Point2f(output_width_px, output_height_px),  // Bottom-Right corner
        cv::Point2f(0, output_height_px)                 // Bottom-Left corner
    };

    std::cout << "Step 2 Complete: Canvas created at " << output_width_px << "x" << output_height_px << std::endl;

    // =========================================================
    // STEP 3: DEFINE THE SOURCE POINTS (From your 611x428 photo)
    // =========================================================
    
    // We create a list of 4 points. 
    // IMPORTANT: The order MUST be: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    
    // NOTE: These are example points! You will need to change these numbers 
    // to match the actual corners of an object in your specific photo.
    std::vector<cv::Point2f> src_pts = {
        cv::Point2f(100, 150),   // Top-Left corner of the rug
        cv::Point2f(500, 120),   // Top-Right corner of the rug
        cv::Point2f(580, 400),   // Bottom-Right corner of the rug
        cv::Point2f(50, 380)     // Bottom-Left corner of the rug
    };

    std::cout << "Step 3 Complete: Mapped the distorted shape in the photo." << std::endl;
    // =========================================================
    // STEP 4: CONNECT THE DOTS (The Math & Warp)
    // =========================================================
    
    std::cout << "Calculating the math..." << std::endl;

    // 1. Calculate the "Rulebook" (The Homography Matrix)
    // OpenCV looks at src_pts and dst_pts and figures out exactly how much 
    // to stretch, pull, and rotate the image to make them match.
    cv::Mat H = cv::getPerspectiveTransform(src_pts, dst_pts);

    // 2. Create an empty grid in memory to hold our final mapped image
    cv::Mat final_map;

    std::cout << "Warping the image. This might take a millisecond..." << std::endl;

    // 3. Do the actual stretching!
    // We give it: Original Photo -> Empty Grid -> The Rulebook -> The Canvas Size
    cv::warpPerspective(input_image, final_map, H, cv::Size(output_width_px, output_height_px));
    cv::imwrite(OUTPUT_PATH, final_map);

    std::cout << "Step 4 Complete: Image successfully warped." << std::endl;

    cv::imshow("src", input_image);
    cv::imshow("dst", final_map);
    cv::waitKey(0);
    return 0;
}


