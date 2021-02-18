#include <iostream>

// include depthai library
#include <depthai/depthai.hpp>

// include depthai-utility
#include "utility.hpp"

// include opencv library (Optional, used only for the following example)
#include <opencv2/opencv.hpp>

int main(){
    // Create an empty Pipeline object
    dai::Pipeline pipeline;

    // Define ColorCamera Node
    auto cam_rgb = pipeline.create<dai::node::ColorCamera>();
    cam_rgb->setPreviewSize(300, 300); // to match the mobilenet-ssd input size
    cam_rgb->setInterleaved(true);

    // Create XLinkOut Nodes to receive outputs from OAK
    // Receive color camera frames from the ColorCamera Node
    auto xout_rgb = pipeline.create<dai::node::XLinkOut>();
    xout_rgb->setStreamName("rgb");
    cam_rgb->preview.link(xout_rgb->input);
    // Create pipeline
    /*dai::Pipeline pipeline;
    std::shared_ptr<dai::node::ColorCamera> colorCam = pipeline.create<dai::node::ColorCamera>();
    std::shared_ptr<dai::node::XLinkOut> xlinkOut = pipeline.create<dai::node::XLinkOut>();
    xlinkOut->setStreamName("preview");
    colorCam->setInterleaved(true);
    colorCam->preview.link(xlinkOut->input);
    //colorCam->setPreviewSize(1280, 720);
    //colorCam->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
*/
    try {
        // Try connecting to device
        dai::Device device(pipeline);

        // Get output queues
        auto q_rgb = device.getOutputQueue("rgb");

        // Start pipeline
        device.startPipeline();

        // Variables to store Node outputs
        //cv::Mat frame;
        struct Detection {
            unsigned int label;
            float score;
            float x_min;
            float y_min;
            float x_max;
            float y_max;
        };

        while (true) {
            // Receive 'preview' frame from device 
            std::shared_ptr<dai::ImgFrame> in_rgb = q_rgb->get<dai::ImgFrame>();

            // Show the received 'preview' frame
            /*if (in_rgb) {
                printf("Frame - w: %d, h: %d\n", in_rgb->getWidth(), in_rgb->getHeight());
                // Convert to cv::Mat data type
                frame = toMat(in_rgb->getData(), in_rgb->getWidth(), in_rgb->getHeight(), 3, 1);
            }*/
            cv::Mat frame(in_rgb->getHeight(), in_rgb->getWidth(), CV_8UC3, in_rgb->getData().data());
            cv::imshow("preview", frame);

            // Wait and check if 'q' pressed
            if (cv::waitKey(1) == 'q') {
                return 0;
            }
        }
    } catch (const std::runtime_error& err) {
        std::cout << err.what() << std::endl;
    }

    return 0;
}
