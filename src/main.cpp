#include <iostream>

// include depthai library
#include <depthai/depthai.hpp>

// include opencv library (Optional, used only for the following example)
#include <opencv2/opencv.hpp>

int main(){
    // Create pipeline
    dai::Pipeline pipeline;
    std::shared_ptr<dai::node::ColorCamera> colorCam = pipeline.create<dai::node::ColorCamera>();
    std::shared_ptr<dai::node::XLinkOut> xlinkOut = pipeline.create<dai::node::XLinkOut>();
    xlinkOut->setStreamName("preview");
    colorCam->setInterleaved(true);
    colorCam->preview.link(xlinkOut->input);
    //colorCam->setPreviewSize(1280, 720);
    //colorCam->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);

    try {
        // Try connecting to device
        dai::Device device(pipeline);

        // Get output queue
        std::shared_ptr<dai::DataOutputQueue> preview = device.getOutputQueue("preview");

        // Start pipeline
        device.startPipeline();

        while (true) {
            // Receive 'preview' frame from device 
            std::shared_ptr<dai::ImgFrame> imgFrame = preview->get<dai::ImgFrame>();

            // Show the received 'preview' frame
            cv::Mat frame(imgFrame->getHeight(), imgFrame->getWidth(), CV_8UC3, imgFrame->getData().data());
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
