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
    cam_rgb->setInterleaved(false);

    // Define NeuralNetwork Node
    auto detection_nn = pipeline.create<dai::node::NeuralNetwork>();
    //detection_nn->setBlobPath("/home/eli/apps/depthai-helloworld-cpp/model/mobilenet-ssd.blob");
    detection_nn->setBlobPath("/home/eli/apps/depthai-helloworld-cpp/model/mobilenet-ssd_openvino_2021.2_6shave.blob");

    // For inference, connect the ColorCamera output to the NeuralNetwork input
    cam_rgb->preview.link(detection_nn->input);

    // Create XLinkOut Nodes to receive outputs from OAK
    // Receive color camera frames from the ColorCamera Node
    auto xout_rgb = pipeline.create<dai::node::XLinkOut>();
    xout_rgb->setStreamName("rgb");
    cam_rgb->preview.link(xout_rgb->input);

    // Receive neural network inference results from the NeuralNetwork Node
    auto xout_nn = pipeline.create<dai::node::XLinkOut>();
    xout_nn->setStreamName("nn");
    detection_nn->out.link(xout_nn->input);

    try {
        // Try connecting to device
        dai::Device device(pipeline);

        // Get output queues
        auto q_rgb = device.getOutputQueue("rgb");
        auto q_nn = device.getOutputQueue("nn");

        // Start pipeline
        device.startPipeline();

        // Variables to store Node outputs
        cv::Mat frame;
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
            auto in_nn = q_nn->get<dai::NNData>();

            // Show the received 'preview' frame
            if (in_rgb) {
                //printf("Frame - w: %d, h: %d\n", in_rgb->getWidth(), in_rgb->getHeight());
                // Convert to cv::Mat data type
                frame = toMat(in_rgb->getData(), in_rgb->getWidth(), in_rgb->getHeight(), 3, 1);
                //frame = cv::Mat(in_rgb->getHeight(), in_rgb->getWidth(), CV_8UC3, in_rgb->getData().data());
            }
            
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
