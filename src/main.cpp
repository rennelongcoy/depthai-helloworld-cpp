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
    std::shared_ptr<dai::node::ColorCamera> cam_rgb = pipeline.create<dai::node::ColorCamera>();
    cam_rgb->setPreviewSize(300, 300); // to match the mobilenet-ssd input size
    cam_rgb->setInterleaved(false);

    // Define NeuralNetwork Node
    std::shared_ptr<dai::node::NeuralNetwork> detection_nn = pipeline.create<dai::node::NeuralNetwork>();
    // https://github.com/luxonis/depthai-tutorials/blob/master/1-hello-world/mobilenet-ssd/mobilenet-ssd.blob
    detection_nn->setBlobPath("/home/eli/apps/depthai-helloworld-cpp/mobilenet-ssd/mobilenet-ssd.blob");

    // For inference, connect the ColorCamera output to the NeuralNetwork input
    cam_rgb->preview.link(detection_nn->input);

    // Create XLinkOut Nodes to receive outputs from OAK
    // Connect ColorCamera Node output to XLinkOut Node input
    auto xout_rgb = pipeline.create<dai::node::XLinkOut>();
    xout_rgb->setStreamName("rgb");
    cam_rgb->preview.link(xout_rgb->input);

    // Receive neural network inference results from the NeuralNetwork Node
    // Connect NerualNetwork Node output to another XLinkOut Node input
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

        // Struct to hold NeuralNetwork outputs
        struct Detection {
            unsigned int label;
            float score;
            float x_min;
            float y_min;
            float x_max;
            float y_max;
        };

        // Variable to store ColorCamera Node output
        cv::Mat frame;

        while (true) {
            // Pop ColorCamera output frame from queue 
            std::shared_ptr<dai::ImgFrame> in_rgb = q_rgb->get<dai::ImgFrame>();
            if (in_rgb) {
                // Convert to cv::Mat data type
                frame = toMat(in_rgb->getData(), in_rgb->getWidth(), in_rgb->getHeight(), 3, 1);
            }

            // Pop NeuralNetwork output from queue
            std::vector<Detection> dets;
            auto in_nn = q_nn->get<dai::NNData>();
            std::vector<float> detData = in_nn->getFirstLayerFp16();
            if (detData.size() > 0) {
                int i = 0;
                while (detData[i*7] != -1.0f) {
                    Detection d;
                    d.label = detData[i*7 + 1]; // label
                    d.score = detData[i*7 + 2]; // score
                    d.x_min = detData[i*7 + 3]; // x_min
                    d.y_min = detData[i*7 + 4]; // y_min
                    d.x_max = detData[i*7 + 5]; // x_max
                    d.y_max = detData[i*7 + 6]; // y_max
                    ++i;
                    // Consider only outputs with confidence score > 0.8
                    if (d.score > 0.8) {
                        dets.push_back(d);
                    }
                }
            }

            // Draw bounding boxes in-frame
            for (const auto& d : dets) {
                int x1 = d.x_min * frame.cols;
                int y1 = d.y_min * frame.rows;
                int x2 = d.x_max * frame.cols;
                int y2 = d.y_max * frame.rows;

                cv::rectangle(frame, cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), cv::Scalar(0,0,255));
            }

            printf("===================== %lu detection(s) =======================\n", dets.size());
            for (unsigned det = 0; det < dets.size(); ++det) {
                printf("%5d | %6.4f | %7.4f | %7.4f | %7.4f | %7.4f\n",
                        dets[det].label,
                        dets[det].score,
                        dets[det].x_min,
                        dets[det].y_min,
                        dets[det].x_max,
                        dets[det].y_max);
            }

            // Display frame in "preview" window
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
