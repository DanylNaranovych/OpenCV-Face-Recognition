// CMakeProject1.cpp: определяет точку входа для приложения.
//

#include "CMakeProject1.h"

int main() {
	cv::cuda::setDevice(0);
	cv::VideoCapture cap(0);
	cv::Mat frame, prevFrame, diffFrame;
	cv::cuda::GpuMat Gframe, GprevFrame, GdiffFrame;
	std::string filename;
	unsigned int counter = 0;

	// Check for a camera
	if (!cap.isOpened()) {
		std::cout << "No camera exist\n";
		return -1;
	}

	// Capture the first frame
	cap >> frame;
	Gframe.upload(frame);
	cv::cuda::cvtColor(Gframe, GprevFrame, cv::COLOR_BGR2GRAY);

	while (1) { // Basic processing cycle
		cap >> frame;
		Gframe.upload(frame);
		cv::cuda::cvtColor(Gframe, Gframe, cv::COLOR_BGR2GRAY);

		// Compute the absolute difference between the current frame and the previous frame
		cv::cuda::absdiff(Gframe, GprevFrame, GdiffFrame);

		// Download the difference frame to the CPU
		GdiffFrame.download(diffFrame);

		// Threshold the difference to get the motion areas
		cv::threshold(diffFrame, diffFrame, 50, 255, cv::THRESH_BINARY);

		// Check if there is any motion
		if (cv::countNonZero(diffFrame) > 0) {
			// Display the result if motion is detected
			cv::imshow("Motion Detection", diffFrame);

			// Save the original frame to disk
			filename = std::string("D:/WorkProjects/CMakeProject1/collectedPictures/motion_detected_frame_") + std::to_string(counter++) + std::string(".jpg");
			// cv::imwrite(filename, frame);
			std::cout << "Motion detected! Frame saved to " << filename << "\n";
		}
		else {
			// Perform other actions if no motion is detected
			std::cout << "No motion detected. Start processing collected frames\n";
		}

		// Update the previous frame
		Gframe.copyTo(GprevFrame);

		if (cv::waitKey(1) == 27) { // Press esc to close the program
			std::cout << "End camera loop\n";
			return 1;
		}
	}
	return 0;
}


