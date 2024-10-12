#include "OpenCV-Face-Recognition.h"

int main() {
	cv::cuda::setDevice(0);
	cv::VideoCapture cap(0);
	cv::Mat frame, prevFrame, diffFrame;
	cv::cuda::GpuMat Gframe, GprevFrame, GdiffFrame;
	std::string filename;
	std::vector<dlib::rectangle> faces;
	unsigned short int counter = 0;

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	dlib::full_object_detection shape;
	dlib::deserialize("D:/WorkProjects/OpenCV-Face-Recognition/shape_predictor_68_face_landmarks.dat") >> pose_model;

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

			// Convert the frame to dlib's image format
			dlib::cv_image<dlib::bgr_pixel> cimg(frame);

			// Detect faces
			faces = detector(cimg);

			// Display detected faces
			for (auto& face : faces) {
				shape = pose_model(cimg, face);
				for (int i = 0; i < shape.num_parts(); ++i) {
					cv::circle(frame, cv::Point(shape.part(i).x(), shape.part(i).y()), 2, cv::Scalar(0, 0, 255), -1);
				}
				cv::rectangle(frame, cv::Point(face.left(), face.top()), cv::Point(face.right(), face.bottom()), cv::Scalar(0, 255, 0), 2);
			}

			cv::imshow("Face Detection", frame);
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
