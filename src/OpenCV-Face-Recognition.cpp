#include "../inc/OpenCV-Face-Recognition.h"

// Definition of global variables
matrix<float, 0, 1> lastFaceDescriptor;
string lastPersonName;

int main() {
	cv::cuda::setDevice(0);
	//cv::VideoCapture cap("rtsp://admin:YT90W2!@190.120.1.21/Preview_01_sub");
	cv::VideoCapture cap(0);
	cv::Mat frame, prevFrame, diffFrame;
	cv::cuda::GpuMat Gframe, GprevFrame, GdiffFrame;
	string filename;
	std::vector<rectangle> faces;
	Timer timer;
	short int counter = getLastFrameNumber(COLLECTED_DIR, "motion_detected_frame");

	databaseInitialization();

	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	anet_type face_recognizer;
	deserialize(PROJECT_DIR + string("models/shape_predictor_68_face_landmarks.dat")) >> pose_model;
	deserialize(PROJECT_DIR + string("models/dlib_face_recognition_resnet_model_v1.dat")) >> face_recognizer;


	// Check for a camera
	if (!cap.isOpened()) {
		cout << "No camera exist" << endl;
		return -1;
	}

	// Capture the first frame
	cap >> frame;
	Gframe.upload(frame);
	cv::cuda::cvtColor(Gframe, GprevFrame, cv::COLOR_BGR2GRAY);

	// Basic processing cycle
	while (1) {
		cap >> frame;
		Gframe.upload(frame);
		cv::cuda::cvtColor(Gframe, Gframe, cv::COLOR_BGR2GRAY);

		// Compute the absolute difference between the current frame and the previous frame
		cv::cuda::absdiff(Gframe, GprevFrame, GdiffFrame);

		// Download the difference frame to the CPU
		GdiffFrame.download(diffFrame);

		// Threshold the difference to get the motion areas
		cv::threshold(diffFrame, diffFrame, 70, 255, cv::THRESH_BINARY);

		// Display the result
		cv::imshow("Motion Detection", diffFrame);

		cout << "counter: " << counter << " time passed: " << timer.elapsed() << endl;

		// Check if there is any motion
		if (cv::countNonZero(diffFrame) > 0) {
			// Save the original frame to disk
			filename = COLLECTED_DIR + string("motion_detected_frame_") + to_string(++counter) + ".jpg";
			cv::imwrite(filename, frame);
			cout << "Motion detected! Frame saved to " << filename << endl;
			timer.reset();
		}
		// Check for frames in folder and whether 5 seconds have elapsed since the last motion
		else if (counter >= 0 && timer.elapsed() >= 5) {
			// Perform face detection if no motion is detected
			cout << "No motion detected. Start processing collected frame: " << counter << endl;
			processCollectedPictures(detector, pose_model, face_recognizer, counter);

			counter = getLastFrameNumber(COLLECTED_DIR, "motion_detected_frame");
		}

		// Update the previous frame
		Gframe.copyTo(GprevFrame);

		// Press esc to close the program
		if (cv::waitKey(1) == 27) {
			cout << "End camera loop" << endl;
			return 1;
		}
	}
	return 0;
}
