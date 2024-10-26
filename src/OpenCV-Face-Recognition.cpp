#include "../inc/OpenCV-Face-Recognition.h"

// Definition of global variables
matrix<float, 0, 1> lastFaceDescriptor;
string lastPersonName;

string PROJECT_DIR, COLLECTED_DIR, IDENTIFIED_DIR, UNIDENTIFIED_DIR, DB_DIR;

// Function for setting paths
void setPaths() {
	char cwd[MAX_PATH];

	// Getting the current working path
	if (GetCurrentDirectoryA(MAX_PATH, cwd)) {
		std::string currentPath = cwd;

		// Find the position of the last slash three times to get the root folder of the project
		for (int i = 0; i < 3; ++i) {
			size_t pos = currentPath.find_last_of("\\/");
			if (pos != string::npos) {
				currentPath = currentPath.substr(0, pos);  // Trim the string to the last found slash
			}
		}
		currentPath += "/";

		PROJECT_DIR = currentPath;
		COLLECTED_DIR = currentPath + "collectedPictures/";
		IDENTIFIED_DIR = currentPath + "identifiedPeople/";
		UNIDENTIFIED_DIR = currentPath + "unidentifiedPeople/";
		DB_DIR = currentPath + "records.db";
	}
	else {
		cerr << "Error getting the current path." << endl;
	}
}

// Function for a thread that captures frames from a single camera
void captureFromCamera(const std::string& cameraUrl, bool isEntry) { // isEntry: true - entry camera, false - exit camera
	//cv::VideoCapture cap(cameraUrl, cv::CAP_FFMPEG);
	cv::VideoCapture cap(0);
	cv::Mat frame, croppedFrame, prevFrame, diffFrame, grayFrame;
	string filename, filenamePart, cameraType;

	filenamePart = isEntry ? "motion_detected_frame_entry" : "motion_detected_frame_exit";
	cameraType = isEntry ? "entry" : "exit";
	short int counter = getLastFrameNumber(COLLECTED_DIR, filenamePart);

	// Check for a camera
	if (!cap.isOpened()) {
		cout << "No camera " << cameraType << " exist" << endl;
		return;
	}

	// Capture the first frame
	cap >> frame;

	// Crop the left third of the frame
	cv::Rect cropRect(frame.cols / 3, 0, frame.cols * 2 / 3, frame.rows);
	croppedFrame = frame(cropRect);

	cv::cvtColor(croppedFrame, prevFrame, cv::COLOR_BGR2GRAY);

	// Basic processing cycle
	while (true) {
		cap >> frame;
		cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

		// Crop the left third of the frame
		croppedFrame = frame(cropRect);
		cv::cvtColor(croppedFrame, grayFrame, cv::COLOR_BGR2GRAY);

		// Compute the absolute difference between the current frame and the previous frame
		cv::absdiff(grayFrame, prevFrame, diffFrame);

		// Threshold the difference to get the motion areas
		cv::threshold(diffFrame, diffFrame, 70, 255, cv::THRESH_BINARY);

		// Display the result
		cv::imshow("Camera: " + cameraType, croppedFrame);

		// Check if there is any motion
		if (cv::countNonZero(diffFrame) > 0) {
			// Save the original frame to disk
			counter = getLastFrameNumber(COLLECTED_DIR, filenamePart);
			filename = COLLECTED_DIR + filenamePart + "_" + to_string(++counter) + ".jpg";
			cv::imwrite(filename, croppedFrame);
			cout << "Motion detected! Frame saved to " << filename << endl;
		}

		// Update the previous frame
		grayFrame.copyTo(prevFrame);

		// Press esc to close the program
		if (cv::waitKey(1) == 27) {
			cout << "End " << cameraType << " camera loop" << endl;
			return;
		}
	}
}

int main() {
	cv::cuda::setDevice(0);

	setPaths();
	databaseInitialization();

	// Creating two threads for each camera
	thread camera1Thread(captureFromCamera, "rtsp://admin:YT1771Q1@192.168.1.131/Preview_01_main", true);	// entry camera
	//thread camera2Thread(captureFromCamera, "rtsp://admin:YT1771Q1@192.168.1.115/Preview_01_main", false);	// exit camera

	// Creating thread for processing collected frames
	thread processingFramesThread(processCollectedPictures);

	// Waiting for threads to complete
	camera1Thread.join();
	//camera2Thread.join();
	processingFramesThread.join();

	return 0;
}
