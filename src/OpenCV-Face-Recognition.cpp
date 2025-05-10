#include "../inc/OpenCV-Face-Recognition.h"

// Definition of global variables
matrix<float, 0, 1> lastFaceDescriptor;
string lastPersonName;

string PROJECT_DIR, COLLECTED_ENTRY_DIR, COLLECTED_EXIT_DIR, IDENTIFIED_DIR, UNIDENTIFIED_DIR, DB_DIR;

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
		COLLECTED_ENTRY_DIR = currentPath + "collectedEntryPictures/";
		COLLECTED_EXIT_DIR = currentPath + "collectedExitPictures/";
		IDENTIFIED_DIR = currentPath + "identifiedPeople/";
		UNIDENTIFIED_DIR = currentPath + "unidentifiedPeople/";
		DB_DIR = currentPath + "records.db";
	}
	else {
		cerr << "Error getting the current path." << endl;
		addLog("Error getting the current path.\n");
	}
}

Config loadConfig(const std::string& filename) {
	std::ifstream file(PROJECT_DIR + filename);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open config file.");
		addLog("Could not open config file.\n");
	}

	json configJson;
	file >> configJson;

	Config config;
	config.showingFrames = configJson.value("showing_frames", false);
	config.thresholdValue = configJson.value("threshold_value", 0.15);
	config.firstCameraMainStream = configJson.value("first_camera_main_stream", "error");
	config.firstCameraSubStream = configJson.value("first_camera_sub_stream", "error");
	config.secondCameraMainStream = configJson.value("second_camera_main_stream", "error");
	config.secondCameraSubStream = configJson.value("second_camera_sub_stream", "error");

	return config;
}


// Function for a thread that captures frames from a single camera
void captureFromCamera(const std::string& cameraUrl, double thresholdValue, bool isEntry) { // isEntry: true - entry camera, false - exit camera
	cv::VideoCapture cap(cameraUrl, cv::CAP_FFMPEG);
	//cv::VideoCapture cap(0);
	cv::Mat frame, croppedFrame, prevFrame, diffFrame, grayFrame;
	string filename, filenamePart, cameraType;

	filenamePart = isEntry ? "motion_detected_frame_entry" : "motion_detected_frame_exit";
	cameraType = isEntry ? "entry" : "exit";
	short int counter = getLastFrameNumber(isEntry ? COLLECTED_ENTRY_DIR : COLLECTED_EXIT_DIR, filenamePart);

	// Check for a camera
	if (!cap.isOpened()) {
		cerr << "No " << cameraType << " camera for capture exist" << endl;
		addLog("No " + cameraType + " camera for capture exist.\n");
		return;
	}

	// Capture the first frame
	cap >> frame;

	int topCrop = frame.rows / 10; // 10% of frame height

	// Crop the frame
	cv::Rect cropRect(frame.cols / 3, topCrop, frame.cols * 2 / 3, frame.rows - topCrop);
	croppedFrame = frame(cropRect);

	cv::cvtColor(croppedFrame, prevFrame, cv::COLOR_BGR2GRAY);

	// Basic processing cycle
	while (true) {
		cap >> frame;

		// Crop the left third of the frame
		croppedFrame = frame(cropRect);
		cv::cvtColor(croppedFrame, grayFrame, cv::COLOR_BGR2GRAY);

		// Compute the absolute difference between the current frame and the previous frame
		cv::absdiff(grayFrame, prevFrame, diffFrame);

		// Threshold the difference to get the motion areas
		cv::threshold(diffFrame, diffFrame, 70, 255, cv::THRESH_BINARY);

		// Calculate percentage of difference
		int totalPixels = diffFrame.rows * diffFrame.cols;
		int motionPixels = cv::countNonZero(diffFrame);
		double differencePercentage = (static_cast<double>(motionPixels) / totalPixels) * 100.0;

		// Check if there is any motion
		if (differencePercentage > thresholdValue) {
			// Save the original frame to disk
			counter = getLastFrameNumber(isEntry ? COLLECTED_ENTRY_DIR : COLLECTED_EXIT_DIR, filenamePart);
			filename = (isEntry ? COLLECTED_ENTRY_DIR : COLLECTED_EXIT_DIR) + filenamePart + "_" + to_string(++counter) + ".jpg";
			cv::imwrite(filename, grayFrame);
		}

		// Update the previous frame
		grayFrame.copyTo(prevFrame);

		// Press esc to close the program
		if (GetAsyncKeyState(VK_ESCAPE)) {
			cerr << "End " << cameraType << " camera loop for capture" << endl;
			return;
		}
	}
}

// Function for a thread that captures frames from a single camera
void displayFromCamera(const std::string& cameraUrl, bool isEntry) { // isEntry: true - entry camera, false - exit camera
	cv::VideoCapture cap(cameraUrl, cv::CAP_FFMPEG);
	//cv::VideoCapture cap(0);
	cv::Mat frame;
	string cameraType;

	cameraType = isEntry ? "entry" : "exit";

	// Check for a camera
	if (!cap.isOpened()) {
		cerr << "No " << cameraType << " camera for display exist" << endl;
		addLog("No " + cameraType + " camera for display exist.\n");
		return;
	}

	// Basic processing cycle
	while (true) {
		cap >> frame;

		// Display the result
		cv::imshow("Camera: " + cameraType, frame);

		// Press esc to close the program
		if (cv::waitKey(1) == 27) {
			cerr << "End " << cameraType << " camera loop for display" << endl;
			addLog("End " + cameraType + " camera loop for display.\n");
			return;
		}
	}
}

int main() {
	cv::cuda::setDevice(0);

	setPaths();
	databaseInitialization();

	Config config = loadConfig("config.json");

	// Creating thread for each camera
	thread camera1Thread(captureFromCamera, config.firstCameraMainStream, config.thresholdValue, true);	// entry camera
	thread camera2Thread(captureFromCamera, config.secondCameraMainStream, config.thresholdValue, false);// exit camera

	// Creating thread for processing collected frames
	thread processingEntryFramesThread(processCollectedPictures, true);
	thread processingExitFramesThread(processCollectedPictures, false);

	// Check whether frames should be displayed on the screen
	if (config.showingFrames) {
		thread camera1DisplayThread(displayFromCamera, config.firstCameraSubStream, true);
		thread camera2DisplayThread(displayFromCamera, config.secondCameraSubStream, false);
		camera1DisplayThread.join();
		camera2DisplayThread.join();
	}

	// Waiting for threads to complete
	camera1Thread.join();
	camera2Thread.join();
	processingEntryFramesThread.join();
	processingExitFramesThread.join();

	return 0;
}
