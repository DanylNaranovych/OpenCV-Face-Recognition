#include "OpenCV-Face-Recognition.h"

// Timer. It is necessary for counting down five seconds from the moment
// of saving the last frame before starting the processing of saved photos
class Timer {
public:
	Timer() : startTime(std::chrono::steady_clock::now()) {}

	// Timer reset
	void reset() {
		startTime = std::chrono::steady_clock::now();
	}

	// Returns the time in seconds since the last reset or startup
	double elapsed() const {
		auto currentTime = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsedTime = currentTime - startTime;
		return elapsedTime.count();
	}

private:
	std::chrono::steady_clock::time_point startTime;
};

// Function for getting counter
int getLastFrameNumber(const std::string& directoryPath, const std::string& patternPart) {
	std::regex filePattern(std::format(R"({}_(\d+))", patternPart));
	std::string filename;
	std::smatch match;
	int lastFrameNumber = -1;

	for (const auto& entry : fs::directory_iterator(directoryPath)) {
		if (entry.is_regular_file()) {
			filename = entry.path().filename().string();
			if (std::regex_search(filename, match, filePattern)) {
				int frameNumber = std::stoi(match[1].str());
				if (frameNumber > lastFrameNumber) {
					lastFrameNumber = frameNumber;
				}
			}
		}
	}

	return lastFrameNumber;
}

void processCollectedPictures(dlib::frontal_face_detector& detector, dlib::shape_predictor& pose_model, int imgIndex) {
	bool knownPerson = false;
	std::string personName, identifiedPersonName;
	std::string filePath = COLLECTED_DIR + std::string("motion_detected_frame_" + std::to_string(imgIndex) + ".jpg");
	cv::Mat img = cv::imread(filePath);

	if (img.empty()) return;

	dlib::cv_image<dlib::bgr_pixel> cimg(img);
	std::vector<dlib::rectangle> faces = detector(cimg);

	// Checking the presence of faces on the frame
	if (!faces.empty()) {
		personName = "unknownPerson";

		for (const auto& personEntry : fs::directory_iterator(IDENTIFIED_DIR)) {
			identifiedPersonName = personEntry.path().filename().string();

			for (const auto& imgEntry : fs::directory_iterator(IDENTIFIED_DIR + identifiedPersonName)) {
				cv::Mat knownImg = cv::imread(imgEntry.path().string());
				dlib::cv_image<dlib::bgr_pixel> knownCimg(knownImg);
				std::vector<dlib::rectangle> knownFaces = detector(knownCimg);

				if (!knownFaces.empty() && knownFaces.size() == faces.size()) {
					dlib::full_object_detection knownShape = pose_model(knownCimg, knownFaces[0]);
					dlib::full_object_detection newShape = pose_model(cimg, faces[0]);

					// Простое сравнение по количеству точек на лице
					if (knownShape.num_parts() == newShape.num_parts()) {
						knownPerson = true;
						personName = identifiedPersonName;
						break;
					}
				}
			}
			if (knownPerson) break;
		}

		if (knownPerson) {
			std::string destination = IDENTIFIED_DIR + personName + "/" + personName + "_" + std::to_string(getLastFrameNumber(IDENTIFIED_DIR + personName + "/", personName) + 1) + ".jpg";
			fs::rename(filePath, destination);
		}
		else {
			std::string destination = UNIDENTIFIED_DIR + personName + "_" + std::to_string(getLastFrameNumber(UNIDENTIFIED_DIR, personName) + 1) + ".jpg";
			fs::rename(filePath, destination);
		}
	}
	else {
		std::cout << "No faces detected in the file: " << filePath << std::endl;
		fs::remove(filePath);
	}
}

int main() {
	cv::cuda::setDevice(0);
	cv::VideoCapture cap(0);
	cv::Mat frame, prevFrame, diffFrame;
	cv::cuda::GpuMat Gframe, GprevFrame, GdiffFrame;
	std::string filename;
	std::vector<dlib::rectangle> faces;
	Timer timer;
	short int counter = getLastFrameNumber(COLLECTED_DIR, "motion_detected_frame");

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	dlib::full_object_detection shape;
	dlib::deserialize(PROJECT_DIR + std::string("shape_predictor_68_face_landmarks.dat")) >> pose_model;

	// Check for a camera
	if (!cap.isOpened()) {
		std::cout << "No camera exist\n";
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
		cv::threshold(diffFrame, diffFrame, 50, 255, cv::THRESH_BINARY);

		// Display the result
		cv::imshow("Motion Detection", diffFrame);

		std::cout << "counter: " << counter << " time passed: " << timer.elapsed() << std::endl;

		// Check if there is any motion
		if (cv::countNonZero(diffFrame) > 0) {
			// Save the original frame to disk
			filename = COLLECTED_DIR + std::string("motion_detected_frame_") + std::to_string(++counter) + ".jpg";
			cv::imwrite(filename, frame);
			std::cout << "Motion detected! Frame saved to " << filename << "\n";
			timer.reset();
		}
		// Check for frames in folder and whether 5 seconds have elapsed since the last motion
		else if (counter >= 0 && timer.elapsed() >= 5) {
			// Perform face detection if no motion is detected
			std::cout << "No motion detected. Start processing collected frames\n";
			processCollectedPictures(detector, pose_model, counter);

			counter = getLastFrameNumber(COLLECTED_DIR, "motion_detected_frame");
		}

		// Update the previous frame
		Gframe.copyTo(GprevFrame);

		// Press esc to close the program
		if (cv::waitKey(1) == 27) {
			std::cout << "End camera loop\n";
			return 1;
		}
	}
	return 0;
}
