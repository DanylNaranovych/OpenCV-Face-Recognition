#include "OpenCV-Face-Recognition.h"

// Timer. It is necessary for counting down five seconds from the moment
// of saving the last frame before starting the processing of saved photos
class Timer {
public:
	Timer() : startTime(chrono::steady_clock::now()) {}

	// Timer reset
	void reset() {
		startTime = chrono::steady_clock::now();
	}

	// Returns the time in seconds since the last reset or startup
	double elapsed() const {
		auto currentTime = chrono::steady_clock::now();
		chrono::duration<double> elapsedTime = currentTime - startTime;
		return elapsedTime.count();
	}

private:
	chrono::steady_clock::time_point startTime;
};

// Function for getting counter
int getLastFrameNumber(const string& directoryPath, const string& patternPart) {
	regex filePattern(format(R"({}_(\d+))", patternPart));
	string filename;
	smatch match;
	int lastFrameNumber = -1;

	for (const auto& entry : fs::directory_iterator(directoryPath)) {
		if (entry.is_regular_file()) {
			filename = entry.path().filename().string();
			if (regex_search(filename, match, filePattern)) {
				int frameNumber = stoi(match[1].str());
				if (frameNumber > lastFrameNumber) {
					lastFrameNumber = frameNumber;
				}
			}
		}
	}

	return lastFrameNumber;
}

void processCollectedPictures(frontal_face_detector& detector, shape_predictor& pose_model, anet_type& face_recognizer, int imgIndex) {
	bool knownPerson = false;
	short int index;
	string personName, identifiedPersonName;
	string filePath = COLLECTED_DIR + string("motion_detected_frame_" + to_string(imgIndex) + ".jpg");
	cv::Mat img = cv::imread(filePath);

	if (img.empty()) return;

	cv_image<bgr_pixel> cimg(img);
	std::vector<rectangle> faces = detector(cimg);

	// Checking the presence of faces on the frame
	if (!faces.empty()) {
		personName = "unknownPerson";

		// Извлекаем дескриптор для лица на текущем изображении
		matrix<rgb_pixel> face_chip;
		full_object_detection shape = pose_model(cimg, faces[0]);
		extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);
		matrix<float, 0, 1> face_descriptor = face_recognizer(face_chip);

		for (const auto& personEntry : fs::directory_iterator(IDENTIFIED_DIR)) {
			identifiedPersonName = personEntry.path().filename().string();

			for (const auto& imgEntry : fs::directory_iterator(IDENTIFIED_DIR + identifiedPersonName)) {
				cv::Mat knownImg = cv::imread(imgEntry.path().string());
				cv_image<bgr_pixel> knownCimg(knownImg);
				std::vector<rectangle> knownFaces = detector(knownCimg);

				if (!knownFaces.empty()) {
					full_object_detection knownShape = pose_model(knownCimg, knownFaces[0]);
					extract_image_chip(knownCimg, get_face_chip_details(knownShape, 150, 0.25), face_chip);
					matrix<float, 0, 1> known_face_descriptor = face_recognizer(face_chip);

					// Сравниваем дескрипторы лиц с помощью эвклидовой дистанции
					double distance = length(face_descriptor - known_face_descriptor);
					if (distance < 0.6) { // Пороговое значение для определения похожих лиц
						knownPerson = true;
						personName = identifiedPersonName;
						break;
					}
				}
				if (knownPerson) break;
			}
			if (knownPerson) break;
		}

		if (knownPerson) {
			index = getLastFrameNumber(IDENTIFIED_DIR + personName + "/", personName);
			if (index < 49) {
				string destination = IDENTIFIED_DIR + personName + "/" + personName + "_" + to_string(++index) + ".jpg";
				fs::rename(filePath, destination);
			}
			else {
				fs::remove(filePath);
			}
		}
		else {
			index = getLastFrameNumber(UNIDENTIFIED_DIR, personName);
			string destination = UNIDENTIFIED_DIR + personName + "_" + to_string(++index) + ".jpg";
			fs::rename(filePath, destination);
		}
	}
	else {
		cout << "No faces detected in the file: " << filePath << endl;
		fs::remove(filePath);
	}
}

int main() {
	cv::cuda::setDevice(0);
	cv::VideoCapture cap(0);
	cv::Mat frame, prevFrame, diffFrame;
	cv::cuda::GpuMat Gframe, GprevFrame, GdiffFrame;
	string filename;
	std::vector<rectangle> faces;
	Timer timer;
	short int counter = getLastFrameNumber(COLLECTED_DIR, "motion_detected_frame");

	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	anet_type face_recognizer;
	deserialize(PROJECT_DIR + string("shape_predictor_68_face_landmarks.dat")) >> pose_model;
	deserialize(PROJECT_DIR + string("dlib_face_recognition_resnet_model_v1.dat")) >> face_recognizer;


	// Check for a camera
	if (!cap.isOpened()) {
		cout << "No camera exist\n";
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

		cout << "counter: " << counter << " time passed: " << timer.elapsed() << endl;

		// Check if there is any motion
		if (cv::countNonZero(diffFrame) > 0) {
			// Save the original frame to disk
			filename = COLLECTED_DIR + string("motion_detected_frame_") + to_string(++counter) + ".jpg";
			cv::imwrite(filename, frame);
			cout << "Motion detected! Frame saved to " << filename << "\n";
			timer.reset();
		}
		// Check for frames in folder and whether 5 seconds have elapsed since the last motion
		else if (counter >= 0 && timer.elapsed() >= 5) {
			// Perform face detection if no motion is detected
			cout << "No motion detected. Start processing collected frames\n";
			processCollectedPictures(detector, pose_model, face_recognizer, counter);

			counter = getLastFrameNumber(COLLECTED_DIR, "motion_detected_frame");
		}

		// Update the previous frame
		Gframe.copyTo(GprevFrame);

		// Press esc to close the program
		if (cv::waitKey(1) == 27) {
			cout << "End camera loop\n";
			return 1;
		}
	}
	return 0;
}
