#include "../inc/OpenCV-Face-Recognition.h"

cv::Mat sharpeningKernel = (cv::Mat_<float>(3, 3) <<
	0, -1, 0,
	-1, 5, -1,
	0, -1, 0);

std::string getFileCreationTime(const string& filePath) {
	// Get the time of the last modification of the file
	auto ftime = fs::last_write_time(filePath);

	// Convert file_time_type to system_clock::time_point
	auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
		ftime - decltype(ftime)::clock::now() + std::chrono::system_clock::now()
	);

	// Convert to time_t
	std::time_t ctime = std::chrono::system_clock::to_time_t(sctp);

	// Convert to a string using stringstream
	std::stringstream ss;
	ss << std::put_time(std::localtime(&ctime), "%Y-%m-%d %H:%M:%S");

	// Return a string representation of the time
	return ss.str();
}

void processCollectedPictures(bool isEntry) {
	bool knownPerson = false;
	short int index, collectedIndex;
	size_t identifiedPersonNameSize;
	string personName, identifiedPersonName, filePath;
	cv::Mat img, knownImg, grayFace, sharpenedFace, filteredFace, enhancedFace;
	matrix<rgb_pixel> face_chip;
	Timer timer;

	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	anet_type face_recognizer;
	string pose_model_path = PROJECT_DIR + "models/shape_predictor_68_face_landmarks.dat";
	string face_recognizer_path = PROJECT_DIR + "models/dlib_face_recognition_resnet_model_v1.dat";
	deserialize(pose_model_path) >> pose_model;
	deserialize(face_recognizer_path) >> face_recognizer;

	cout << "Processing of " << (isEntry ? "entry" : "exit") << " collected frames has started" << endl;

	while (true) {
		// Press esc to close the program
		if (GetAsyncKeyState(VK_ESCAPE)) {
			cout << "End processing of " << (isEntry ? "entry" : "exit") << " collected frames loop" << endl;
			return;
		}

		if (timer.elapsed() > 5) {

			if (isEntry) {
				collectedIndex = getLastFrameNumber(COLLECTED_ENTRY_DIR, "motion_detected_frame_entry");

				if (collectedIndex >= 0) {
					filePath = COLLECTED_ENTRY_DIR + string("motion_detected_frame_entry_" + to_string(collectedIndex) + ".jpg");
				}
				else {
					cout << "No frames from entry camera detected. Wait 5 second before another check" << endl;
					timer.reset();
					continue;
				}
			}
			else if (!isEntry) {
				collectedIndex = getLastFrameNumber(COLLECTED_EXIT_DIR, "motion_detected_frame_exit");

				if (collectedIndex >= 0) {
					filePath = COLLECTED_EXIT_DIR + string("motion_detected_frame_exit_" + to_string(collectedIndex) + ".jpg");
				}
				else {
					cout << "No frames from exit camera detected. Wait 5 second before another check" << endl;
					timer.reset();
					continue;
				}
			}
		}
		else {
			continue;
		}

		img = cv::imread(filePath);

		if (img.empty()) {
			cerr << "File is empty or unreachable: " << filePath << endl;
			addLog("File is empty or unreachable: " + filePath + "\n");
			continue;
		}

		cv_image<bgr_pixel> cimg(img);
		std::vector<rectangle> faces = detector(cimg);

		// Checking the presence of faces on the frame
		if (!faces.empty()) {
			// Loop through all faces in the frame
			for (const auto& face : faces) {
				personName = "unknownPerson";

				// Retrieve the descriptor for the face in the current image
				full_object_detection shape = pose_model(cimg, face);

				// If the model could not find 68 points correctly, we ignore this face
				if (shape.num_parts() != 68) {
					continue;
				}

				extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);
				matrix<float, 0, 1> faceDescriptor = face_recognizer(face_chip);

				//Check if the current face matches the previously recognized face
				if (!lastPersonName.empty()) {
					double distance = length(faceDescriptor - lastFaceDescriptor);

					// Threshold for identifying similar individuals
					if (distance < 0.6) {
						personName = lastPersonName;
						knownPerson = true;
					}
				}

				// If the person has not been identified in the past condition,
				// then we start identification relative to all known people
				if (!knownPerson) {
					// Search the folder of identified people
					for (const auto& personEntry : fs::directory_iterator(IDENTIFIED_DIR)) {
						identifiedPersonName = personEntry.path().filename().string();
						identifiedPersonNameSize = identifiedPersonName.size();
						if (identifiedPersonName.substr(identifiedPersonNameSize - 4) != ".jpg")  continue;

						knownImg = cv::imread(IDENTIFIED_DIR + identifiedPersonName);

						identifiedPersonName.erase(identifiedPersonNameSize - 4);

						cv_image<bgr_pixel> knownCimg(knownImg);
						std::vector<rectangle> knownFaces = detector(knownCimg);

						if (!knownFaces.empty()) {
							full_object_detection knownShape = pose_model(knownCimg, knownFaces[0]);
							extract_image_chip(knownCimg, get_face_chip_details(knownShape, 150, 0.25), face_chip);
							matrix<float, 0, 1> knownFaceDescriptor = face_recognizer(face_chip);

							// Comparing face descriptors using Euclidean distance
							double distance = length(faceDescriptor - knownFaceDescriptor);
							// Threshold for identifying similar individuals
							if (distance < 0.6) {
								knownPerson = true;
								personName = identifiedPersonName;
								lastFaceDescriptor = knownFaceDescriptor; // Сохраняем последний дескриптор
								lastPersonName = personName; // Сохраняем имя последнего человека
								break;
							}
						}
						if (knownPerson) break;
					}
				}
				if (knownPerson) {
					isEntry ? addRecord(personName, getFileCreationTime(filePath)) : addExitTimeToRecord(personName, getFileCreationTime(filePath));
				}
				else {
					index = getLastFrameNumber(UNIDENTIFIED_DIR, personName);
					string destination = UNIDENTIFIED_DIR + personName + "_" + to_string(++index) + ".jpg";

					// The enlargement of the rectangle around the face
					int paddingX = static_cast<int>(face.width() * 0.2);  // 20% from the width of the face
					int paddingY = static_cast<int>(face.height() * 0.2); // 20% from the height of the face

					// Coordinates and dimensions of the rectangle taking into account the image boundaries
					int x = std::max(0, static_cast<int>(face.left()) - paddingX);
					int y = std::max(0, static_cast<int>(face.top()) - paddingY);
					int width = std::min(static_cast<int>(face.width() + 2 * paddingX), img.cols - x);
					int height = std::min(static_cast<int>(face.height() + 2 * paddingY), img.rows - y);

					cv::Rect expandedFaceRect(x, y, width, height);
					cv::Mat croppedFace = img(expandedFaceRect).clone();

					cv::cvtColor(croppedFace, croppedFace, cv::COLOR_BGR2GRAY);

					// Noise removal using OpenCV Denoise functions
					cv::fastNlMeansDenoising(croppedFace, filteredFace, 3, 7, 21);

					// Contrast enhancement with CLAHE
					cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4.0);
					clahe->apply(filteredFace, enhancedFace);

					// Applying a sharpening filter
					cv::filter2D(enhancedFace, sharpenedFace, -1, sharpeningKernel);

					// Saving a cropped face image
					cv::imwrite(destination, sharpenedFace);
				}
			}
		}

		fs::remove(filePath);
	}
}
