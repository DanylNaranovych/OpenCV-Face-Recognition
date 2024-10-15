#include "../inc/OpenCV-Face-Recognition.h"

void processCollectedPictures(frontal_face_detector& detector, shape_predictor& pose_model, anet_type& face_recognizer, int imgIndex) {
	bool knownPerson = false;
	short int index, identifiedPersonNameSize;
	string personName, identifiedPersonName;
	string filePath = COLLECTED_DIR + string("motion_detected_frame_" + to_string(imgIndex) + ".jpg");
	cv::Mat img = cv::imread(filePath);
	cv::Mat knownImg;
	matrix<rgb_pixel> face_chip;

	if (img.empty()) return;

	cv_image<bgr_pixel> cimg(img);
	std::vector<rectangle> faces = detector(cimg);

	// Checking the presence of faces on the frame
	if (!faces.empty()) {
		personName = "unknownPerson";

		// Loop through all faces in the frame
		for (const auto& face : faces) {
			// Retrieve the descriptor for the face in the current image
			full_object_detection shape = pose_model(cimg, face);
			extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);
			matrix<float, 0, 1> faceDescriptor = face_recognizer(face_chip);

			//Check if the current face matches the previously recognized face
			if (!lastPersonName.empty()) {
				double distance = length(faceDescriptor - lastFaceDescriptor);
				if (distance < 0.6) { // Threshold for identifying similar individuals
					cout << "The same person detected: " << lastPersonName << endl;
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
						if (distance < 0.6) { // Threshold for identifying similar individuals
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
		}

		if (knownPerson) {
			index = getLastFrameNumber(IDENTIFIED_DIR + personName + "/", personName);
			if (index < 49) {
				string destination = IDENTIFIED_DIR + personName + "/" + personName + "_" + to_string(++index) + ".jpg";
				cout << "The person is identified, preservation in: " << destination << endl;
				fs::rename(filePath, destination);
			}
			else {
				cout << "The person is identified, but enough photos have been accumulated, photo deletion at path: " << filePath << endl;
				fs::remove(filePath);
			}
		}
		else {
			index = getLastFrameNumber(UNIDENTIFIED_DIR, personName);
			string destination = UNIDENTIFIED_DIR + personName + "_" + to_string(++index) + ".jpg";
			cout << "The person is unidentified, preservation in: " << destination << endl;
			fs::rename(filePath, destination);
		}
	}
	else {
		cout << "No faces detected in the file: " << filePath << endl;
		fs::remove(filePath);
	}
}
