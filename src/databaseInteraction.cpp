#include "../inc/OpenCV-Face-Recognition.h"

void databaseInitialization() {
	try {
		// Create or open the database “records.db”
		sqlite::database db(DB_DIR);

		// Create a table if it does not exist
		db << "CREATE TABLE IF NOT EXISTS attendance ("
			"id INTEGER PRIMARY KEY AUTOINCREMENT, "
			"name TEXT, "
			"entryTime TEXT, "
			"exitTime TEXT);";
	}
	catch (sqlite::sqlite_exception& e) {
		cerr << "SQLite error: " << e.what() << endl;
	}
}

void addRecord(const string& name, const string& entryTime) {
	try {
		sqlite::database db(DB_DIR);

		if (checkOpenRecord(name)) return;

		// Add a new entry with name and entry time
		db << "INSERT INTO attendance (name, entryTime) VALUES (?, ?);"
			<< name << entryTime;
		cout << "Record added: " << name << " came in at " << entryTime << endl;
	}
	catch (sqlite::sqlite_exception& e) {
		cerr << "Error when adding a record: " << e.what() << endl;
	}
}

void addExitTimeToRecord(const string& name, const string& exitTime) {
	try {
		sqlite::database db(DB_DIR);

		int recordId = getIdOfOpenRecord(name);

		if (recordId < 0) return;

		// Update the last record with the passed name, which has not yet filled in the exitTime fieldÌî
		db << "UPDATE attendance SET exitTime = ? WHERE id = ?;"
			<< exitTime << recordId;
		cout << "Exit time updated for " << name << ": " << exitTime << endl;
	}
	catch (sqlite::sqlite_exception& e) {
		cerr << "Error when updating the exit time: " << e.what() << endl;
	}
}

bool checkOpenRecord(const string& name) {
	bool exists = false;

	try {
		sqlite::database db(DB_DIR);

		// Execute a query to find a record with an unfilled exitTime
		db << "SELECT EXISTS(SELECT 1 FROM attendance WHERE name = ? AND exitTime IS NULL LIMIT 1);"
			<< name
			>> exists;  // Assign the result to the "exists" variable (0 - no, 1 - yes)

	}
	catch (sqlite::sqlite_exception& e) {
		cerr << "Error during record verification: " << e.what() << endl;
	}

	return exists;
}

int getIdOfOpenRecord(const std::string& name) {
	int id = -1;

	try {
		sqlite::database db(DB_DIR);

		// finding the id by name
		db << "SELECT id FROM attendance WHERE name = ? AND exitTime IS NULL ORDER BY id DESC;"
			<< name
			>> id;

		std::cout << "Get the id of an open record for: " << name << "; id: " << id << std::endl;
	}
	catch (sqlite::sqlite_exception& e) {
		std::cerr << "Error when getting the id of an open record: " << e.what() << std::endl;
	}

	return id;
}
