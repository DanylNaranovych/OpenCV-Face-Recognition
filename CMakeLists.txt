# CMakeList.txt: проект CMake для CMakeProject1; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.8)

# Включение горячей перезагрузки для компиляторов MSVC, если поддерживается.
if (POLICY CMP0141)
  cmake_policy (SET CMP0141 NEW)
  set (CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif ()

set (CMAKE_TOOLCHAIN_FILE "D:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
# set (OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv4")

project ("OpenCV-Face-Recognition")

find_package (SQLite3 REQUIRED)
find_package (CUDA REQUIRED)
find_package (OpenCV REQUIRED)
find_package (dlib CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

find_path(SQLITE_MODERN_CPP_INCLUDE_DIRS "sqlite_modern_cpp.h")

# set (DLIB_USE_CUDA True)

include_directories (${CUDA_INCLUDE_DIRS})
include_directories (${OpenCV_INCLUDE_DIRS})
include_directories ("D:/vcpkg/packages/dlib_x64-windows/include")
link_directories (${DLIB_LIBRARY_DIRS})

set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_75")  # Здесь укажите архитектуру вашей видеокарты

# Поиск всех исходных файлов в папке src
file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# Поиск всех заголовочных файлов в папке inc (опционально, если нужно указать заголовочные файлы)
file(GLOB_RECURSE HEADER_FILES ${CMAKE_SOURCE_DIR}/inc/*.h)

# Добавьте источник в исполняемый файл этого проекта.
add_executable ("OpenCV-Face-Recognition" ${HEADER_FILES} ${SRC_FILES})
target_link_libraries ("OpenCV-Face-Recognition" PRIVATE SQLite::SQLite3)
target_link_libraries ("OpenCV-Face-Recognition" PRIVATE ${SQLITE_MODERN_CPP_INCLUDE_DIRS})
target_link_libraries ("OpenCV-Face-Recognition" PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries ("OpenCV-Face-Recognition" PRIVATE ${CUDA_LIBS} ${OpenCV_LIBS} dlib::dlib)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property (TARGET OpenCV-Face-Recognition PROPERTY CXX_STANDARD 20)
endif ()

# TODO: Добавьте тесты и целевые объекты, если это необходимо.
