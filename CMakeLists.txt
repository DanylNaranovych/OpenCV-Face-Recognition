# CMakeList.txt: проект CMake для CMakeProject1; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.8)

# Включение горячей перезагрузки для компиляторов MSVC, если поддерживается.
if (POLICY CMP0141)
  cmake_policy (SET CMP0141 NEW)
  set (CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif ()

set (CMAKE_TOOLCHAIN_FILE "D:/vcpkg/scripts/buildsystems/vcpkg.cmake")
set (OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv4")

project ("OpenCV-Face-Recognition")

find_package (CUDA REQUIRED)
find_package (OpenCV REQUIRED)
find_package (dlib CONFIG REQUIRED)

include_directories (${CUDA_INCLUDE_DIRS})
include_directories (${OpenCV_INCLUDE_DIRS})
include_directories ("D:/vcpkg/packages/dlib_x64-windows/include")
link_directories (${DLIB_LIBRARY_DIRS})

set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_75")  # Здесь укажите архитектуру вашей видеокарты

# Добавьте источник в исполняемый файл этого проекта.
add_executable ("OpenCV-Face-Recognition" "OpenCV-Face-Recognition.cpp" "OpenCV-Face-Recognition.h")
target_link_libraries ("OpenCV-Face-Recognition" PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES} dlib::dlib)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property (TARGET OpenCV-Face-Recognition PROPERTY CXX_STANDARD 20)
endif ()

# TODO: Добавьте тесты и целевые объекты, если это необходимо.
