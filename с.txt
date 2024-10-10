# CMakeList.txt: проект CMake для CMakeProject1; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.8)

# Включение горячей перезагрузки для компиляторов MSVC, если поддерживается.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("CMakeProject1")

get_directory_property(has_parent PARENT_DIRECTORY)
if(NOT has_parent)
   set(DLIB_IN_PROJECT_BUILD false)
endif()

add_subdirectory(../dlib dlib_build)

# set("dlib_DIR" "C:/Users/Danyl/Downloads/dlib/build/dlib/config")

find_package(CUDA REQUIRED)
find_package (OpenCV REQUIRED)
# find_package (dlib REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories (${OpenCV_INCLUDE_DIRS})
# include_directories ("C:/Users/Danyl/Downloads/dlib/source")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_75")  # Здесь укажите архитектуру вашей видеокарты

# Добавьте источник в исполняемый файл этого проекта.
add_executable ("CMakeProject1" "CMakeProject1.cpp" "CMakeProject1.h")
target_link_libraries ("CMakeProject1" ${OpenCV_LIBS} ${CUDA_LIBRARIES} dlib::dlib)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CMakeProject1 PROPERTY CXX_STANDARD 20)
endif()

# TODO: Добавьте тесты и целевые объекты, если это необходимо.
