CMAKE_MINIMUM_REQUIRED(VERSION 3.27)

PROJECT(vector LANGUAGES CXX)

SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET(CMAKE_CXX_FLAGS "-std=c++23 -Wall -Wextra -pedantic -fopenmp ")

ADD_SUBDIRECTORY(libs/Catch2/)
#ADD_SUBDIRECTORY(libs/criterion)

#INCLUDE_DIRECTORIES(libs/criterion/include)

ADD_EXECUTABLE(${CMAKE_PROJECT_NAME} "src/tests.cpp")
TARGET_LINK_LIBRARIES(${CMAKE_PROJECT_NAME} Catch2::Catch2WithMain)
