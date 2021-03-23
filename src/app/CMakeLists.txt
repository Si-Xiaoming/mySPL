file(GLOB SOURCES "*.h" "*.cpp" "*.hpp" "*.ui" "*.qrc")

add_executable(app ${SOURCES})

include_directories(${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(app PUBLIC Laszip::Laszip ranger)