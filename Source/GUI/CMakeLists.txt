cmake_minimum_required(VERSION 3.10)

set(LIB_NAMES GlutGUI QtGUI)

include_directories("${CMAKE_SOURCE_DIR}/Source")                                                              #默认为本项目下各编译目标添加Source/ 目录

add_subdirectory(GlutGUI)

if(PhysIKA_Qt5_GUI)
    add_subdirectory(QtGUI)
endif()


file(GLOB FILE_GUI_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${FILE_GUI_HEADER}  DESTINATION ${PHYSIKA_INC_INSTALL_DIR}/GUI)