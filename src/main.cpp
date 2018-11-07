/**
* @file      main.cpp
* @brief     Example points flocking simulation for CIS 565
* @authors   Liam Boone, Kai Ninomiya, Kangning (Gary) Li
* @date      2013-2017
* @copyright University of Pennsylvania
*/

#include "main.hpp"

// ================
// Configuration
// ================

#define VISUALIZE 0

#define FREQ 1 // sample 1 pt from every FREQ pts in original
#define SEP ' '
const float DT = 0.2f;

PointCloud* pointcloud;

static int N;

/**
* C main function.
*/
int main(int argc, char* argv[]) {
    projectName = "565 CUDA Project 5: Point Cloud Registration, ICP";
    if (argc != 2) {
        cout << "Usage: [pc file]. Press Enter to exit" << endl;
        getchar();
        return 0;
    }

    string input_filename(argv[1]);
    string ext = utilityCore::getFilePathExtension(input_filename);

    if (ext.compare("txt") == 0) {
        pointcloud = new PointCloud(input_filename, FREQ, SEP);
		N = pointcloud->getNumPoints();
    } else {
        printf("Non Supported pc Format\n");
        return -1;
    }

    if (init(argc, argv)) {
        mainLoop();
        registrationFree();
        return 0;
    } else {
        return 1;
    }
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

/**
* Initialization of CUDA and GLFW.
*/
bool init(int argc, char **argv) {
    // Set window title to "Student Name: [SM 2.0] GPU Name"
    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (gpuDevice > device_count) {
        std::cout
                << "Error: GPU device number is greater than the number of devices!"
                << " Perhaps a CUDA-capable GPU is not installed?"
                << std::endl;
        return false;
    }
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    int major = deviceProp.major;
    int minor = deviceProp.minor;

    std::ostringstream ss;
    ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
    deviceName = ss.str();

    // Window setup stuff
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        std::cout
                << "Error: Could not initialize GLFW!"
                << " Perhaps OpenGL 3.3 isn't available?"
                << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    // Initialize drawing state
    initVAO();

    // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
    // change the device ID.
    cudaGLSetGLDevice(0);

    cudaGLRegisterBufferObject(pointVBO_positions);
    cudaGLRegisterBufferObject(pointVBO_velocities);

    // Initialize N-body simulation
	
    registrationInit(pointcloud->getPoints());

    updateCamera();

    initShaders(program);

    glEnable(GL_DEPTH_TEST);

    return true;
}

void initVAO() {
    std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (2 * N)] };
    std::unique_ptr<GLuint[]> bindices{ new GLuint[2 * N] };

    glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
    glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

    for (int i = 0; i < 2 * N; i++) {
        bodies[4 * i + 0] = 0.0f;
        bodies[4 * i + 1] = 0.0f;
        bodies[4 * i + 2] = 0.0f;
        bodies[4 * i + 3] = 1.0f;
        bindices[i] = i;
    }


    glGenVertexArrays(1, &pointVAO); // Attach everything needed to draw a particle to this
    glGenBuffers(1, &pointVBO_positions);
    glGenBuffers(1, &pointVBO_velocities);
    glGenBuffers(1, &pointIBO);

    glBindVertexArray(pointVAO);

    // Bind the positions array to the pointVAO by way of the pointVBO_positions
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO_positions); // bind the buffer
    glBufferData(GL_ARRAY_BUFFER, 4 * (2 * N) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

    glEnableVertexAttribArray(positionLocation);
    glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

    // Bind the velocities array to the pointVAO by way of the pointVBO_velocities
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO_velocities);
    glBufferData(GL_ARRAY_BUFFER, 4 * (2 * N) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(velocitiesLocation);
    glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pointIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (2 * N) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void initShaders(GLuint * program) {
    GLint location;

	program[PROG_POINT] = glslUtility::createProgram(
		"../shaders/boid.vert.glsl",
		"../shaders/boid.geom.glsl",
		"../shaders/boid.frag.glsl", attributeLocations, 2);
	glUseProgram(program[PROG_POINT]);

    if ((location = glGetUniformLocation(program[PROG_POINT], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG_POINT], "u_cameraPos")) != -1) {
        glUniform3fv(location, 1, &cameraPosition[0]);
    }
}

//====================================
// Main loop
//====================================
void runCUDA() {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
    // use this buffer

    float4 *dptr = NULL;
    float *dptrVertPositions = NULL;
    float *dptrVertVelocities = NULL;

    cudaGLMapBufferObject((void**)&dptrVertPositions, pointVBO_positions);
    cudaGLMapBufferObject((void**)&dptrVertVelocities, pointVBO_velocities);

    // execute the kernel
    registration();
	//registration_cpu(pointcloud->getPoints(), pointcloud->getPoints());
#if VISUALIZE
    copyPointsToVBO(dptrVertPositions, dptrVertVelocities);
#endif
    // unmap buffer object
    cudaGLUnmapBufferObject(pointVBO_positions);
    cudaGLUnmapBufferObject(pointVBO_velocities);
}

void mainLoop() {
    double fps = 0;
    double timebase = 0;
    int frame = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        frame++;
        double time = glfwGetTime();

        if (time - timebase > 1.0) {
            fps = frame / (time - timebase);
            timebase = time;
            frame = 0;
        }

        runCUDA();

        std::ostringstream ss;
        ss << "[";
        ss.precision(1);
        ss << std::fixed << fps;
        ss << " fps]  N: " << N;
        glfwSetWindowTitle(window, ss.str().c_str());

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#if VISUALIZE
        glUseProgram(program[PROG_POINT]);
        glBindVertexArray(pointVAO);
        glPointSize((GLfloat)pointSize);
        glDrawElements(GL_POINTS, 2 * N + 1, GL_UNSIGNED_INT, 0);
        glPointSize(1.0f);

        glUseProgram(0);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
#endif
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}


void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (leftMousePressed) {
        // compute new camera parameters
        phi += (xpos - lastX) / width;
        theta -= (ypos - lastY) / height;
        theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
        updateCamera();
    }
    else if (rightMousePressed) {
        zoom += (ypos - lastY) / height;
        zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
        updateCamera();
    }else if (middleMousePressed){
        glm::vec3 forward = -glm::normalize(cameraPosition);
        forward.y = 0.0f;
        forward = glm::normalize(forward);
        glm::vec3 right = glm::cross(forward, glm::vec3(0, 1, 0));
        right.y = 0.0f;
        right = glm::normalize(right);

        lookAt -= (float) (xpos - lastX) * right * 0.01f;
        lookAt += (float) (ypos - lastY) * forward * 0.01f;
        updateCamera();
    }

    lastX = xpos;
    lastY = ypos;
}

void updateCamera() {
    cameraPosition.x = zoom * sin(phi) * sin(theta);
    cameraPosition.z = zoom * cos(theta);
    cameraPosition.y = zoom * cos(phi) * sin(theta);
    cameraPosition += lookAt;
    cout << lookAt.x << ", " << lookAt.y << ", " << lookAt.z << "," << endl;

    projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
    glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
    projection = projection * view;

    GLint location;

    glUseProgram(program[PROG_POINT]);
    if ((location = glGetUniformLocation(program[PROG_POINT], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
}