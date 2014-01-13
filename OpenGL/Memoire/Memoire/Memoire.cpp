// ---------------------------------------------------------------------------
//
// ESGI OpenGL (ES) 2.0 Framework
// Malek Bengougam, 2012							malek.bengougam@gmail.com
//
// ---------------------------------------------------------------------------

// --- Includes --------------------------------------------------------------

#include "../../EsgiGL/EsgiGL.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>

#include "../../EsgiGL/Common/vector.h"
#include "../../EsgiGL/Common/matrix.h"
#include "../../EsgiGL/EsgiShader.h"

#include "PMC.h"

using namespace std;

// --- Globales --------------------------------------------------------------

static vector<vec3> listMousePoints;

static int sizeWindowX = 600;
static int sizeWindowY = 600;

EsgiShader shader;

struct EsgiCamera
{
	vec3 position;
	vec3 target;
	vec3 orientation;
};
EsgiCamera camera;

// --- Fonctions -------------------------------------------------------------

// ---

void Update(float elapsedTime)
{
}

void Draw()
{
	// efface le color buffer et le depth buffer
	glClearColor(1.f, 1.f, 1.f, 1.f);
	glClearDepth(1.f);
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

#ifndef ESGI_GLES_20
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
#endif

	// ---
	GLuint programObject = shader.GetProgram();
	glUseProgram(programObject);
	// alternativement
	// shader.Bind();
	// ---

	GLint position_attrib = glGetAttribLocation(programObject, "a_Position");
	glVertexAttribPointer(position_attrib, 3, GL_FLOAT, false, 0, listMousePoints.data());
	
	GLint color_uniform = glGetUniformLocation(programObject, "u_Color");
	glUniform4f(color_uniform, 0.f, 0.f, 0.f, 1.f);
	
	mat4 projectionMatrix = esgiPerspective(45, (float)sizeWindowX/sizeWindowY, 0.1f, 1000.f);
	GLint projectionUniform = glGetUniformLocation(programObject, "u_ProjectionMatrix");
	glUniformMatrix4fv(projectionUniform, 1, false, &projectionMatrix.I.x);
	
	mat4 cameraMatrix = esgiLookAt(camera.position, camera.target, vec3(0.0, 1.0, 0.0));
	mat4 viewMatrix = esgiMultiplyMatrix(cameraMatrix, esgiRotateY(camera.orientation.y));
	GLuint viewUniform = glGetUniformLocation(programObject, "u_ViewMatrix");
	glUniformMatrix4fv(viewUniform, 1, false, &viewMatrix.I.x);

	mat4 worldMatrix;
	worldMatrix.Identity();
	worldMatrix.T.set(0.f, 0.f, -700.f, 1.f);
	GLint worldUniform = glGetUniformLocation(programObject, "u_WorldMatrix");
	glUniformMatrix4fv(worldUniform, 1, 0, &worldMatrix.I.x);

	glEnableVertexAttribArray(position_attrib);
	glDrawArrays(GL_POINTS, 0, listMousePoints.size() - 1);
	glDisableVertexAttribArray(position_attrib);

	// alternativement
	// shader.Unbind();
}

//
// Initialise les shader & mesh
//
bool Setup()
{
	shader.LoadVertexShader("basic.vert");
	shader.LoadFragmentShader("basic.frag");
	shader.Create();

	camera.position = vec3(0.f, 0.f, 10.f);
	camera.orientation = vec3(0.f, 0.f, 0.f);
	camera.target = vec3(0.f, 0.f, 0.f);

	vector<int> sizePMC;
	sizePMC.push_back(3);
	sizePMC.push_back(2);
	sizePMC.push_back(1);

	vector<TrainingStruct> listTraining;

	TrainingStruct trainingStruct0;
	trainingStruct0.data.push_back(0);
	trainingStruct0.data.push_back(0);
	trainingStruct0.data.push_back(1);
	trainingStruct0.result.push_back(0);
	listTraining.push_back(trainingStruct0);
	
	TrainingStruct trainingStruct1;
	trainingStruct1.data.push_back(0);
	trainingStruct1.data.push_back(1);
	trainingStruct1.data.push_back(1);
	trainingStruct1.result.push_back(1);
	listTraining.push_back(trainingStruct1);

	TrainingStruct trainingStruct2;
	trainingStruct2.data.push_back(1);
	trainingStruct2.data.push_back(0);
	trainingStruct2.data.push_back(1);
	trainingStruct2.result.push_back(1);
	listTraining.push_back(trainingStruct2);

	TrainingStruct trainingStruct3;
	trainingStruct3.data.push_back(1);
	trainingStruct3.data.push_back(1);
	trainingStruct3.data.push_back(1);
	trainingStruct3.result.push_back(0);
	listTraining.push_back(trainingStruct3);

	PMC perceptron(sizePMC);
	perceptron.LaunchLearning(10000, 0.1, true, listTraining);
	
	return true;
}

//
// Libere la memoire occupee
//
void Clean()
{
	shader.Destroy();
}

//
// Fonction d'event de la souris
//
void ActiveMouse(int mousex, int mousey)
{
	float tempX = mousex - sizeWindowX/2, tempY = (mousey - sizeWindowY/2) * -1;
	cout << "tempX " << tempX << " tempY " << tempY << endl;
	listMousePoints.push_back(vec3(tempX, tempY, 0));
}
void PassiveMouse(int mousex, int mousey)
{
}

void Keyboard(unsigned char key, int mx, int my)
{
	switch(key)
	{
	case 'c':
		cout << "Clear" << endl;
		listMousePoints.clear();
		break;
	}
}

// 
//
//
int main(int argc, char *argv[])
{
	EsgiGLApplication esgi;
    
	esgi.InitWindowPosition(0, 0);
	esgi.InitWindowSize(sizeWindowX, sizeWindowY);
	esgi.InitDisplayMode(ESGI_WINDOW_RGBA|ESGI_WINDOW_DEPTH|ESGI_WINDOW_DOUBLEBUFFER);
	esgi.CreateWindow("M�moire", ESGI_WINDOW_CENTERED);
	
    esgi.IdleFunc(&Update);
	esgi.DisplayFunc(&Draw);
    esgi.InitFunc(&Setup);
    esgi.CleanFunc(&Clean);
	esgi.MotionFunc(&ActiveMouse);
	esgi.PassiveMotionFunc(&PassiveMouse);
	esgi.KeyboardFunction(&Keyboard);
    
	esgi.MainLoop();
    
    return 0;
}